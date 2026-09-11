"""Trust model and pairing-record stores."""

from __future__ import annotations

import asyncio
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final

from aiosendspin.models.types import PairMethod, TrustLevel

from .keys import (
    PSK_SIZE,
    b64url_decode,
    b64url_encode,
    generate_psk,
    psk_id_for,
)
from .pairing_code import is_valid_static_pairing_code

# Dynamic pairing-code pairing escalates to gesture-gating when its failure counter reaches
# this value.
PAIRING_CODE_ESCALATION_THRESHOLD: Final[int] = 5

__all__ = [
    "PAIRING_CODE_ESCALATION_THRESHOLD",
    "ClientPairingConfig",
    "ClientPairingRecord",
    "ClientPairingStore",
    "FileClientPairingStore",
    "FileServerPairingStore",
    "InMemoryClientPairingStore",
    "InMemoryServerPairingStore",
    "PairingPsk",
    "PskCategory",
    "ResolvedPsk",
    "ServerPairingRecord",
    "ServerPairingStore",
    "StagedPairingPsk",
    "StorageExhaustedError",
    "StorageReport",
    "TrustLevel",
    "TrustedUnpairedClient",
]


class PskCategory(StrEnum):
    """Which kind of PSK was matched during a handshake."""

    LONG_TERM = "long_term"
    """A per-pair long-term PSK established through a successful pairing."""
    PAIRING = "pairing"
    """A Pairing PSK distributed out-of-band to admit a new client."""
    SENTINEL = "sentinel"
    """The published Sentinel PSK — used for pairing-code pairing and unpaired playback."""

    @property
    def code(self) -> str:
        """The two-letter identifier this category travels under in Noise message 1."""
        return _PSK_CATEGORY_CODES[self]

    @classmethod
    def from_code(cls, code: str) -> PskCategory | None:
        """Return the category a Noise message 1 code names, or None if it names none."""
        return _PSK_CATEGORIES_BY_CODE.get(code)


# The wire codes are deliberately equal-length, so the encrypted payload's length does not
# reveal which category the server referenced.
_PSK_CATEGORY_CODES: dict[PskCategory, str] = {
    PskCategory.LONG_TERM: "lt",
    PskCategory.PAIRING: "pr",
    PskCategory.SENTINEL: "sn",
}
_PSK_CATEGORIES_BY_CODE: dict[str, PskCategory] = {c: k for k, c in _PSK_CATEGORY_CODES.items()}


class StorageExhaustedError(Exception):
    """A pairing cannot persist its record and has no shared-PSK fallback."""


@dataclass(frozen=True, slots=True)
class StorageReport:
    """A bounded client's record-storage accounting."""

    capacity: int
    free: int
    cost_individual: int
    cost_shared: int


@dataclass(frozen=True, slots=True)
class ResolvedPsk:
    """A PSK selected during a handshake, with its trust metadata."""

    psk_id: str
    psk: bytes
    category: PskCategory
    counterparty_id: str | None = None
    """Peer's ``client_id``/``server_id`` for stored-pubkey records; ``None`` otherwise."""


@dataclass(frozen=True, slots=True)
class ServerPairingRecord:
    """A long-term credential a server stores for one client."""

    psk_id: str
    psk: bytes
    client_id: str
    pair_methods: list[PairMethod]
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    owner: str | None = None
    """Application-defined id of the authorization this record is bound to.

    Owned records are meant to be removed once the owning authorization (e.g. a user
    account or session) ends; the server attaches no behavior to this field. ``None``
    marks a self-standing credential.
    """

    def __post_init__(self) -> None:
        """Validate the PSK size."""
        _check_psk(self.psk)

    def as_resolved(self) -> ResolvedPsk:
        """Project to the handshake-time currency (category ``long_term``)."""
        return ResolvedPsk(self.psk_id, self.psk, PskCategory.LONG_TERM, self.client_id)

    def with_method(self, method: PairMethod) -> ServerPairingRecord:
        """Return a copy with ``method`` appended (unchanged if already present)."""
        if method in self.pair_methods:
            return self
        return replace(self, pair_methods=[*self.pair_methods, method])

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dict (PSK base64url, timestamp ISO-8601)."""
        return {
            "psk_id": self.psk_id,
            "psk": b64url_encode(self.psk),
            "client_id": self.client_id,
            "pair_methods": [m.value for m in self.pair_methods],
            "created_at": self.created_at.isoformat(),
            "owner": self.owner,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ServerPairingRecord:
        """Reconstruct a record from ``to_dict`` output."""
        return cls(
            psk_id=_str(data, "psk_id"),
            psk=b64url_decode(_str(data, "psk")),
            client_id=_str(data, "client_id"),
            pair_methods=_pair_methods(data, "pair_methods"),
            created_at=datetime.fromisoformat(_str(data, "created_at")),
            owner=_opt_str(data, "owner"),
        )


@dataclass(frozen=True, slots=True)
class ClientPairingRecord:
    """A long-term credential a client stores for a server."""

    psk_id: str
    psk: bytes
    server_id: str | None = None
    used: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        """Validate the PSK size."""
        _check_psk(self.psk)

    def as_resolved(self) -> ResolvedPsk:
        """Project to the handshake-time currency (category ``long_term``)."""
        return ResolvedPsk(self.psk_id, self.psk, PskCategory.LONG_TERM, self.server_id)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dict (PSK base64url, timestamp ISO-8601)."""
        return {
            "psk_id": self.psk_id,
            "psk": b64url_encode(self.psk),
            "server_id": self.server_id,
            "used": self.used,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ClientPairingRecord:
        """Reconstruct a record from ``to_dict`` output."""
        return cls(
            psk_id=_str(data, "psk_id"),
            psk=b64url_decode(_str(data, "psk")),
            server_id=_opt_str(data, "server_id"),
            used=_bool(data, "used"),
            created_at=datetime.fromisoformat(_str(data, "created_at")),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ClientPairingConfig:
    """Pairing policy a client persists."""

    pairing_psk_enabled: bool = True
    dynamic_pairing_code_enabled: bool = True
    static_pairing_code_enabled: bool = False
    unpaired_access_enabled: bool = False
    record_mode_psk_id: str
    """Shared-PSK record used as the storage-exhaustion fallback when pairing."""

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dict."""
        return {
            "pairing_psk_enabled": self.pairing_psk_enabled,
            "dynamic_pin_enabled": self.dynamic_pairing_code_enabled,
            "static_pin_enabled": self.static_pairing_code_enabled,
            "unpaired_access_enabled": self.unpaired_access_enabled,
            "record_mode_psk_id": self.record_mode_psk_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ClientPairingConfig:
        """Reconstruct from ``to_dict`` output (defaults for absent keys)."""
        return cls(
            pairing_psk_enabled=_bool(data, "pairing_psk_enabled", default=True),
            dynamic_pairing_code_enabled=_bool(data, "dynamic_pin_enabled", default=True),
            static_pairing_code_enabled=_bool(data, "static_pin_enabled", default=False),
            unpaired_access_enabled=_bool(data, "unpaired_access_enabled", default=False),
            record_mode_psk_id=_str(data, "record_mode_psk_id"),
        )


@dataclass(frozen=True, slots=True)
class PairingPsk:
    """A Pairing PSK a client accepts to admit a server."""

    psk_id: str
    psk: bytes

    def __post_init__(self) -> None:
        """Validate the PSK size."""
        _check_psk(self.psk)

    def as_resolved(self) -> ResolvedPsk:
        """Project to the handshake-time currency (category ``pairing``)."""
        return ResolvedPsk(self.psk_id, self.psk, PskCategory.PAIRING, None)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dict (PSK base64url)."""
        return {"psk_id": self.psk_id, "psk": b64url_encode(self.psk)}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> PairingPsk:
        """Reconstruct a Pairing PSK from ``to_dict`` output."""
        return cls(
            psk_id=_str(data, "psk_id"),
            psk=b64url_decode(_str(data, "psk")),
        )


@dataclass(frozen=True, slots=True)
class StagedPairingPsk:
    """An operator-staged Pairing PSK awaiting a client, with its staging time."""

    psk_id: str
    psk: bytes
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        """Validate the PSK size."""
        _check_psk(self.psk)

    def as_resolved(self) -> ResolvedPsk:
        """Project to the handshake-time currency (category ``pairing``)."""
        return ResolvedPsk(self.psk_id, self.psk, PskCategory.PAIRING, None)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dict (PSK base64url, timestamp ISO-8601)."""
        return {
            "psk_id": self.psk_id,
            "psk": b64url_encode(self.psk),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> StagedPairingPsk:
        """Reconstruct from ``to_dict`` output."""
        return cls(
            psk_id=_str(data, "psk_id"),
            psk=b64url_decode(_str(data, "psk")),
            created_at=datetime.fromisoformat(_str(data, "created_at")),
        )


@dataclass(frozen=True, slots=True)
class TrustedUnpairedClient:
    """A ``client_id`` the operator approved for unpaired playback."""

    client_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dict (timestamp ISO-8601)."""
        return {
            "client_id": self.client_id,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> TrustedUnpairedClient:
        """Reconstruct from ``to_dict`` output."""
        return cls(
            client_id=_str(data, "client_id"),
            created_at=datetime.fromisoformat(_str(data, "created_at")),
        )


class ServerPairingStore(ABC):
    """Long-term records, operator-staged Pairing PSKs, and trusted-unpaired clients.

    Every entry is keyed by ``client_id``.
    """

    @abstractmethod
    async def record_by_client_id(self, client_id: str) -> ServerPairingRecord | None:
        """Return the long-term record for ``client_id``, if any."""

    @abstractmethod
    async def list_records(self) -> Sequence[ServerPairingRecord]:
        """Return all stored long-term records."""

    @abstractmethod
    async def store_record(self, record: ServerPairingRecord) -> None:
        """Persist a long-term record (keyed by its ``client_id``)."""

    @abstractmethod
    async def remove_record(self, client_id: str) -> None:
        """Remove the long-term record for ``client_id`` (no-op if absent)."""

    @abstractmethod
    async def staged_pairing_psk(self, client_id: str) -> StagedPairingPsk | None:
        """Return the Pairing PSK staged to admit ``client_id`` for pairing, if any."""

    @abstractmethod
    async def list_staged_pairing_psks(self) -> Sequence[StagedPairingPsk]:
        """Return all operator-staged Pairing PSKs."""

    @abstractmethod
    async def stage_pairing_psk(self, client_id: str, staged: StagedPairingPsk) -> None:
        """Stage an operator-entered Pairing PSK to admit ``client_id`` for pairing."""

    @abstractmethod
    async def unstage_pairing_psk(self, client_id: str) -> None:
        """Remove the staged Pairing PSK for ``client_id`` (no-op if absent)."""

    @abstractmethod
    async def trusted_unpaired(self, client_id: str) -> TrustedUnpairedClient | None:
        """Return the trusted-unpaired approval for ``client_id``, if any."""

    @abstractmethod
    async def list_trusted_unpaired(self) -> Sequence[TrustedUnpairedClient]:
        """Return all trusted-unpaired approvals."""

    @abstractmethod
    async def add_trusted_unpaired(self, client: TrustedUnpairedClient) -> None:
        """Approve a client for unpaired playback."""

    @abstractmethod
    async def remove_trusted_unpaired(self, client_id: str) -> None:
        """Revoke client's unpaired-playback approval (no-op if absent)."""

    async def records_by_owner(self, owner: str) -> Sequence[ServerPairingRecord]:
        """Return all long-term records bound to ``owner``."""
        return [record for record in await self.list_records() if record.owner == owner]


class ClientPairingStore(ABC):
    """Pairing state a client holds: long-term records plus its accepted Pairing PSKs."""

    @abstractmethod
    async def resolve_by_psk_id(self, psk_id: str) -> ResolvedPsk | None:
        """Resolve a ``psk_id`` to its PSK for the handshake, or ``None``."""

    @abstractmethod
    async def record_by_psk_id(self, psk_id: str) -> ClientPairingRecord | None:
        """Return the long-term record identified by ``psk_id``, if any."""

    @abstractmethod
    async def record_by_server_id(self, server_id: str) -> ClientPairingRecord | None:
        """Return the stored-pubkey record bound to ``server_id``, if any."""

    @abstractmethod
    async def store_record(self, record: ClientPairingRecord) -> None:
        """Persist a long-term record."""

    @abstractmethod
    async def remove_record(self, psk_id: str) -> None:
        """Remove the long-term record identified by ``psk_id`` (no-op if absent)."""

    @abstractmethod
    async def mark_record_used(self, psk_id: str) -> None:
        """Flag the record at ``psk_id`` as used (no-op if absent)."""

    @abstractmethod
    async def list_records(self) -> Sequence[ClientPairingRecord]:
        """Return all stored long-term records."""

    @abstractmethod
    async def get_pairing_config(self) -> ClientPairingConfig:
        """Return the persisted pairing policy (defaults if unset)."""

    @abstractmethod
    async def store_pairing_config(self, config: ClientPairingConfig) -> None:
        """Persist the pairing policy."""

    @abstractmethod
    async def set_pairing_psk(self, pairing_psk: PairingPsk) -> None:
        """Set the accepted Pairing PSK, replacing any existing one (admits a server with it)."""

    @abstractmethod
    async def clear_pairing_psk(self) -> None:
        """Remove the accepted Pairing PSK (no-op if absent)."""

    @abstractmethod
    async def pairing_psk(self) -> PairingPsk | None:
        """Return the accepted Pairing PSK, if any."""

    @abstractmethod
    async def set_static_pairing_code(self, pairing_code: str) -> None:
        """Set the configured static pairing code (8 decimal digits), replacing any existing one."""

    @abstractmethod
    async def clear_static_pairing_code(self) -> None:
        """Remove the configured static pairing code (no-op if absent)."""

    @abstractmethod
    async def static_pairing_code(self) -> str | None:
        """Return the configured static pairing code, if any."""

    @abstractmethod
    async def pairing_code_failure_count(self) -> int:
        """Return the persisted dynamic-pairing-code failure count."""

    @abstractmethod
    async def record_pairing_code_failure(self) -> int:
        """Increment the dynamic-pairing-code failure counter and return the new count."""

    @abstractmethod
    async def reset_pairing_code_failures(self) -> None:
        """Reset the dynamic-pairing-code failure counter to zero (on ``server_kc`` success)."""

    @abstractmethod
    async def is_pairing_code_escalated(self) -> bool:
        """Return whether dynamic pairing code is escalated to gesture-gating.

        Escalation begins when the failure count reaches the threshold.
        """

    @abstractmethod
    async def get_last_playback_server_id(self) -> str | None:
        """Return the persisted last-playback server id, if one is stored."""

    @abstractmethod
    async def set_last_playback_server_id(self, server_id: str | None) -> None:
        """Persist the last-playback server id."""

    async def can_store_record(self) -> bool:
        """Return whether the store can persist another record (default: unlimited)."""
        return True

    async def storage_accounting(self) -> StorageReport | None:
        """Return record-storage accounting, or ``None`` if storage is unbounded/unknown."""
        return None

    async def resolve_pairing_outcome(
        self,
        *,
        server_id: str,
    ) -> tuple[bytes, ClientPairingRecord | None]:
        """Decide a pairing's outcome: a fresh per-server record, or the shared-PSK fallback."""
        if await self.can_store_record():
            psk = generate_psk()
            record = ClientPairingRecord(
                psk_id=psk_id_for(psk),
                psk=psk,
                server_id=server_id,
            )
            return psk, record
        # Storage exhausted: admit under the shared-PSK fallback record.
        psk_id = (await self.get_pairing_config()).record_mode_psk_id
        resolved = await self.resolve_by_psk_id(psk_id)
        if resolved is None or not _is_shared_record(resolved):
            msg = f"shared-PSK fallback record {psk_id!r} is missing or not shared"
            raise StorageExhaustedError(msg)
        return resolved.psk, None

    async def set_record_mode_psk_id(self, psk_id: str) -> None:
        """Set the shared-PSK fallback record; ``psk_id`` must name a shared record."""
        resolved = await self.resolve_by_psk_id(psk_id)
        if resolved is None:
            msg = f"record_mode psk_id {psk_id!r} references no record"
            raise ValueError(msg)
        if not _is_shared_record(resolved):
            msg = f"record_mode psk_id {psk_id!r} must reference a shared-PSK record"
            raise ValueError(msg)
        config = await self.get_pairing_config()
        await self.store_pairing_config(replace(config, record_mode_psk_id=psk_id))

    async def _record_mode_references(self, psk_id: str) -> bool:
        """Return whether the record_mode fallback references ``psk_id``."""
        return (await self.get_pairing_config()).record_mode_psk_id == psk_id

    async def can_remove_record(self, psk_id: str) -> bool:
        """Return whether the record at ``psk_id`` may be removed (not record_mode-referenced)."""
        return not await self._record_mode_references(psk_id)

    async def replace_record_for_server_id(self, record: ClientPairingRecord) -> None:
        """Persist ``record``, dropping any prior removable record bound to the same server."""
        stale = (
            [
                existing.psk_id
                for existing in await self.list_records()
                if existing.server_id == record.server_id and existing.psk_id != record.psk_id
            ]
            if record.server_id is not None
            else []
        )
        await self.store_record(record)
        for psk_id in stale:
            if await self.can_remove_record(psk_id):
                await self.remove_record(psk_id)


class _ServerPairingStoreBase(ServerPairingStore):
    """Shared query/mutation logic for server pairing stores; subclasses add persistence."""

    def __init__(self) -> None:
        """Start with empty record, staged-Pairing-PSK, and trusted-unpaired tables."""
        self._records: dict[str, ServerPairingRecord] = {}
        self._staged: dict[str, StagedPairingPsk] = {}
        self._trusted: dict[str, TrustedUnpairedClient] = {}

    async def _save(self) -> None:
        """Flush mutated state to durable storage; a no-op for non-persistent stores."""

    async def record_by_client_id(self, client_id: str) -> ServerPairingRecord | None:
        """Return the long-term record for ``client_id``, if any."""
        return self._records.get(client_id)

    async def list_records(self) -> Sequence[ServerPairingRecord]:
        """Return all stored long-term records."""
        return list(self._records.values())

    async def store_record(self, record: ServerPairingRecord) -> None:
        """Persist a long-term record keyed by its ``client_id``."""
        self._records[record.client_id] = record
        await self._save()

    async def remove_record(self, client_id: str) -> None:
        """Remove the long-term record for ``client_id`` (no-op if absent)."""
        if self._records.pop(client_id, None) is not None:
            await self._save()

    async def staged_pairing_psk(self, client_id: str) -> StagedPairingPsk | None:
        """Return the Pairing PSK staged to admit ``client_id``, if any."""
        return self._staged.get(client_id)

    async def list_staged_pairing_psks(self) -> Sequence[StagedPairingPsk]:
        """Return all operator-staged Pairing PSKs."""
        return list(self._staged.values())

    async def stage_pairing_psk(self, client_id: str, pairing_psk: StagedPairingPsk) -> None:
        """Stage an operator-entered Pairing PSK to admit ``client_id``."""
        self._staged[client_id] = pairing_psk
        await self._save()

    async def unstage_pairing_psk(self, client_id: str) -> None:
        """Remove the staged Pairing PSK for ``client_id`` (no-op if absent)."""
        if self._staged.pop(client_id, None) is not None:
            await self._save()

    async def trusted_unpaired(self, client_id: str) -> TrustedUnpairedClient | None:
        """Return the trusted-unpaired approval for ``client_id``, if any."""
        return self._trusted.get(client_id)

    async def list_trusted_unpaired(self) -> Sequence[TrustedUnpairedClient]:
        """Return all trusted-unpaired approvals."""
        return list(self._trusted.values())

    async def add_trusted_unpaired(self, client: TrustedUnpairedClient) -> None:
        """Approve ``client`` for unpaired playback keyed by its ``client_id``."""
        self._trusted[client.client_id] = client
        await self._save()

    async def remove_trusted_unpaired(self, client_id: str) -> None:
        """Revoke ``client_id``'s unpaired-playback approval (no-op if absent)."""
        if self._trusted.pop(client_id, None) is not None:
            await self._save()


class InMemoryServerPairingStore(_ServerPairingStoreBase):
    """In-memory reference ``ServerPairingStore`` (tests, ephemeral servers); not persisted."""


class FileServerPairingStore(_ServerPairingStoreBase):
    """A ``ServerPairingStore`` persisted atomically to a single JSON file."""

    def __init__(self, path: str | Path) -> None:
        """Internal-only; call ``open()`` to load the store instead."""
        super().__init__()
        self._path = Path(path)
        self._lock = asyncio.Lock()

    @classmethod
    async def open(cls, path: str | Path) -> FileServerPairingStore:
        """Load the store."""
        store = cls(path)
        await store._load()
        return store

    async def _load(self) -> None:
        """Populate state from the JSON file, if present."""
        data = await asyncio.to_thread(_read_json_object, self._path)
        if data is None:
            return
        self._records = {
            cid: ServerPairingRecord.from_dict(v) for cid, v in _section(data, "records").items()
        }
        self._staged = {
            cid: StagedPairingPsk.from_dict(v)
            for cid, v in _section(data, "staged_pairing_psks").items()
        }
        self._trusted = {
            cid: TrustedUnpairedClient.from_dict(v)
            for cid, v in _section(data, "trusted_unpaired_clients").items()
        }

    async def _save(self) -> None:
        """Atomically write the current state to the JSON file."""
        async with self._lock:
            payload = {
                "records": {c: r.to_dict() for c, r in self._records.items()},
                "staged_pairing_psks": {c: p.to_dict() for c, p in self._staged.items()},
                "trusted_unpaired_clients": {c: t.to_dict() for c, t in self._trusted.items()},
            }
            await asyncio.to_thread(_atomic_write_json, self._path, payload)


class _ClientPairingStoreBase(ClientPairingStore):
    """Shared query/mutation logic for client pairing stores; subclasses add persistence."""

    def __init__(self) -> None:
        """Start with empty state; subclasses provision the shared-PSK fallback record."""
        self._records: dict[str, ClientPairingRecord] = {}
        self._pairing_psk: PairingPsk | None = None
        self._static_pairing_code: str | None = None
        self._pin_failures = 0
        self._pairing_config: ClientPairingConfig | None = None
        self._last_playback_server_id: str | None = None

    async def _save(self) -> None:
        """Flush mutated state to durable storage; a no-op for non-persistent stores."""

    async def get_last_playback_server_id(self) -> str | None:
        """Return the persisted last-playback server id, if any."""
        return self._last_playback_server_id

    async def set_last_playback_server_id(self, server_id: str | None) -> None:
        """Persist the last-playback server id."""
        if server_id == self._last_playback_server_id:
            return
        previous_server_id = self._last_playback_server_id
        self._last_playback_server_id = server_id
        try:
            await self._save()
        except BaseException:
            self._last_playback_server_id = previous_server_id
            raise

    async def resolve_by_psk_id(self, psk_id: str) -> ResolvedPsk | None:
        """Resolve a ``psk_id`` (long-term record first, then the accepted Pairing PSK)."""
        record = self._records.get(psk_id)
        if record is not None:
            return record.as_resolved()
        if self._pairing_psk is not None and self._pairing_psk.psk_id == psk_id:
            return self._pairing_psk.as_resolved()
        return None

    async def record_by_psk_id(self, psk_id: str) -> ClientPairingRecord | None:
        """Return the long-term record identified by ``psk_id`` (O(1))."""
        return self._records.get(psk_id)

    async def record_by_server_id(self, server_id: str) -> ClientPairingRecord | None:
        """Return the stored-pubkey record bound to ``server_id`` (linear scan)."""
        for record in self._records.values():
            if record.server_id == server_id:
                return record
        return None

    async def store_record(self, record: ClientPairingRecord) -> None:
        """Persist a long-term record keyed by its ``psk_id``."""
        self._records[record.psk_id] = record
        await self._save()

    async def remove_record(self, psk_id: str) -> None:
        """Remove the long-term record identified by ``psk_id`` (no-op if absent)."""
        if self._records.pop(psk_id, None) is not None:
            await self._save()

    async def mark_record_used(self, psk_id: str) -> None:
        """Flag the record at ``psk_id`` as used (no-op if absent or already used)."""
        record = self._records.get(psk_id)
        if record is not None and not record.used:
            self._records[psk_id] = replace(record, used=True)
            await self._save()

    async def list_records(self) -> Sequence[ClientPairingRecord]:
        """Return all stored long-term records."""
        return list(self._records.values())

    async def get_pairing_config(self) -> ClientPairingConfig:
        """Return the pairing policy."""
        assert self._pairing_config is not None, "store is not initialized"
        return self._pairing_config

    async def store_pairing_config(self, config: ClientPairingConfig) -> None:
        """Persist the pairing policy."""
        self._pairing_config = config
        await self._save()

    async def set_pairing_psk(self, pairing_psk: PairingPsk) -> None:
        """Set the accepted Pairing PSK, replacing any existing one."""
        self._pairing_psk = pairing_psk
        await self._save()

    async def clear_pairing_psk(self) -> None:
        """Remove the accepted Pairing PSK (no-op if absent)."""
        if self._pairing_psk is not None:
            self._pairing_psk = None
            await self._save()

    async def pairing_psk(self) -> PairingPsk | None:
        """Return the accepted Pairing PSK, if any."""
        return self._pairing_psk

    async def set_static_pairing_code(self, pairing_code: str) -> None:
        """Set the configured static pairing code, replacing any existing one."""
        if not is_valid_static_pairing_code(pairing_code):
            raise ValueError("static pairing code must be exactly 8 decimal digits")
        self._static_pairing_code = pairing_code
        await self._save()

    async def clear_static_pairing_code(self) -> None:
        """Remove the configured static pairing code (no-op if absent)."""
        if self._static_pairing_code is not None:
            self._static_pairing_code = None
            await self._save()

    async def static_pairing_code(self) -> str | None:
        """Return the configured static pairing code, if any."""
        return self._static_pairing_code

    async def pairing_code_failure_count(self) -> int:
        """Return the dynamic-pairing-code failure count."""
        return self._pin_failures

    async def record_pairing_code_failure(self) -> int:
        """Increment the dynamic-pairing-code failure counter and return the new count."""
        self._pin_failures += 1
        await self._save()
        return self._pin_failures

    async def reset_pairing_code_failures(self) -> None:
        """Reset the dynamic-pairing-code failure counter to zero (no-op if already zero)."""
        if self._pin_failures:
            self._pin_failures = 0
            await self._save()

    async def is_pairing_code_escalated(self) -> bool:
        """Return whether dynamic pairing code has escalated to gesture-gating."""
        return self._pin_failures >= PAIRING_CODE_ESCALATION_THRESHOLD


class InMemoryClientPairingStore(_ClientPairingStoreBase):
    """In-memory reference ``ClientPairingStore`` (tests, ephemeral clients); not persisted."""

    def __init__(self) -> None:
        """Start with a pre-provisioned shared-PSK fallback record and no other state."""
        super().__init__()
        shared_psk = generate_psk()
        shared = ClientPairingRecord(psk_id=psk_id_for(shared_psk), psk=shared_psk)
        self._records[shared.psk_id] = shared
        self._pairing_config = ClientPairingConfig(record_mode_psk_id=shared.psk_id)


class FileClientPairingStore(_ClientPairingStoreBase):
    """A ``ClientPairingStore`` persisted atomically to a single JSON file."""

    def __init__(self, path: str | Path) -> None:
        """Internal-only; call ``open()`` to load the store instead."""
        super().__init__()
        self._path = Path(path)
        self._lock = asyncio.Lock()

    @classmethod
    async def open(cls, path: str | Path) -> FileClientPairingStore:
        """Load the store."""
        store = cls(path)
        await store._load()
        return store

    async def _load(self) -> None:
        """Populate state from the JSON file, seeding a fresh store if absent."""
        data = await asyncio.to_thread(_read_json_object, self._path)
        if data is None:
            await self._seed()
            return
        self._records = {
            psk_id: ClientPairingRecord.from_dict(v)
            for psk_id, v in _section(data, "records").items()
        }
        self._pairing_config = ClientPairingConfig.from_dict(_object(data, "pairing_config"))
        raw_psk = data.get("pairing_psk")
        self._pairing_psk = PairingPsk.from_dict(raw_psk) if isinstance(raw_psk, Mapping) else None
        self._static_pairing_code = _opt_str(data, "static_pin")
        raw_failures = data.get("pin_failures", 0)
        if isinstance(raw_failures, Mapping):
            # Pre-escalation format kept per-method counters; carry over the
            # dynamic pairing-code counter.
            raw_failures = raw_failures.get(PairMethod.DYNAMIC_PAIRING_CODE.value, 0)
        if isinstance(raw_failures, bool) or not isinstance(raw_failures, int):
            msg = "pairing store 'pin_failures' must be an integer"
            raise TypeError(msg)
        self._pin_failures = raw_failures
        self._last_playback_server_id = _opt_str(data, "last_playback_server_id")

    async def _seed(self) -> None:
        """Provision the default pre-provisioned shared-PSK fallback record (spec §Record mode)."""
        shared_psk = generate_psk()
        shared = ClientPairingRecord(psk_id=psk_id_for(shared_psk), psk=shared_psk)
        self._records[shared.psk_id] = shared
        self._pairing_config = ClientPairingConfig(record_mode_psk_id=shared.psk_id)
        await self._save()

    async def _save(self) -> None:
        """Atomically write the current state to the JSON file."""
        async with self._lock:
            assert self._pairing_config is not None
            payload: dict[str, object] = {
                "records": {psk_id: r.to_dict() for psk_id, r in self._records.items()},
                "pairing_config": self._pairing_config.to_dict(),
                "pairing_psk": self._pairing_psk.to_dict() if self._pairing_psk else None,
                "static_pin": self._static_pairing_code,
                "pin_failures": self._pin_failures,
                "last_playback_server_id": self._last_playback_server_id,
            }
            await asyncio.to_thread(_atomic_write_json, self._path, payload)


# --- private helpers -----------------------------------------------------


def _is_shared_record(resolved: ResolvedPsk) -> bool:
    """Return whether ``resolved`` is a shared-PSK record (long-term, no counterparty)."""
    return resolved.category is PskCategory.LONG_TERM and resolved.counterparty_id is None


def _check_psk(psk: bytes) -> None:
    if len(psk) != PSK_SIZE:
        msg = f"PSK must be {PSK_SIZE} bytes, got {len(psk)}"
        raise ValueError(msg)


def _read_json_object(path: Path) -> Mapping[str, object] | None:
    """Read ``path`` as a JSON object; ``None`` if absent, ``TypeError`` if not an object."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    data = json.loads(raw)
    if not isinstance(data, Mapping):
        msg = f"pairing store {path} must contain a JSON object"
        raise TypeError(msg)
    return data


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Serialize ``payload`` and atomically replace ``path``, owner-readable only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        file.write(json.dumps(payload, indent=2))
    tmp.replace(path)


def _section(data: Mapping[str, object], key: str) -> dict[str, Mapping[str, object]]:
    if key not in data:
        return {}
    value = data[key]
    if not isinstance(value, Mapping):
        msg = f"pairing store section {key!r} must be an object, got {type(value).__name__}"
        raise TypeError(msg)
    entries: dict[str, Mapping[str, object]] = {}
    for entry_key, entry in value.items():
        if not isinstance(entry, Mapping):
            msg = f"entry {entry_key!r} in section {key!r} must be an object"
            raise TypeError(msg)
        entries[entry_key] = entry
    return entries


def _object(data: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        msg = f"pairing store field {key!r} must be an object"
        raise TypeError(msg)
    return value


def _bool(data: Mapping[str, object], key: str, *, default: bool | None = None) -> bool:
    value = data[key] if default is None else data.get(key, default)
    if not isinstance(value, bool):
        msg = f"{key!r} must be a boolean, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def _int(data: Mapping[str, object], key: str, *, default: int) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{key!r} must be an integer, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def _str(data: Mapping[str, object], key: str) -> str:
    value = data[key]
    if not isinstance(value, str):
        msg = f"{key!r} must be a string, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def _opt_str(data: Mapping[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"{key!r} must be a string or null, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def _pair_methods(data: Mapping[str, object], key: str) -> list[PairMethod]:
    value = data.get(key, [])
    if not isinstance(value, list):
        msg = f"{key!r} must be a list, got {type(value).__name__}"
        raise TypeError(msg)
    methods: list[PairMethod] = []
    for item in value:
        if not isinstance(item, str):
            msg = f"{key!r} entries must be strings, got {type(item).__name__}"
            raise TypeError(msg)
        methods.append(PairMethod(item))
    return methods
