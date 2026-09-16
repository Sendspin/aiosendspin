"""Client-side handling of management commands.

Pure logic that maps a parsed server→client request onto the client's
``ClientPairingStore``, returning the
``ManagementResultPayload`` to reply with and an
``ManagementEffect`` telling the connection what to do after the reply. Kept free of
transport concerns so it is unit-testable against an ``InMemoryClientPairingStore``.
"""

from __future__ import annotations

from dataclasses import replace
from enum import Enum, auto
from typing import TYPE_CHECKING

from aiosendspin.models.core import UnpairedAccess
from aiosendspin.models.management import (
    ManagementResultData,
    ManagementResultPayload,
    PairingMethodConfig,
    RecordModeConfig,
    RecordSummary,
    StorageAccounting,
)
from aiosendspin.models.types import ManagementResult, PairMethod
from aiosendspin.noise.keys import PSK_SIZE, b64url_decode, psk_id_for
from aiosendspin.noise.trust_store import (
    ClientPairingRecord,
    PairingPsk,
    PskCategory,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Container

    from aiosendspin.models.management import (
        ManagementAddRecordPayload,
        ManagementRemoveRecordPayload,
        ManagementSetPairingConfigPayload,
        SetDynamicPairingCodeConfig,
        SetPairingPskConfig,
        SetStaticPairingCodeConfig,
        SetUnpairedAccessConfig,
    )
    from aiosendspin.noise.trust_store import ClientPairingStore

_STATIC_CODE_DIGITS = 8


class ManagementEffect(Enum):
    """Action the connection takes after sending the management/result."""

    NONE = auto()
    GOODBYE_UNAUTHORIZED = auto()
    """The requester demoted/removed its own record; close with goodbye 'unauthorized'."""


def _result(
    result: ManagementResult, data: ManagementResultData | None = None
) -> ManagementResultPayload:
    return ManagementResultPayload(result=result, data=data)


async def handle_unpair(store: ClientPairingStore, *, matched_psk_id: str) -> None:
    """Drop the matched record on server/unpair; a shared-PSK record is never removed here."""
    record = await store.record_by_psk_id(matched_psk_id)
    if record is None or record.server_id is None:
        return
    await store.remove_record(matched_psk_id)


async def handle_list_records(
    store: ClientPairingStore,
) -> tuple[ManagementResultPayload, ManagementEffect]:
    """Return summaries of all stored long-term records."""
    summaries = [
        RecordSummary(psk_id=r.psk_id, server_id=r.server_id, used=r.used)
        for r in await store.list_records()
    ]
    data = ManagementResultData(records=summaries)
    return _result(ManagementResult.OK, data), ManagementEffect.NONE


async def handle_add_record(
    store: ClientPairingStore, payload: ManagementAddRecordPayload
) -> tuple[ManagementResultPayload, ManagementEffect]:
    """Add a stored-pubkey or shared-PSK record from a provided PSK."""
    try:
        psk = b64url_decode(payload.psk)
    except ValueError:
        return _result(ManagementResult.INVALID), ManagementEffect.NONE
    if len(psk) != PSK_SIZE:
        return _result(ManagementResult.INVALID), ManagementEffect.NONE
    psk_id = psk_id_for(psk)
    if await store.resolve_by_psk_id(psk_id) is not None:
        return _result(ManagementResult.ALREADY_EXISTS), ManagementEffect.NONE
    if not await store.can_store_record():
        return _result(ManagementResult.STORAGE_EXHAUSTED), ManagementEffect.NONE
    await store.store_record(
        ClientPairingRecord(
            psk_id=psk_id,
            psk=psk,
            server_id=payload.server_id,
        )
    )
    return _result(ManagementResult.OK), ManagementEffect.NONE


async def handle_remove_record(
    store: ClientPairingStore,
    payload: ManagementRemoveRecordPayload,
    *,
    requester_psk_id: str | None,
) -> tuple[ManagementResultPayload, ManagementEffect]:
    """Remove a record; removing one's own record closes the session."""
    record = await store.record_by_psk_id(payload.psk_id)
    if record is None:
        return _result(ManagementResult.NOT_FOUND), ManagementEffect.NONE
    if not await store.can_remove_record(payload.psk_id):
        return _result(ManagementResult.INVALID), ManagementEffect.NONE
    effect = (
        ManagementEffect.GOODBYE_UNAUTHORIZED
        if record.psk_id == requester_psk_id
        else ManagementEffect.NONE
    )
    await store.remove_record(payload.psk_id)
    return _result(ManagementResult.OK), effect


async def handle_get_pairing_config(
    store: ClientPairingStore,
    *,
    implemented_pair_methods: Container[PairMethod],
) -> tuple[ManagementResultPayload, ManagementEffect]:
    """Assemble the pairing-config view; secrets are never included."""
    config = await store.get_pairing_config()
    data = ManagementResultData(
        pairing_psk=PairingMethodConfig(enabled=config.pairing_psk_enabled),
        static_pairing_code=(
            PairingMethodConfig(enabled=config.static_pairing_code_enabled)
            if PairMethod.STATIC_PAIRING_CODE in implemented_pair_methods
            else None
        ),
        dynamic_pairing_code=(
            PairingMethodConfig(
                enabled=config.dynamic_pairing_code_enabled,
                escalated=await store.is_pairing_round_limit_reached(),
            )
            if PairMethod.DYNAMIC_PAIRING_CODE in implemented_pair_methods
            else None
        ),
        record_mode=RecordModeConfig(psk_id=config.record_mode_psk_id),
        unpaired_access=UnpairedAccess(enabled=config.unpaired_access_enabled),
    )
    return _result(ManagementResult.OK, data), ManagementEffect.NONE


async def handle_set_pairing_config(
    store: ClientPairingStore,
    payload: ManagementSetPairingConfigPayload,
    *,
    implemented_pair_methods: Container[PairMethod],
) -> tuple[ManagementResultPayload, ManagementEffect]:
    """Apply a validated config patch (enabled flags, secrets, record mode, unpaired access)."""
    methods = (
        (PairMethod.PAIRING_PSK, payload.pairing_psk),
        (PairMethod.STATIC_PAIRING_CODE, payload.static_pairing_code),
        (PairMethod.DYNAMIC_PAIRING_CODE, payload.dynamic_pairing_code),
    )
    # 1. A patch on a method the client does not implement is invalid.
    if any(cfg is not None and method not in implemented_pair_methods for method, cfg in methods):
        return _result(ManagementResult.INVALID), ManagementEffect.NONE
    # 2. Validate secrets and record mode before mutating anything.
    psk_bytes: bytes | None = None
    if payload.pairing_psk is not None and payload.pairing_psk.psk is not None:
        psk_bytes = _decode_psk(payload.pairing_psk.psk)
        if psk_bytes is None:
            return _result(ManagementResult.INVALID), ManagementEffect.NONE
    if (
        payload.static_pairing_code is not None
        and payload.static_pairing_code.code is not None
        and not _valid_static_pairing_code(payload.static_pairing_code.code)
    ):
        return _result(ManagementResult.INVALID), ManagementEffect.NONE
    if (
        payload.static_pairing_code is not None
        and payload.static_pairing_code.enabled is True
        and payload.static_pairing_code.code is None
        and await store.static_pairing_code() is None
    ):
        # Enabling static_pairing_code with no static pairing code configured is invalid.
        return _result(ManagementResult.INVALID), ManagementEffect.NONE
    if payload.record_mode is not None and not await _is_shared_record(
        store, payload.record_mode.psk_id
    ):
        return _result(ManagementResult.INVALID), ManagementEffect.NONE
    # 3. Apply (all inputs validated; nothing below fails).
    if payload.record_mode is not None:
        await store.set_record_mode_psk_id(payload.record_mode.psk_id)
    config = await store.get_pairing_config()
    await store.store_pairing_config(
        replace(
            config,
            pairing_psk_enabled=_merge_enabled(
                payload.pairing_psk, current=config.pairing_psk_enabled
            ),
            static_pairing_code_enabled=_merge_enabled(
                payload.static_pairing_code, current=config.static_pairing_code_enabled
            ),
            dynamic_pairing_code_enabled=_merge_enabled(
                payload.dynamic_pairing_code, current=config.dynamic_pairing_code_enabled
            ),
            unpaired_access_enabled=_merge_enabled(
                payload.unpaired_access, current=config.unpaired_access_enabled
            ),
        )
    )
    if psk_bytes is not None:
        await store.set_pairing_psk(PairingPsk(psk_id=psk_id_for(psk_bytes), psk=psk_bytes))
    if payload.static_pairing_code is not None and payload.static_pairing_code.code is not None:
        await store.set_static_pairing_code(payload.static_pairing_code.code)
    return _result(ManagementResult.OK), ManagementEffect.NONE


async def handle_open_pairing_window(
    store: ClientPairingStore,
    *,
    implemented_pair_methods: Container[PairMethod],
    open_window: Callable[[], None],
) -> tuple[ManagementResultPayload, ManagementEffect]:
    """Open a pairing window in place of the operator gesture.

    Invalid when no pairing-code method is enabled; a no-op ``ok`` when a window is
    already open (``open_window`` is expected to absorb that case).
    """
    config = await store.get_pairing_config()
    static_enabled = (
        PairMethod.STATIC_PAIRING_CODE in implemented_pair_methods
        and config.static_pairing_code_enabled
    )
    dynamic_enabled = (
        PairMethod.DYNAMIC_PAIRING_CODE in implemented_pair_methods
        and config.dynamic_pairing_code_enabled
    )
    if not (static_enabled or dynamic_enabled):
        return _result(ManagementResult.INVALID), ManagementEffect.NONE
    open_window()
    return _result(ManagementResult.OK), ManagementEffect.NONE


async def with_storage(
    payload: ManagementResultPayload, store: ClientPairingStore, *, include_static: bool
) -> ManagementResultPayload:
    """Attach storage accounting to a result, or leave it absent.

    ``free`` is always set; ``capacity`` and the per-kind costs are added only when
    ``include_static`` (list-records and get-pairing-config). A store that reports no
    accounting (unbounded/unknown storage) leaves the payload unchanged.
    """
    report = await store.storage_accounting()
    if report is None:
        return payload
    storage = (
        StorageAccounting(
            free=report.free,
            capacity=report.capacity,
            cost_individual=report.cost_individual,
            cost_shared=report.cost_shared,
        )
        if include_static
        else StorageAccounting(free=report.free)
    )
    return replace(payload, storage=storage)


async def _is_shared_record(store: ClientPairingStore, psk_id: str) -> bool:
    """Return whether ``psk_id`` names a shared-PSK record (long-term, no counterparty)."""
    resolved = await store.resolve_by_psk_id(psk_id)
    return (
        resolved is not None
        and resolved.category is PskCategory.LONG_TERM
        and resolved.counterparty_id is None
    )


def _decode_psk(value: str) -> bytes | None:
    """Decode a base64url PSK, returning ``None`` if it is not a 32-byte key."""
    try:
        psk = b64url_decode(value)
    except ValueError:
        return None
    return psk if len(psk) == PSK_SIZE else None


def _valid_static_pairing_code(pairing_code: str) -> bool:
    """Return whether ``pairing_code`` is exactly 8 ASCII decimal digits."""
    return (
        len(pairing_code) == _STATIC_CODE_DIGITS
        and pairing_code.isascii()
        and pairing_code.isdigit()
    )


def _merge_enabled(
    cfg: SetPairingPskConfig
    | SetStaticPairingCodeConfig
    | SetDynamicPairingCodeConfig
    | SetUnpairedAccessConfig
    | None,
    *,
    current: bool,
) -> bool:
    """Return ``cfg.enabled`` when present, else the current value."""
    if cfg is None or cfg.enabled is None:
        return current
    return cfg.enabled
