"""Tests for :mod:`aiosendspin.noise.trust_store`."""

from __future__ import annotations

import json
import logging
import stat
import sys
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from aiosendspin.models.types import PairMethod
from aiosendspin.noise.keys import b64url_encode, generate_psk, psk_id_for
from aiosendspin.noise.trust_store import (
    PAIRING_ROUND_LIMIT,
    ClientPairingRecord,
    ClientPairingStore,
    FileClientPairingStore,
    FileServerPairingStore,
    InMemoryClientPairingStore,
    InMemoryServerPairingStore,
    PairingPsk,
    PskCategory,
    ServerPairingRecord,
    ServerPairingStore,
    StagedPairingPsk,
    TrustedUnpairedClient,
)
from tests.pairing_stores import ExhaustedClientStore, seed_used_client_records

if TYPE_CHECKING:
    from pathlib import Path


def _server_record(client_id: str = "client-A") -> ServerPairingRecord:
    psk = generate_psk()
    return ServerPairingRecord(
        psk_id=psk_id_for(psk), psk=psk, client_id=client_id, pair_methods=[]
    )


_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def _client_record(server_id: str = "server-X") -> ClientPairingRecord:
    psk = generate_psk()
    return ClientPairingRecord(psk_id=psk_id_for(psk), psk=psk, server_id=server_id)


# DEPRECATED(spec-pr-183): remove in aiosendspin <version>
def _shared_record() -> ClientPairingRecord:
    psk = generate_psk()
    return ClientPairingRecord(psk_id=psk_id_for(psk), psk=psk, server_id=None)


def _pairing_psk() -> PairingPsk:
    psk = generate_psk()
    return PairingPsk(psk_id=psk_id_for(psk), psk=psk)


def _staged_psk() -> StagedPairingPsk:
    psk = generate_psk()
    return StagedPairingPsk(psk_id=psk_id_for(psk), psk=psk)


def _record_dict(client_id: str, pair_methods: list[str]) -> dict[str, object]:
    psk = generate_psk()
    return {
        "psk_id": psk_id_for(psk),
        "psk": b64url_encode(psk),
        "client_id": client_id,
        "pair_methods": pair_methods,
        "created_at": "2026-05-01T12:00:00+00:00",
        "owner": None,
    }


@pytest.fixture(params=["memory", "file"])
async def client_store(request: pytest.FixtureRequest, tmp_path: Path) -> ClientPairingStore:
    """Each concrete client store, so conformance tests cover both implementations."""
    if request.param == "file":
        return await FileClientPairingStore.open(tmp_path / "client.json")
    return InMemoryClientPairingStore()


@pytest.fixture(params=["memory", "file"])
async def server_store(request: pytest.FixtureRequest, tmp_path: Path) -> ServerPairingStore:
    """Each concrete server store, so conformance tests cover both implementations."""
    if request.param == "file":
        return await FileServerPairingStore.open(tmp_path / "server.json")
    return InMemoryServerPairingStore()


def test_records_reject_wrong_psk_size() -> None:
    """Each record type enforces the 32-byte PSK invariant."""
    with pytest.raises(ValueError, match="PSK must be 32 bytes"):
        ServerPairingRecord(psk_id="x", psk=b"short", client_id="c", pair_methods=[])
    with pytest.raises(ValueError, match="PSK must be 32 bytes"):
        ClientPairingRecord(psk_id="x", psk=b"short", server_id="s")
    with pytest.raises(ValueError, match="PSK must be 32 bytes"):
        PairingPsk(psk_id="x", psk=b"short")


def test_server_record_round_trips_and_resolves() -> None:
    """ServerPairingRecord to/from dict round-trips; as_resolved names the client."""
    record = _server_record(client_id="client-A")
    assert ServerPairingRecord.from_dict(record.to_dict()) == record
    resolved = record.as_resolved()
    assert resolved.category is PskCategory.LONG_TERM
    assert resolved.counterparty_id == "client-A"
    assert "trust" not in record.to_dict()  # server never persists trust level


def test_server_record_with_method_appends_in_first_use_order() -> None:
    """with_method appends unseen methods in order and is a no-op for ones already present."""
    record = _server_record()
    assert record.pair_methods == []
    first = record.with_method(PairMethod.PAIRING_PSK)
    second = first.with_method(PairMethod.DYNAMIC_PAIRING_CODE)
    assert first.pair_methods == [PairMethod.PAIRING_PSK]
    assert second.pair_methods == [PairMethod.PAIRING_PSK, PairMethod.DYNAMIC_PAIRING_CODE]
    assert second.with_method(PairMethod.PAIRING_PSK) is second  # already present, unchanged


def test_server_record_pair_methods_round_trip_and_back_compat() -> None:
    """pair_methods round-trips; a legacy dict without the key loads as an empty list."""
    record = _server_record().with_method(PairMethod.DYNAMIC_PAIRING_CODE)
    assert ServerPairingRecord.from_dict(record.to_dict()) == record
    legacy = record.to_dict()
    del legacy["pair_methods"]
    assert ServerPairingRecord.from_dict(legacy).pair_methods == []


# DEPRECATED(spec-pr-179): remove in aiosendspin <version>
def test_server_record_maps_legacy_pair_method_names() -> None:
    """Pre-rename method names load as their pairing-code methods, de-duplicated in order."""
    data = _record_dict(
        "client-A", ["static_pin", "pairing_psk", "dynamic_pin", "static_pairing_code"]
    )
    record = ServerPairingRecord.from_dict(data)
    assert record.pair_methods == [
        PairMethod.STATIC_PAIRING_CODE,
        PairMethod.PAIRING_PSK,
        PairMethod.DYNAMIC_PAIRING_CODE,
    ]
    assert record.to_dict()["pair_methods"] == [
        "static_pairing_code",
        "pairing_psk",
        "dynamic_pairing_code",
    ]


def test_server_record_rejects_unknown_pair_method() -> None:
    """An unrecognised method name raises ValueError."""
    with pytest.raises(ValueError, match="unknown pair method 'carrier_pigeon'"):
        ServerPairingRecord.from_dict(_record_dict("client-A", ["carrier_pigeon"]))


def test_server_record_owner_round_trips_and_back_compat() -> None:
    """Owner round-trips; a legacy dict without the key loads as ``None``."""
    record = _server_record()
    assert record.owner is None
    owned = ServerPairingRecord(
        psk_id=record.psk_id,
        psk=record.psk,
        client_id=record.client_id,
        pair_methods=[],
        owner="user-1",
    )
    assert ServerPairingRecord.from_dict(owned.to_dict()) == owned
    legacy = owned.to_dict()
    del legacy["owner"]
    assert ServerPairingRecord.from_dict(legacy).owner is None


async def test_server_store_records_by_owner(server_store: ServerPairingStore) -> None:
    """records_by_owner returns only the records bound to the given owner."""
    unowned = _server_record(client_id="client-A")
    psk_b, psk_c = generate_psk(), generate_psk()
    owned_b = ServerPairingRecord(
        psk_id=psk_id_for(psk_b), psk=psk_b, client_id="client-B", pair_methods=[], owner="user-1"
    )
    owned_c = ServerPairingRecord(
        psk_id=psk_id_for(psk_c), psk=psk_c, client_id="client-C", pair_methods=[], owner="user-2"
    )
    for record in (unowned, owned_b, owned_c):
        await server_store.store_record(record)
    assert list(await server_store.records_by_owner("user-1")) == [owned_b]
    assert list(await server_store.records_by_owner("user-3")) == []


def test_client_record_round_trips_and_resolves() -> None:
    """ClientPairingRecord to/from dict round-trips; as_resolved names the server."""
    record = _client_record(server_id="server-X")
    restored = ClientPairingRecord.from_dict(record.to_dict())
    assert restored == record
    resolved = record.as_resolved()
    assert resolved.category is PskCategory.LONG_TERM
    assert resolved.counterparty_id == "server-X"


def test_pairing_psk_round_trips_and_resolves() -> None:
    """PairingPsk to/from dict round-trips; as_resolved has no counterparty."""
    pairing = _pairing_psk()
    assert PairingPsk.from_dict(pairing.to_dict()) == pairing
    resolved = pairing.as_resolved()
    assert resolved.category is PskCategory.PAIRING
    assert resolved.counterparty_id is None


def test_staged_pairing_psk_round_trips_and_resolves() -> None:
    """StagedPairingPsk to/from dict round-trips (created_at included) and resolves as pairing."""
    staged = _staged_psk()
    assert StagedPairingPsk.from_dict(staged.to_dict()) == staged
    resolved = staged.as_resolved()
    assert resolved.category is PskCategory.PAIRING
    assert resolved.counterparty_id is None


def test_psk_serialized_as_unpadded_base64url() -> None:
    """to_dict emits the PSK as an unpadded base64url string."""
    record = _server_record()
    psk_field = record.to_dict()["psk"]
    assert isinstance(psk_field, str)
    assert "=" not in psk_field


async def test_server_store_record_round_trip(server_store: ServerPairingStore) -> None:
    """store_record then record_by_client_id returns the record by client_id."""
    record = _server_record(client_id="client-A")
    await server_store.store_record(record)
    assert await server_store.record_by_client_id("client-A") == record
    assert await server_store.record_by_client_id("client-B") is None


async def test_server_store_remove_record(server_store: ServerPairingStore) -> None:
    """remove_record deletes the record by client_id; removing an absent one is a no-op."""
    await server_store.store_record(_server_record(client_id="client-A"))
    await server_store.remove_record("client-A")
    assert await server_store.record_by_client_id("client-A") is None
    await server_store.remove_record("client-A")  # no-op


def test_trusted_unpaired_client_round_trip() -> None:
    """TrustedUnpairedClient.to_dict/from_dict preserves every field."""
    client = TrustedUnpairedClient(client_id="client-A")
    restored = TrustedUnpairedClient.from_dict(client.to_dict())
    assert restored == client


async def test_server_store_trusted_unpaired_lifecycle(server_store: ServerPairingStore) -> None:
    """Trusted-unpaired approvals: add, look up, list, remove (no-op when absent)."""
    assert await server_store.trusted_unpaired("client-A") is None
    assert list(await server_store.list_trusted_unpaired()) == []

    client = TrustedUnpairedClient(client_id="client-A")
    await server_store.add_trusted_unpaired(client)
    assert await server_store.trusted_unpaired("client-A") == client
    assert list(await server_store.list_trusted_unpaired()) == [client]

    await server_store.remove_trusted_unpaired("client-A")
    assert await server_store.trusted_unpaired("client-A") is None
    await server_store.remove_trusted_unpaired("client-A")  # no-op


async def test_server_store_list_staged_pairing_psks(server_store: ServerPairingStore) -> None:
    """list_staged_pairing_psks returns every staged PSK; unstaging removes it."""
    assert list(await server_store.list_staged_pairing_psks()) == []
    pp = _staged_psk()
    await server_store.stage_pairing_psk("client-A", pp)
    assert list(await server_store.list_staged_pairing_psks()) == [pp]
    await server_store.unstage_pairing_psk("client-A")
    assert list(await server_store.list_staged_pairing_psks()) == []


async def test_file_server_store_absent_file_is_empty(tmp_path: Path) -> None:
    """A FileServerPairingStore over a missing path opens empty."""
    store = await FileServerPairingStore.open(tmp_path / "pairings.json")
    assert await store.record_by_client_id("client-A") is None
    assert list(await store.list_records()) == []
    assert await store.trusted_unpaired("client-A") is None


async def test_file_server_store_persists_all_categories(tmp_path: Path) -> None:
    """Records, staged Pairing PSKs, and trusted-unpaired clients survive a reload."""
    path = tmp_path / "sub" / "pairings.json"  # parent dir created on write
    store = await FileServerPairingStore.open(path)
    record = _server_record(client_id="client-A")
    staged = _staged_psk()
    trusted = TrustedUnpairedClient(client_id="client-T")
    await store.store_record(record)
    await store.stage_pairing_psk("client-S", staged)
    await store.add_trusted_unpaired(trusted)

    reloaded = await FileServerPairingStore.open(path)
    assert await reloaded.record_by_client_id("client-A") == record
    assert list(await reloaded.list_records()) == [record]
    assert await reloaded.staged_pairing_psk("client-S") == staged
    assert list(await reloaded.list_staged_pairing_psks()) == [staged]
    assert await reloaded.trusted_unpaired("client-T") == trusted


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
async def test_file_stores_are_owner_readable_only(tmp_path: Path) -> None:
    """Store files hold PSKs; writes must produce 0600 files."""
    path = tmp_path / "sub" / "pairings.json"
    server_store = await FileServerPairingStore.open(path)
    await server_store.store_record(_server_record())
    client_path = tmp_path / "sub" / "client.json"
    await FileClientPairingStore.open(client_path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(client_path.stat().st_mode) == 0o600


async def test_file_server_store_removal_persists(tmp_path: Path) -> None:
    """Removing a trusted-unpaired approval is reflected after a reload."""
    path = tmp_path / "pairings.json"
    store = await FileServerPairingStore.open(path)
    await store.add_trusted_unpaired(TrustedUnpairedClient(client_id="client-T"))
    await store.remove_trusted_unpaired("client-T")
    reloaded = await FileServerPairingStore.open(path)
    assert await reloaded.trusted_unpaired("client-T") is None


async def test_file_server_store_tolerates_absent_sections(tmp_path: Path) -> None:
    """An older-format file missing newer sections loads what's present; the rest are empty."""
    path = tmp_path / "pairings.json"
    record = _server_record(client_id="client-A")
    path.write_text(json.dumps({"records": {"client-A": record.to_dict()}}), encoding="utf-8")
    store = await FileServerPairingStore.open(path)
    assert await store.record_by_client_id("client-A") == record
    assert list(await store.list_staged_pairing_psks()) == []
    assert list(await store.list_trusted_unpaired()) == []


# DEPRECATED(spec-pr-179): remove in aiosendspin <version>
async def test_file_server_store_loads_legacy_pair_method_names(tmp_path: Path) -> None:
    """A store saved with pre-rename method names loads, and saves only the new names."""
    path = tmp_path / "pairings.json"
    staged = _staged_psk()
    path.write_text(
        json.dumps(
            {
                "records": {
                    "client-A": _record_dict("client-A", ["dynamic_pin"]),
                    "client-B": _record_dict("client-B", ["pairing_psk", "static_pin"]),
                },
                "staged_pairing_psks": {"client-S": staged.to_dict()},
                "trusted_unpaired_clients": {
                    "client-T": {"client_id": "client-T", "created_at": "2026-05-01T12:00:00+00:00"}
                },
            }
        ),
        encoding="utf-8",
    )

    store = await FileServerPairingStore.open(path)
    record_a = await store.record_by_client_id("client-A")
    record_b = await store.record_by_client_id("client-B")
    assert record_a is not None
    assert record_a.pair_methods == [PairMethod.DYNAMIC_PAIRING_CODE]
    assert record_b is not None
    assert record_b.pair_methods == [PairMethod.PAIRING_PSK, PairMethod.STATIC_PAIRING_CODE]
    assert await store.staged_pairing_psk("client-S") == staged
    assert await store.trusted_unpaired("client-T") is not None

    await store.store_record(_server_record(client_id="client-C"))
    saved = json.loads(path.read_text(encoding="utf-8"))["records"]
    assert saved["client-A"]["pair_methods"] == ["dynamic_pairing_code"]
    assert saved["client-B"]["pair_methods"] == ["pairing_psk", "static_pairing_code"]
    assert saved["client-C"]["pair_methods"] == []


async def test_file_server_store_skips_record_with_unknown_pair_method(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A record naming an unknown method is skipped with a warning; the rest of the store loads."""
    path = tmp_path / "pairings.json"
    staged = _staged_psk()
    known = _record_dict("client-A", ["pairing_psk"])
    path.write_text(
        json.dumps(
            {
                "records": {
                    "client-A": known,
                    "client-X": _record_dict("client-X", ["pairing_psk", "carrier_pigeon"]),
                },
                "staged_pairing_psks": {"client-S": staged.to_dict()},
                "trusted_unpaired_clients": {
                    "client-T": {"client_id": "client-T", "created_at": "2026-05-01T12:00:00+00:00"}
                },
            }
        ),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="aiosendspin.noise.trust_store"):
        store = await FileServerPairingStore.open(path)

    assert list(await store.list_records()) == [ServerPairingRecord.from_dict(known)]
    assert await store.record_by_client_id("client-X") is None
    assert await store.staged_pairing_psk("client-S") == staged
    assert await store.trusted_unpaired("client-T") is not None
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert "client-X" in caplog.records[0].getMessage()
    assert "carrier_pigeon" in caplog.records[0].getMessage()


async def test_file_server_store_rejects_malformed_file(tmp_path: Path) -> None:
    """A present-but-malformed store fails loud rather than silently dropping credentials."""
    non_object = tmp_path / "top.json"
    non_object.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(TypeError, match="must contain a JSON object"):
        await FileServerPairingStore.open(non_object)

    bad_section = tmp_path / "section.json"
    bad_section.write_text(json.dumps({"records": "nope"}), encoding="utf-8")
    with pytest.raises(TypeError, match="must be an object"):
        await FileServerPairingStore.open(bad_section)

    bad_entry = tmp_path / "entry.json"
    bad_entry.write_text(json.dumps({"records": {"client-A": 5}}), encoding="utf-8")
    with pytest.raises(TypeError, match="must be an object"):
        await FileServerPairingStore.open(bad_entry)


# DEPRECATED(spec-pr-183): remove in aiosendspin <version>
async def test_file_client_store_seeds_shared_record_on_first_open(tmp_path: Path) -> None:
    """First open provisions a shared-PSK fallback record referenced by record_mode."""
    path = tmp_path / "client.json"
    store = await FileClientPairingStore.open(path)
    config = await store.get_pairing_config()
    records = list(await store.list_records())
    assert len(records) == 1
    shared = records[0]
    assert shared.server_id is None  # shared-PSK record
    assert config.record_mode_psk_id == shared.psk_id
    # The seed is persisted, so it is stable across reopen.
    reopened = await FileClientPairingStore.open(path)
    assert (await reopened.get_pairing_config()).record_mode_psk_id == shared.psk_id


async def test_file_client_store_persists_state(tmp_path: Path) -> None:
    """Records, config, Pairing PSK, static code, and failure count survive a reload."""
    path = tmp_path / "client.json"
    store = await FileClientPairingStore.open(path)
    record = _client_record(server_id="server-X")
    pairing = _pairing_psk()
    await store.store_record(record)
    await store.set_pairing_psk(pairing)
    await store.set_static_pairing_code("12345678")
    await store.record_pairing_round()

    reloaded = await FileClientPairingStore.open(path)
    assert await reloaded.record_by_server_id("server-X") == record
    assert await reloaded.pairing_psk() == pairing
    assert await reloaded.static_pairing_code() == "12345678"
    assert await reloaded.pairing_round_count() == 1


# DEPRECATED(spec-pr-179): remove in aiosendspin <version>
async def test_file_client_store_migrates_per_method_pin_failures(tmp_path: Path) -> None:
    """A store with per-method counters carries its dynamic count over as the round count."""
    path = tmp_path / "client.json"
    await FileClientPairingStore.open(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["pin_failures"] = {"dynamic_pin": PAIRING_ROUND_LIMIT, "static_pin": 3}
    path.write_text(json.dumps(data), encoding="utf-8")

    reloaded = await FileClientPairingStore.open(path)
    assert await reloaded.pairing_round_count() == PAIRING_ROUND_LIMIT


async def test_file_client_store_persists_last_playback_server(tmp_path: Path) -> None:
    """The last-playback server id survives a reload."""
    path = tmp_path / "client.json"
    store = await FileClientPairingStore.open(path)
    await store.set_last_playback_server_id("server-X")

    reloaded = await FileClientPairingStore.open(path)
    assert await reloaded.get_last_playback_server_id() == "server-X"


async def test_last_playback_server_write_retries_after_save_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed save leaves the prior value intact so the same write can retry."""
    store = InMemoryClientPairingStore()
    save_attempts = 0

    async def fail_once() -> None:
        nonlocal save_attempts
        save_attempts += 1
        if save_attempts == 1:
            raise OSError("store unavailable")

    monkeypatch.setattr(store, "_save", fail_once)

    with pytest.raises(OSError, match="store unavailable"):
        await store.set_last_playback_server_id("server-X")
    assert await store.get_last_playback_server_id() is None

    await store.set_last_playback_server_id("server-X")

    assert save_attempts == 2
    assert await store.get_last_playback_server_id() == "server-X"


async def test_file_client_store_pairing_outcome_generates_per_server_record(
    tmp_path: Path,
) -> None:
    """An unbounded file client mints a fresh per-server record on pairing."""
    store = await FileClientPairingStore.open(tmp_path / "client.json")
    psk, record = await store.resolve_pairing_outcome(server_id="server-Y")
    assert record is not None
    assert record.server_id == "server-Y"
    assert record.psk == psk


async def test_client_pairing_psk_lifecycle(client_store: ClientPairingStore) -> None:
    """The client's accepted Pairing PSK: set, look up, replace, clear."""
    pairing = _pairing_psk()
    await client_store.set_pairing_psk(pairing)
    assert await client_store.pairing_psk() == pairing
    assert await client_store.resolve_by_psk_id(pairing.psk_id) == pairing.as_resolved()
    # Setting a new one replaces the old.
    other = _pairing_psk()
    await client_store.set_pairing_psk(other)
    assert await client_store.pairing_psk() == other
    assert await client_store.resolve_by_psk_id(pairing.psk_id) is None
    await client_store.clear_pairing_psk()
    assert await client_store.pairing_psk() is None
    assert await client_store.resolve_by_psk_id(other.psk_id) is None
    # Clearing when absent is a no-op.
    await client_store.clear_pairing_psk()


async def test_client_static_pairing_code_lifecycle(client_store: ClientPairingStore) -> None:
    """The client's configured static pairing code: set, look up, replace, clear."""
    assert await client_store.static_pairing_code() is None
    await client_store.set_static_pairing_code("12345678")
    assert await client_store.static_pairing_code() == "12345678"
    await client_store.set_static_pairing_code("87654321")
    assert await client_store.static_pairing_code() == "87654321"
    await client_store.clear_static_pairing_code()
    assert await client_store.static_pairing_code() is None
    # Clearing when absent is a no-op.
    await client_store.clear_static_pairing_code()


@pytest.mark.parametrize("bad_code", ["1234", "123456789", "abcdefgh", "1234567 "])
async def test_set_static_pairing_code_rejects_non_8_digit(bad_code: str) -> None:
    """The static pairing code must be exactly 8 decimal digits (spec definition)."""
    store = InMemoryClientPairingStore()
    with pytest.raises(ValueError, match="8 decimal digits"):
        await store.set_static_pairing_code(bad_code)


async def test_client_store_resolves_by_psk_id_and_finds_by_server_id(
    client_store: ClientPairingStore,
) -> None:
    """ClientPairingStore resolves a record by psk_id and finds it by server_id."""
    record = _client_record(server_id="server-X")
    await client_store.store_record(record)
    assert await client_store.resolve_by_psk_id(record.psk_id) == record.as_resolved()
    assert await client_store.record_by_server_id("server-X") == record
    assert await client_store.resolve_by_psk_id("nope") is None
    assert await client_store.record_by_server_id("server-Y") is None


async def test_client_store_record_takes_precedence_over_pairing_psk(
    client_store: ClientPairingStore,
) -> None:
    """When a long-term record and a Pairing PSK share a psk_id, the record wins."""
    record = _client_record(server_id="server-X")
    pairing = PairingPsk(psk_id=record.psk_id, psk=record.psk)
    await client_store.set_pairing_psk(pairing)
    await client_store.store_record(record)
    assert await client_store.resolve_by_psk_id(record.psk_id) == record.as_resolved()


async def test_client_store_mark_record_used(client_store: ClientPairingStore) -> None:
    """mark_record_used flags the record and refreshes its last use; absent is a no-op."""
    record = replace(_client_record(server_id="server-X"), last_used_at=_EPOCH)
    await client_store.store_record(record)
    stored = await client_store.record_by_psk_id(record.psk_id)
    assert stored is not None
    assert stored.used is False

    await client_store.mark_record_used(record.psk_id)
    used = await client_store.record_by_psk_id(record.psk_id)
    assert used is not None
    assert used.used is True
    assert used.last_used_at > _EPOCH

    await client_store.mark_record_used(record.psk_id)
    reused = await client_store.record_by_psk_id(record.psk_id)
    assert reused is not None
    assert reused.last_used_at >= used.last_used_at
    await client_store.mark_record_used("absent")  # no-op


async def test_client_store_remove_and_list(client_store: ClientPairingStore) -> None:
    """Removing deletes a record; records() reflects current contents."""
    a = _client_record(server_id="server-A")
    b = _client_record(server_id="server-B")
    await client_store.store_record(a)
    await client_store.store_record(b)
    added = {r for r in await client_store.list_records() if r.server_id is not None}
    assert added == {a, b}
    await client_store.remove_record(a.psk_id)
    assert await client_store.resolve_by_psk_id(a.psk_id) is None
    added = {r for r in await client_store.list_records() if r.server_id is not None}
    assert added == {b}
    # Removing an absent record is a no-op.
    await client_store.remove_record("absent")


async def test_client_store_replace_record_drops_prior_for_server(
    client_store: ClientPairingStore,
) -> None:
    """Re-pairing a server leaves a single record, keyed by the newest psk_id."""
    old = _client_record(server_id="server-X")
    await client_store.store_record(old)
    new = _client_record(server_id="server-X")
    await client_store.replace_record_for_server_id(new)
    for_server = [r for r in await client_store.list_records() if r.server_id == "server-X"]
    assert for_server == [new]
    assert await client_store.record_by_psk_id(old.psk_id) is None


# DEPRECATED(spec-pr-183): remove in aiosendspin <version>
async def test_client_store_replace_record_keeps_shared_records(
    client_store: ClientPairingStore,
) -> None:
    """A shared record binds to no server, so replacing one leaves the others alone."""
    existing = _shared_record()
    await client_store.store_record(existing)

    await client_store.replace_record_for_server_id(_shared_record())

    assert await client_store.record_by_psk_id(existing.psk_id) == existing


# DEPRECATED(spec-pr-183): remove in aiosendspin <version>
async def test_client_store_reports_no_storage_accounting_by_default(
    client_store: ClientPairingStore,
) -> None:
    """The default store is unbounded and reports no storage accounting."""
    assert await client_store.storage_accounting() is None


async def test_pairing_round_counter_increments_and_resets(
    client_store: ClientPairingStore,
) -> None:
    """Rounds accumulate and reset clears the count."""
    assert await client_store.pairing_round_count() == 0
    assert await client_store.record_pairing_round() == 1
    assert await client_store.record_pairing_round() == 2
    await client_store.reset_pairing_rounds()
    assert await client_store.pairing_round_count() == 0


async def test_pairing_round_limit_is_reached_at_20_and_clears_on_reset(
    client_store: ClientPairingStore,
) -> None:
    """The round limit is reached at 20 rounds and clears only on reset."""
    assert PAIRING_ROUND_LIMIT == 20
    for _ in range(PAIRING_ROUND_LIMIT - 1):
        await client_store.record_pairing_round()
    assert not await client_store.is_pairing_round_limit_reached()
    await client_store.record_pairing_round()
    assert await client_store.is_pairing_round_limit_reached()
    await client_store.reset_pairing_rounds()
    assert not await client_store.is_pairing_round_limit_reached()


# DEPRECATED(spec-pr-183): remove in aiosendspin <version>
# --- shared-PSK records --------------------------------------------------


# DEPRECATED(spec-pr-183): remove in aiosendspin <version>
def test_shared_record_round_trips_and_resolves() -> None:
    """A shared-PSK record omits server_id; as_resolved carries no counterparty."""
    record = _shared_record()
    assert record.server_id is None
    restored = ClientPairingRecord.from_dict(record.to_dict())
    assert restored == record
    assert restored.server_id is None
    resolved = record.as_resolved()
    assert resolved.category is PskCategory.LONG_TERM
    assert resolved.counterparty_id is None


# DEPRECATED(spec-pr-183): remove in aiosendspin <version>
async def test_shared_record_excluded_from_server_id_lookup(
    client_store: ClientPairingStore,
) -> None:
    """A shared-PSK record is found by psk_id but never by server_id."""
    record = _shared_record()
    await client_store.store_record(record)
    assert await client_store.resolve_by_psk_id(record.psk_id) == record.as_resolved()
    assert await client_store.record_by_server_id("any-server") is None


# DEPRECATED(spec-pr-183): remove in aiosendspin <version>
async def test_set_record_mode_psk_id_validates_reference(
    client_store: ClientPairingStore,
) -> None:
    """The record_mode psk_id must reference an existing shared-PSK record."""
    shared = _shared_record()
    pubkey = _client_record(server_id="server-X")
    await client_store.store_record(shared)
    await client_store.store_record(pubkey)

    # The store is pre-provisioned with a shared fallback at construction.
    initial = (await client_store.get_pairing_config()).record_mode_psk_id
    assert initial is not None
    assert await client_store.record_by_psk_id(initial) is not None

    with pytest.raises(ValueError, match="references no record"):
        await client_store.set_record_mode_psk_id("missing")
    with pytest.raises(ValueError, match="must reference a shared-PSK record"):
        await client_store.set_record_mode_psk_id(pubkey.psk_id)

    await client_store.set_record_mode_psk_id(shared.psk_id)
    assert (await client_store.get_pairing_config()).record_mode_psk_id == shared.psk_id
    assert not await client_store.can_remove_record(shared.psk_id)


async def test_resolve_outcome_mints_stored_pubkey_record(
    client_store: ClientPairingStore,
) -> None:
    """A storable store generates a fresh PSK bound to the server_id."""
    psk, record = await client_store.resolve_pairing_outcome(server_id="server-X")
    assert record is not None
    assert record.psk == psk
    assert record.psk_id == psk_id_for(psk)
    assert record.server_id == "server-X"


# DEPRECATED(spec-pr-183): remove in aiosendspin <version>
async def test_resolve_outcome_falls_back_to_shared_on_exhaustion() -> None:
    """A configured shared fallback admits under the shared record when full."""
    store = ExhaustedClientStore()
    shared = _shared_record()
    await store.store_record(shared)
    await store.set_record_mode_psk_id(shared.psk_id)
    psk, record = await store.resolve_pairing_outcome(server_id="server-X")
    assert psk == shared.psk
    assert record is None


# --- record capacity and eviction -----------------------------------------


async def _client_store_with_capacity(
    kind: str, tmp_path: Path, record_capacity: int
) -> ClientPairingStore:
    if kind == "file":
        return await FileClientPairingStore.open(
            tmp_path / "client.json", record_capacity=record_capacity
        )
    return InMemoryClientPairingStore(record_capacity=record_capacity)


async def _per_server_psk_ids(store: ClientPairingStore) -> set[str]:
    return {r.psk_id for r in await store.list_records() if r.server_id is not None}


async def test_client_store_record_capacity_defaults_to_16(
    client_store: ClientPairingStore,
) -> None:
    """A client store holds 16 per-server records unless configured otherwise."""
    assert client_store.record_capacity == 16


@pytest.mark.parametrize("kind", ["memory", "file"])
async def test_client_store_record_capacity_is_configurable(kind: str, tmp_path: Path) -> None:
    """The record capacity is set at construction, down to the spec minimum of 5."""
    store = await _client_store_with_capacity(kind, tmp_path, 5)
    assert store.record_capacity == 5


@pytest.mark.parametrize("kind", ["memory", "file"])
async def test_client_store_rejects_capacity_below_5(kind: str, tmp_path: Path) -> None:
    """A record capacity below the spec minimum of 5 is rejected."""
    with pytest.raises(ValueError, match="at least 5"):
        await _client_store_with_capacity(kind, tmp_path, 4)


@pytest.mark.parametrize("kind", ["memory", "file"])
async def test_pairing_at_capacity_evicts_least_recently_used(kind: str, tmp_path: Path) -> None:
    """At capacity, persisting a new server's record evicts the least recently used one."""
    store = await _client_store_with_capacity(kind, tmp_path, 5)
    oldest, *rest = await seed_used_client_records(store, 5)
    new = _client_record(server_id="server-new")

    await store.replace_record_for_server_id(new)

    assert await _per_server_psk_ids(store) == {new.psk_id, *(r.psk_id for r in rest)}
    assert await store.record_by_psk_id(oldest.psk_id) is None


@pytest.mark.parametrize("kind", ["memory", "file"])
async def test_eviction_skips_records_backing_open_connections(kind: str, tmp_path: Path) -> None:
    """A protected record is never evicted, even when it is the least recently used."""
    store = await _client_store_with_capacity(kind, tmp_path, 5)
    oldest, second, *_ = await seed_used_client_records(store, 5)
    new = _client_record(server_id="server-new")

    await store.replace_record_for_server_id(new, protected={oldest.psk_id})

    assert await store.record_by_psk_id(oldest.psk_id) is not None
    assert await store.record_by_psk_id(second.psk_id) is None
    assert await store.record_by_psk_id(new.psk_id) == new


async def test_eviction_never_touches_shared_records(tmp_path: Path) -> None:
    """Shared records neither count toward the capacity nor get evicted."""
    store = await _client_store_with_capacity("memory", tmp_path, 5)
    shared = [r for r in await store.list_records() if r.server_id is None]
    assert shared
    oldest, *rest = await seed_used_client_records(store, 5)
    new = _client_record(server_id="server-new")

    await store.replace_record_for_server_id(new)

    remaining = await store.list_records()
    assert all(r in remaining for r in shared)
    assert await _per_server_psk_ids(store) == {new.psk_id, *(r.psk_id for r in rest)}
    assert await store.record_by_psk_id(oldest.psk_id) is None


async def test_repairing_a_known_server_at_capacity_evicts_nothing(tmp_path: Path) -> None:
    """Re-pairing a stored server replaces its record without evicting another."""
    store = await _client_store_with_capacity("memory", tmp_path, 5)
    seeded = await seed_used_client_records(store, 5)
    renewed = _client_record(server_id="server-2")

    await store.replace_record_for_server_id(renewed)

    expected = {r.psk_id for r in seeded if r is not seeded[2]} | {renewed.psk_id}
    assert await _per_server_psk_ids(store) == expected


@pytest.mark.parametrize("kind", ["memory", "file"])
async def test_repairing_keeps_the_prior_record_while_a_connection_uses_it(
    kind: str, tmp_path: Path
) -> None:
    """Re-pairing a server keeps its prior record while an open connection still uses it."""
    store = await _client_store_with_capacity(kind, tmp_path, 5)
    seeded = await seed_used_client_records(store, 5)
    renewed = _client_record(server_id="server-2")

    await store.replace_record_for_server_id(renewed, protected={seeded[2].psk_id})

    assert await store.record_by_psk_id(seeded[2].psk_id) is not None
    assert await store.record_by_psk_id(renewed.psk_id) == renewed
    assert await store.record_by_psk_id(seeded[0].psk_id) is None
    assert len(await _per_server_psk_ids(store)) == 5


async def test_pairing_persists_when_every_other_record_is_protected(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """With nothing evictable the new record still persists, over capacity, with a warning."""
    store = await _client_store_with_capacity("memory", tmp_path, 5)
    seeded = await seed_used_client_records(store, 5)
    new = _client_record(server_id="server-new")

    with caplog.at_level(logging.WARNING):
        await store.replace_record_for_server_id(new, protected={r.psk_id for r in seeded})

    assert await _per_server_psk_ids(store) == {new.psk_id, *(r.psk_id for r in seeded)}
    assert any("exceed the capacity" in r.message for r in caplog.records)


def test_client_record_without_last_used_at_falls_back_to_created_at() -> None:
    """A record written before last_used_at existed loads with its creation time."""
    record = _client_record()
    data = record.to_dict()
    del data["last_used_at"]

    restored = ClientPairingRecord.from_dict(data)

    assert restored.last_used_at == record.created_at


async def test_file_client_store_loads_records_without_last_used_at(tmp_path: Path) -> None:
    """A pairing store file written before last_used_at existed still loads."""
    path = tmp_path / "client.json"
    store = await FileClientPairingStore.open(path)
    record = _client_record()
    await store.store_record(record)
    data = json.loads(path.read_text(encoding="utf-8"))
    for entry in data["records"].values():
        del entry["last_used_at"]
    path.write_text(json.dumps(data), encoding="utf-8")

    reopened = await FileClientPairingStore.open(path)

    restored = await reopened.record_by_psk_id(record.psk_id)
    assert restored is not None
    assert restored.last_used_at == record.created_at
