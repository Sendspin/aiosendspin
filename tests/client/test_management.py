"""Unit tests for the client-side management command handlers."""

from __future__ import annotations

from dataclasses import replace

from aiosendspin.client.management import (
    ManagementEffect,
    handle_add_record,
    handle_get_pairing_config,
    handle_list_records,
    handle_open_pairing_window,
    handle_remove_record,
    handle_set_pairing_config,
    handle_unpair,
    with_storage,
)
from aiosendspin.models.management import (
    ManagementAddRecordPayload,
    ManagementRemoveRecordPayload,
    ManagementResultPayload,
    ManagementSetPairingConfigPayload,
    RecordModeConfig,
    SetPairingPskConfig,
    SetStaticPairingCodeConfig,
    SetUnpairedAccessConfig,
    StorageAccounting,
)
from aiosendspin.models.types import ManagementResult, PairMethod
from aiosendspin.noise.keys import b64url_encode, generate_psk, psk_id_for
from aiosendspin.noise.trust_store import (
    PAIRING_ROUND_LIMIT,
    ClientPairingRecord,
    InMemoryClientPairingStore,
)
from tests.pairing_stores import BoundedClientStore, ExhaustedClientStore

_ALL_METHODS = frozenset(
    {PairMethod.PAIRING_PSK, PairMethod.STATIC_PAIRING_CODE, PairMethod.DYNAMIC_PAIRING_CODE}
)
_WITHOUT_STATIC_PAIRING_CODE = frozenset({PairMethod.PAIRING_PSK, PairMethod.DYNAMIC_PAIRING_CODE})

SERVER_ID = "srv-self"


async def _seed_record(
    store: InMemoryClientPairingStore,
    *,
    server_id: str | None,
) -> ClientPairingRecord:
    psk = generate_psk()
    record = ClientPairingRecord(psk_id=psk_id_for(psk), psk=psk, server_id=server_id)
    await store.store_record(record)
    return record


# --- list-records -------------------------------------------------------------


async def test_list_records_summarizes_records() -> None:
    """list-records projects each record, omitting server_id for shared-PSK records."""
    store = InMemoryClientPairingStore()
    stored = await _seed_record(store, server_id=SERVER_ID)
    shared = await _seed_record(store, server_id=None)
    await store.mark_record_used(stored.psk_id)
    payload, effect = await handle_list_records(store)
    assert effect is ManagementEffect.NONE
    assert payload.result is ManagementResult.OK
    assert payload.data is not None
    by_id = {s.psk_id: s for s in payload.data.records or []}
    assert by_id[stored.psk_id].server_id == SERVER_ID
    assert by_id[stored.psk_id].used is True
    assert by_id[shared.psk_id].server_id is None
    assert by_id[shared.psk_id].used is False


# --- add-record ---------------------------------------------------------------


async def test_add_record_ok() -> None:
    """A valid add-record persists a new stored-pubkey record."""
    store = InMemoryClientPairingStore()
    psk = generate_psk()
    payload, _ = await handle_add_record(
        store,
        ManagementAddRecordPayload(psk=b64url_encode(psk), server_id="srv2"),
    )
    assert payload.result is ManagementResult.OK
    record = await store.record_by_psk_id(psk_id_for(psk))
    assert record is not None
    assert record.server_id == "srv2"


async def test_add_record_already_exists() -> None:
    """Adding a record whose psk_id is already stored is rejected."""
    store = InMemoryClientPairingStore()
    existing = await _seed_record(store, server_id="srv2")
    payload, _ = await handle_add_record(
        store,
        ManagementAddRecordPayload(psk=b64url_encode(existing.psk), server_id="srv2"),
    )
    assert payload.result is ManagementResult.ALREADY_EXISTS


async def test_add_record_invalid_psk() -> None:
    """A PSK that does not decode to 32 bytes is rejected as invalid."""
    store = InMemoryClientPairingStore()
    payload, _ = await handle_add_record(
        store,
        ManagementAddRecordPayload(psk="not-valid-base64-len"),
    )
    assert payload.result is ManagementResult.INVALID


async def test_add_record_storage_exhausted() -> None:
    """Add-record reports storage_exhausted when the store cannot persist."""
    store = ExhaustedClientStore()
    payload, _ = await handle_add_record(
        store,
        ManagementAddRecordPayload(psk=b64url_encode(generate_psk())),
    )
    assert payload.result is ManagementResult.STORAGE_EXHAUSTED


# --- remove-record ------------------------------------------------------------


async def test_remove_record_ok() -> None:
    """Removing an existing record succeeds and deletes it from the store."""
    store = InMemoryClientPairingStore()
    record = await _seed_record(store, server_id="srv2")
    payload, effect = await handle_remove_record(
        store,
        ManagementRemoveRecordPayload(psk_id=record.psk_id),
        requester_psk_id="other-psk",
    )
    assert payload.result is ManagementResult.OK
    assert effect is ManagementEffect.NONE
    assert await store.record_by_psk_id(record.psk_id) is None


async def test_remove_record_not_found() -> None:
    """Removing an unknown psk_id reports not_found."""
    store = InMemoryClientPairingStore()
    payload, _ = await handle_remove_record(
        store,
        ManagementRemoveRecordPayload(psk_id="missing"),
        requester_psk_id="other-psk",
    )
    assert payload.result is ManagementResult.NOT_FOUND


async def test_remove_record_referenced_by_record_mode_is_invalid() -> None:
    """A record referenced by a record_mode.psk_id cannot be removed."""
    store = InMemoryClientPairingStore()
    shared = await _seed_record(store, server_id=None)
    await store.set_record_mode_psk_id(shared.psk_id)
    payload, _ = await handle_remove_record(
        store,
        ManagementRemoveRecordPayload(psk_id=shared.psk_id),
        requester_psk_id="other-psk",
    )
    assert payload.result is ManagementResult.INVALID
    assert await store.record_by_psk_id(shared.psk_id) is not None


async def test_remove_record_self_closes_session() -> None:
    """Removing the requester's own record signals a goodbye-unauthorized effect."""
    store = InMemoryClientPairingStore()
    record = await _seed_record(store, server_id=SERVER_ID)
    payload, effect = await handle_remove_record(
        store,
        ManagementRemoveRecordPayload(psk_id=record.psk_id),
        requester_psk_id=record.psk_id,
    )
    assert payload.result is ManagementResult.OK
    assert effect is ManagementEffect.GOODBYE_UNAUTHORIZED


async def test_remove_shared_record_backing_session_closes_session() -> None:
    """Removing the shared-PSK record that admitted the session also closes it."""
    store = InMemoryClientPairingStore()
    shared = await _seed_record(store, server_id=None)
    payload, effect = await handle_remove_record(
        store,
        ManagementRemoveRecordPayload(psk_id=shared.psk_id),
        requester_psk_id=shared.psk_id,
    )
    assert payload.result is ManagementResult.OK
    assert effect is ManagementEffect.GOODBYE_UNAUTHORIZED


async def test_remove_other_record_same_server_id_keeps_session() -> None:
    """Removing a different record naming the requester's server does not close the session."""
    store = InMemoryClientPairingStore()
    mine = await _seed_record(store, server_id=SERVER_ID)
    duplicate = await _seed_record(store, server_id=SERVER_ID)
    payload, effect = await handle_remove_record(
        store,
        ManagementRemoveRecordPayload(psk_id=duplicate.psk_id),
        requester_psk_id=mine.psk_id,
    )
    assert payload.result is ManagementResult.OK
    assert effect is ManagementEffect.NONE


# --- server/unpair ------------------------------------------------------------


async def test_unpair_removes_stored_pubkey_record() -> None:
    """server/unpair drops the matched stored-pubkey record."""
    store = InMemoryClientPairingStore()
    record = await _seed_record(store, server_id=SERVER_ID)
    await handle_unpair(store, matched_psk_id=record.psk_id)
    assert await store.record_by_psk_id(record.psk_id) is None


async def test_unpair_keeps_shared_psk_record() -> None:
    """server/unpair must not remove a shared-PSK record (it may back other servers)."""
    store = InMemoryClientPairingStore()
    shared = await _seed_record(store, server_id=None)
    await handle_unpair(store, matched_psk_id=shared.psk_id)
    assert await store.record_by_psk_id(shared.psk_id) is not None


async def test_unpair_unknown_psk_id_is_noop() -> None:
    """server/unpair for an unknown record leaves the store unchanged."""
    store = InMemoryClientPairingStore()
    record = await _seed_record(store, server_id=SERVER_ID)
    await handle_unpair(store, matched_psk_id="missing")
    assert await store.record_by_psk_id(record.psk_id) is not None


# --- get-pairing-config -------------------------------------------------------


async def test_get_pairing_config_projects_state() -> None:
    """get-config reflects enabled flags + round-limit hold-back, omits unimplemented + secrets."""
    store = InMemoryClientPairingStore()
    await store.store_pairing_config(
        replace(
            await store.get_pairing_config(),
            pairing_psk_enabled=False,
            unpaired_access_enabled=True,
        )
    )
    for _ in range(PAIRING_ROUND_LIMIT):
        await store.record_pairing_round()
    payload, effect = await handle_get_pairing_config(
        store, implemented_pair_methods=_WITHOUT_STATIC_PAIRING_CODE
    )
    assert effect is ManagementEffect.NONE
    data = payload.data
    assert data is not None
    assert data.pairing_psk is not None
    assert data.pairing_psk.enabled is False
    assert data.pairing_psk.escalated is None  # not the dynamic-pairing-code method
    assert data.dynamic_pairing_code is not None
    assert data.dynamic_pairing_code.escalated is True
    assert data.static_pairing_code is None  # not implemented
    assert data.unpaired_access is not None
    assert data.unpaired_access.enabled is True
    # The mandatory shared-PSK fallback is always reported.
    assert data.record_mode == RecordModeConfig(
        psk_id=(await store.get_pairing_config()).record_mode_psk_id
    )


async def test_get_pairing_config_shows_static_pairing_code_when_implemented() -> None:
    """A client that implements static pairing code includes it in the config view."""
    store = InMemoryClientPairingStore()
    payload, _ = await handle_get_pairing_config(store, implemented_pair_methods=_ALL_METHODS)
    assert payload.data is not None
    assert payload.data.static_pairing_code is not None
    # Static pairing code has no failure counter.
    assert payload.data.static_pairing_code.escalated is None


# --- set-pairing-config -------------------------------------------------------


async def test_set_pairing_config_toggles_enabled() -> None:
    """Toggling a method's enabled flag persists to the config."""
    store = InMemoryClientPairingStore()
    payload, effect = await handle_set_pairing_config(
        store,
        ManagementSetPairingConfigPayload(pairing_psk=SetPairingPskConfig(enabled=False)),
        implemented_pair_methods=_ALL_METHODS,
    )
    assert payload.result is ManagementResult.OK
    assert effect is ManagementEffect.NONE
    assert (await store.get_pairing_config()).pairing_psk_enabled is False


async def test_set_pairing_config_rotates_psk() -> None:
    """A valid psk replaces the configured Pairing PSK."""
    store = InMemoryClientPairingStore()
    psk = generate_psk()
    payload, _ = await handle_set_pairing_config(
        store,
        ManagementSetPairingConfigPayload(pairing_psk=SetPairingPskConfig(psk=b64url_encode(psk))),
        implemented_pair_methods=_ALL_METHODS,
    )
    assert payload.result is ManagementResult.OK
    stored = await store.pairing_psk()
    assert stored is not None
    assert stored.psk == psk


async def test_set_pairing_config_invalid_psk() -> None:
    """A malformed psk is rejected and nothing is stored."""
    store = InMemoryClientPairingStore()
    payload, _ = await handle_set_pairing_config(
        store,
        ManagementSetPairingConfigPayload(pairing_psk=SetPairingPskConfig(psk="too-short")),
        implemented_pair_methods=_ALL_METHODS,
    )
    assert payload.result is ManagementResult.INVALID
    assert await store.pairing_psk() is None


async def test_set_pairing_config_stores_static_pairing_code() -> None:
    """A valid 8-digit pairing_code is stored as the configured static pairing code."""
    store = InMemoryClientPairingStore()
    payload, _ = await handle_set_pairing_config(
        store,
        ManagementSetPairingConfigPayload(
            static_pairing_code=SetStaticPairingCodeConfig(code="12345678")
        ),
        implemented_pair_methods=_ALL_METHODS,
    )
    assert payload.result is ManagementResult.OK
    assert await store.static_pairing_code() == "12345678"


async def test_set_pairing_config_invalid_static_pairing_code() -> None:
    """A non-8-digit or non-ASCII pairing_code is rejected and nothing is stored."""
    store = InMemoryClientPairingStore()
    for pairing_code in ("12ab", "١٢٣٤٥٦٧٨"):
        payload, _ = await handle_set_pairing_config(
            store,
            ManagementSetPairingConfigPayload(
                static_pairing_code=SetStaticPairingCodeConfig(code=pairing_code)
            ),
            implemented_pair_methods=_ALL_METHODS,
        )
        assert payload.result is ManagementResult.INVALID
    assert await store.static_pairing_code() is None


async def test_get_pairing_config_omits_static_pairing_code_secret() -> None:
    """get-config exposes static-pairing-code policy but never the configured code itself."""
    store = InMemoryClientPairingStore()
    await store.set_static_pairing_code("12345678")
    payload, _ = await handle_get_pairing_config(store, implemented_pair_methods=_ALL_METHODS)
    assert payload.data is not None
    assert payload.data.static_pairing_code is not None
    assert "12345678" not in payload.to_json()


async def test_set_pairing_config_unimplemented_method_is_invalid() -> None:
    """A patch on a method the client does not implement is rejected."""
    store = InMemoryClientPairingStore()
    payload, _ = await handle_set_pairing_config(
        store,
        ManagementSetPairingConfigPayload(
            static_pairing_code=SetStaticPairingCodeConfig(enabled=True)
        ),
        implemented_pair_methods=_WITHOUT_STATIC_PAIRING_CODE,
    )
    assert payload.result is ManagementResult.INVALID


async def test_set_pairing_config_enable_static_pairing_code_without_code_is_invalid() -> None:
    """Enabling static_pairing_code with no code configured is rejected."""
    store = InMemoryClientPairingStore()
    payload, _ = await handle_set_pairing_config(
        store,
        ManagementSetPairingConfigPayload(
            static_pairing_code=SetStaticPairingCodeConfig(enabled=True)
        ),
        implemented_pair_methods=_ALL_METHODS,
    )
    assert payload.result is ManagementResult.INVALID
    assert (await store.get_pairing_config()).static_pairing_code_enabled is False


async def test_set_pairing_config_enable_static_pairing_code_with_code_in_patch() -> None:
    """Enabling static_pairing_code is fine when the same patch provisions the pairing code."""
    store = InMemoryClientPairingStore()
    payload, _ = await handle_set_pairing_config(
        store,
        ManagementSetPairingConfigPayload(
            static_pairing_code=SetStaticPairingCodeConfig(enabled=True, code="12345678")
        ),
        implemented_pair_methods=_ALL_METHODS,
    )
    assert payload.result is ManagementResult.OK
    assert (await store.get_pairing_config()).static_pairing_code_enabled is True
    assert await store.static_pairing_code() == "12345678"


async def test_set_pairing_config_persists_unpaired_access() -> None:
    """An unpaired-access patch is written to the persisted config."""
    store = InMemoryClientPairingStore()
    payload, effect = await handle_set_pairing_config(
        store,
        ManagementSetPairingConfigPayload(unpaired_access=SetUnpairedAccessConfig(enabled=True)),
        implemented_pair_methods=_ALL_METHODS,
    )
    assert payload.result is ManagementResult.OK
    assert effect is ManagementEffect.NONE
    assert (await store.get_pairing_config()).unpaired_access_enabled is True


async def test_set_pairing_config_record_mode_requires_shared_record() -> None:
    """A record_mode referencing a missing/non-shared record is invalid; a shared one applies."""
    store = InMemoryClientPairingStore()
    missing = await handle_set_pairing_config(
        store,
        ManagementSetPairingConfigPayload(record_mode=RecordModeConfig(psk_id="absent")),
        implemented_pair_methods=_ALL_METHODS,
    )
    assert missing[0].result is ManagementResult.INVALID

    stored = await _seed_record(store, server_id="srv-other")
    non_shared = await handle_set_pairing_config(
        store,
        ManagementSetPairingConfigPayload(record_mode=RecordModeConfig(psk_id=stored.psk_id)),
        implemented_pair_methods=_ALL_METHODS,
    )
    assert non_shared[0].result is ManagementResult.INVALID

    shared = await _seed_record(store, server_id=None)
    ok = await handle_set_pairing_config(
        store,
        ManagementSetPairingConfigPayload(record_mode=RecordModeConfig(psk_id=shared.psk_id)),
        implemented_pair_methods=_ALL_METHODS,
    )
    assert ok[0].result is ManagementResult.OK
    assert (await store.get_pairing_config()).record_mode_psk_id == shared.psk_id


# --- open-pairing-window --------------------------------------------------------


async def test_open_pairing_window_opens_when_code_method_offered() -> None:
    """A client offering dynamic pairing code opens the window and reports ok."""
    store = InMemoryClientPairingStore()
    opened: list[None] = []
    payload, effect = await handle_open_pairing_window(
        store,
        implemented_pair_methods=_ALL_METHODS,
        open_window=lambda: opened.append(None),
    )
    assert payload.result is ManagementResult.OK
    assert effect is ManagementEffect.NONE
    assert len(opened) == 1


async def test_open_pairing_window_with_static_pairing_code_enabled() -> None:
    """With dynamic pairing code unavailable, an enabled static_pairing_code qualifies."""
    store = InMemoryClientPairingStore()
    await store.set_static_pairing_code("12345678")
    await store.store_pairing_config(
        replace(await store.get_pairing_config(), static_pairing_code_enabled=True)
    )
    opened: list[None] = []
    payload, _ = await handle_open_pairing_window(
        store,
        implemented_pair_methods=frozenset(
            {PairMethod.PAIRING_PSK, PairMethod.STATIC_PAIRING_CODE}
        ),
        open_window=lambda: opened.append(None),
    )
    assert payload.result is ManagementResult.OK
    assert len(opened) == 1


async def test_open_pairing_window_invalid_when_no_code_method_enabled() -> None:
    """With every pairing-code method disabled or unimplemented, the request is invalid."""
    store = InMemoryClientPairingStore()
    await store.store_pairing_config(
        replace(await store.get_pairing_config(), dynamic_pairing_code_enabled=False)
    )
    opened: list[None] = []
    payload, _ = await handle_open_pairing_window(
        store,
        implemented_pair_methods=_WITHOUT_STATIC_PAIRING_CODE,
        open_window=lambda: opened.append(None),
    )
    assert payload.result is ManagementResult.INVALID
    assert opened == []


# --- storage accounting -------------------------------------------------------


async def test_with_storage_unbounded_store_leaves_payload_unchanged() -> None:
    """A store reporting no accounting (default, unbounded) attaches no storage object."""
    store = InMemoryClientPairingStore()
    payload = ManagementResultPayload(result=ManagementResult.OK)
    result = await with_storage(payload, store, include_static=True)
    assert result.storage is None


async def test_with_storage_free_only_for_mutations() -> None:
    """A non-read result carries only free; capacity and the per-kind costs are absent."""
    store = BoundedClientStore()
    await _seed_record(store, server_id="srv1")
    result = await with_storage(
        ManagementResultPayload(result=ManagementResult.OK), store, include_static=False
    )
    assert result.storage is not None
    # capacity 4, minus the pre-provisioned shared record and the seeded one.
    assert result.storage.free == 2
    assert result.storage.capacity is None
    assert result.storage.cost_individual is None
    assert result.storage.cost_shared is None


async def test_with_storage_full_for_reads() -> None:
    """A read result (include_static) carries free plus capacity and the per-kind costs."""
    store = BoundedClientStore()
    result = await with_storage(
        ManagementResultPayload(result=ManagementResult.OK), store, include_static=True
    )
    # capacity 4, minus the pre-provisioned shared record.
    assert result.storage == StorageAccounting(free=3, capacity=4, cost_individual=1, cost_shared=1)
