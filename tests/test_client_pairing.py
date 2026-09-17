"""Tests for the client's pair-method cross-check (spec server/hello enforcement)."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import pytest
from aiohttp import WSMsgType

from aiosendspin.client.connection import SendspinConnection
from aiosendspin.client.models import PairingSupport, ServerInfo
from aiosendspin.models.core import (
    ActivatePairing,
    ServerActivatePayload,
    ServerTimeMessage,
    ServerTimePayload,
)
from aiosendspin.models.types import (
    Activity,
    GoodbyeReason,
    MediaCommand,
    PairAbortReason,
    PairingCodeFormat,
    PairMethod,
    Roles,
)
from aiosendspin.noise.keys import b64url_encode, generate_psk, psk_id_for
from aiosendspin.noise.models import (
    ClientPairPendingMessage,
    PairAbortMessage,
    PairAbortPayload,
    ServerPairAuthMessage,
    ServerPairAuthPayload,
    ServerPairFinalizeMessage,
)
from aiosendspin.noise.pairing import LocalPairingAbortError, PairingError, RemotePairingAbortError
from aiosendspin.noise.trust_store import (
    PAIRING_ROUND_LIMIT,
    ClientPairingRecord,
    InMemoryClientPairingStore,
    PairingPsk,
    PskCategory,
    ResolvedPsk,
)
from aiosendspin.noise.wire import EncryptedWebSocket

from .conftest import make_sdk_client
from .noise.conftest import make_paired_encrypted_ws

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class _FakeWS:
    """Captures sent text frames; satisfies the bits of EncryptedWebSocket used here."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed = False

    async def send_str(self, data: str) -> None:
        self.sent.append(data)

    async def close(self) -> bool:
        self.closed = True
        return True

    def exception(self) -> BaseException | None:
        return None


def _as_ews(ws: _FakeWS) -> EncryptedWebSocket:
    return cast("EncryptedWebSocket", ws)


def _client_with(category: PskCategory) -> tuple[SendspinConnection, _FakeWS]:
    client = make_sdk_client(client_name="C", roles=[Roles.CONTROLLER])
    connection = SendspinConnection(client)
    ws = _FakeWS()
    connection._ws = ws  # type: ignore[assignment]  # noqa: SLF001
    connection._server_id = "server-1"  # noqa: SLF001
    connection._noise_psk = ResolvedPsk("psk-id", b"\x00" * 32, category)  # noqa: SLF001
    return connection, ws


async def test_pairing_psk_method_accepted_on_pairing_psk() -> None:
    """A Pairing-PSK match with pairing.method=pairing_psk passes the cross-check."""
    connection, ws = _client_with(PskCategory.PAIRING)
    pairing = ActivatePairing(method=PairMethod.PAIRING_PSK)
    assert await connection._validate_pairing(pairing) is pairing  # noqa: SLF001
    assert ws.sent == []


@pytest.mark.parametrize(
    ("category", "method"),
    [
        (PskCategory.PAIRING, PairMethod.DYNAMIC_PAIRING_CODE),  # not offered by this client
        (PskCategory.LONG_TERM, PairMethod.PAIRING_PSK),  # not allowed for long-term PSK
        (PskCategory.PAIRING, None),  # missing when 'pairing' is in activities
    ],
)
async def test_invalid_pair_method_aborts(category: PskCategory, method: PairMethod | None) -> None:
    """A disallowed/unoffered/missing method sends pair/abort and raises."""
    connection, ws = _client_with(category)
    pairing = (
        ActivatePairing(
            method=method,
            format="digits" if method is PairMethod.DYNAMIC_PAIRING_CODE else None,
        )
        if method is not None
        else None
    )
    with pytest.raises(PairingError):
        await connection._validate_pairing(pairing)  # noqa: SLF001
    abort = PairAbortMessage.from_json(ws.sent[0])
    assert abort.payload.reason is PairAbortReason.METHOD_NOT_SUPPORTED


async def test_stray_pairing_frame_is_discarded_quietly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A pairing frame arriving outside an exchange is discarded, not treated as an error."""
    connection, ws = _client_with(PskCategory.LONG_TERM)
    frame = ServerPairAuthMessage(
        payload=ServerPairAuthPayload(pake_msg_1=b64url_encode(b"\x00" * 32)),
    ).to_json()
    with caplog.at_level(logging.DEBUG):
        await connection._handle_json_message(frame)  # noqa: SLF001
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert ws.sent == []


async def test_app_and_time_sends_suppressed_during_exchange() -> None:
    """While an in-band exchange owns the wire, app and time-sync sends are withheld.

    Otherwise they would interleave with the unlocked handshake/pairing sends and desync the
    Noise nonce. Player state is still recorded so the post-exchange resync replays it.
    """
    connection, ws = _client_with(PskCategory.LONG_TERM)
    connection._connected = True  # noqa: SLF001

    connection._exchange_in_progress = True  # noqa: SLF001
    await connection.send_player_state(available=True, volume=7, muted=True)
    await connection.send_group_command(MediaCommand.PLAY)
    await connection._send_time_message()  # noqa: SLF001
    assert ws.sent == []
    assert connection._reported_volume == 7  # noqa: SLF001
    assert connection._reported_muted is True  # noqa: SLF001

    connection._exchange_in_progress = False  # noqa: SLF001
    await connection.send_player_state(available=True, volume=7, muted=True)
    assert len(ws.sent) == 1


async def test_pair_abort_and_goodbye_bypass_exchange_suppression() -> None:
    """pair/abort and client/goodbye still reach the wire while an exchange owns it."""
    connection, ws = _client_with(PskCategory.PAIRING)
    connection._connected = True  # noqa: SLF001
    connection._exchange_in_progress = True  # noqa: SLF001

    await connection.send_pair_abort(PairAbortReason.CONCURRENT_ATTEMPT)
    await connection.send_goodbye(GoodbyeReason.ANOTHER_SERVER)

    assert len(ws.sent) == 2
    abort = PairAbortMessage.from_json(ws.sent[0])
    assert abort.payload.reason is PairAbortReason.CONCURRENT_ATTEMPT


def _pairing_connection(pairing_support: PairingSupport) -> tuple[SendspinConnection, _FakeWS]:
    """Build a Sentinel-keyed connection whose client offers ``pairing_support``."""
    client = make_sdk_client(
        client_name="C",
        roles=[Roles.CONTROLLER],
        pairing_support=pairing_support,
    )
    connection = SendspinConnection(client)
    ws = _FakeWS()
    connection._ws = ws  # type: ignore[assignment]  # noqa: SLF001
    connection._server_id = "server-1"  # noqa: SLF001
    connection._handshake_hash = b"\x00" * 32  # noqa: SLF001
    connection._noise_psk = ResolvedPsk(  # noqa: SLF001
        "psk-id", b"\x00" * 32, PskCategory.SENTINEL
    )
    return connection, ws


def _dynamic_pairing_code_connection() -> tuple[SendspinConnection, _FakeWS]:
    """Build a connection whose client offers the dynamic pairing code."""

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
        pass

    return _pairing_connection(PairingSupport(pairing_code_display=display))


def _static_pairing_code_connection() -> tuple[SendspinConnection, _FakeWS]:
    """Build a connection whose client offers the static pairing code and no dynamic one."""
    return _pairing_connection(PairingSupport())


async def test_dynamic_attempt_at_round_limit_is_held_back() -> None:
    """At the round limit a dynamic attempt signals pair-pending and keeps the count."""
    connection, ws = _dynamic_pairing_code_connection()
    store = connection._client.pairing_store  # noqa: SLF001
    for _ in range(PAIRING_ROUND_LIMIT):
        await store.record_pairing_round()
    connection._selected_pairing = ActivatePairing(  # noqa: SLF001
        method=PairMethod.DYNAMIC_PAIRING_CODE, format="digits"
    )
    queue: asyncio.Queue[object] = asyncio.Queue()
    ws.receive = queue.get  # type: ignore[attr-defined]

    attempt = asyncio.create_task(
        connection._run_pairing_protocol(_as_ews(ws), 1)  # noqa: SLF001
    )
    async with asyncio.timeout(1):
        while not ws.sent:  # noqa: ASYNC110
            await asyncio.sleep(0)
    attempt.cancel()
    with suppress(asyncio.CancelledError):
        await attempt

    assert ClientPairPendingMessage.from_json(ws.sent[0]).payload.pairing_index == 1
    assert await store.pairing_round_count() == PAIRING_ROUND_LIMIT


async def _sent_pair_pending(
    method: PairMethod, pair_pending_message: str | None
) -> dict[str, object]:
    """Run an attempt of ``method`` up to its client/pair-pending and return the sent JSON."""

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
        pass

    connection, ws = _pairing_connection(
        PairingSupport(pairing_code_display=display, pair_pending_message=pair_pending_message)
    )
    store = connection._client.pairing_store  # noqa: SLF001
    if method is PairMethod.STATIC_PAIRING_CODE:
        # Every static attempt is gesture-gated.
        await store.set_static_pairing_code("12345678")
        config = await store.get_pairing_config()
        await store.store_pairing_config(
            replace(config, static_pairing_code_enabled=True, dynamic_pairing_code_enabled=False)
        )
    else:
        # A dynamic attempt is held back at the round limit.
        for _ in range(PAIRING_ROUND_LIMIT):
            await store.record_pairing_round()
    connection._selected_pairing = ActivatePairing(  # noqa: SLF001
        method=method, format="digits" if method is PairMethod.DYNAMIC_PAIRING_CODE else None
    )
    queue: asyncio.Queue[object] = asyncio.Queue()
    ws.receive = queue.get  # type: ignore[attr-defined]

    attempt = asyncio.create_task(
        connection._run_pairing_protocol(_as_ews(ws), 1)  # noqa: SLF001
    )
    async with asyncio.timeout(1):
        while not ws.sent:  # noqa: ASYNC110
            await asyncio.sleep(0)
    attempt.cancel()
    with suppress(asyncio.CancelledError):
        await attempt

    pending: dict[str, object] = json.loads(ws.sent[0])
    assert pending["type"] == "client/pair-pending"
    return pending


@pytest.mark.parametrize(
    "method", [PairMethod.STATIC_PAIRING_CODE, PairMethod.DYNAMIC_PAIRING_CODE]
)
async def test_pair_pending_carries_the_configured_message(method: PairMethod) -> None:
    """The gesture gate and the round-limit hold-back both name what the client waits for."""
    pending = await _sent_pair_pending(method, "Press the pairing button")
    assert pending["payload"] == {"pairing_index": 1, "message": "Press the pairing button"}


@pytest.mark.parametrize(
    "method", [PairMethod.STATIC_PAIRING_CODE, PairMethod.DYNAMIC_PAIRING_CODE]
)
async def test_pair_pending_omits_an_unconfigured_message(method: PairMethod) -> None:
    """Without a configured message the key is left out."""
    pending = await _sent_pair_pending(method, None)
    assert pending["payload"] == {"pairing_index": 1}


def test_pair_pending_message_is_limited_to_200_characters() -> None:
    """A message the spec allows is kept whole; a longer one is refused rather than cut."""
    assert PairingSupport(pair_pending_message="x" * 200).pair_pending_message == "x" * 200
    with pytest.raises(ValueError, match="200 characters"):
        PairingSupport(pair_pending_message="x" * 201)


async def test_ungated_dynamic_attempt_starts_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Below the round limit a dynamic attempt runs without pair-pending or a window."""
    connection, ws = _dynamic_pairing_code_connection()
    connection._selected_pairing = ActivatePairing(  # noqa: SLF001
        method=PairMethod.DYNAMIC_PAIRING_CODE, format="digits"
    )
    captured: dict[str, object] = {}

    async def fake_run(_ws: object, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("aiosendspin.client.connection.run_dynamic_pairing_code_client", fake_run)

    await connection._run_pairing_protocol(_as_ews(ws), 1)  # noqa: SLF001
    assert captured["pairing_format"] is PairingCodeFormat.DIGITS
    assert ws.sent == []  # no pair-pending


async def test_unrecognized_activation_format_aborts() -> None:
    """A format identifier from a newer spec revision is one this client does not offer."""
    connection, ws = _dynamic_pairing_code_connection()
    connection._selected_pairing = ActivatePairing(  # noqa: SLF001
        method=PairMethod.DYNAMIC_PAIRING_CODE, format="holographic"
    )

    with pytest.raises(PairingError):
        await connection._run_pairing_protocol(_as_ews(ws), 1)  # noqa: SLF001

    abort = PairAbortMessage.from_json(ws.sent[0])
    assert abort.payload.reason is PairAbortReason.METHOD_NOT_SUPPORTED


async def test_held_back_attempt_spends_open_window_on_resetting_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a window already open, a held-back attempt skips pair-pending and spends it."""
    connection, ws = _dynamic_pairing_code_connection()
    client = connection._client  # noqa: SLF001
    store = client.pairing_store
    for _ in range(PAIRING_ROUND_LIMIT):
        await store.record_pairing_round()
    client.open_pairing_window()
    connection._selected_pairing = ActivatePairing(  # noqa: SLF001
        method=PairMethod.DYNAMIC_PAIRING_CODE, format="digits"
    )

    async def fake_run(_ws: object, **kwargs: object) -> None:
        pass

    monkeypatch.setattr("aiosendspin.client.connection.run_dynamic_pairing_code_client", fake_run)

    await connection._run_pairing_protocol(_as_ews(ws), 1)  # noqa: SLF001
    assert ws.sent == []  # no pair-pending
    assert not client.pairing_window_open  # the operator action lifted the hold-back
    assert await store.pairing_round_count() == 0  # the operator action resets the count


async def test_ungated_dynamic_attempt_leaves_an_open_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Below the round limit a dynamic attempt neither needs nor spends an open window."""
    connection, ws = _dynamic_pairing_code_connection()
    client = connection._client  # noqa: SLF001
    client.open_pairing_window()
    connection._selected_pairing = ActivatePairing(  # noqa: SLF001
        method=PairMethod.DYNAMIC_PAIRING_CODE, format="digits"
    )

    async def fake_run(_ws: object, **kwargs: object) -> None:
        pass

    monkeypatch.setattr("aiosendspin.client.connection.run_dynamic_pairing_code_client", fake_run)

    await connection._run_pairing_protocol(_as_ews(ws), 1)  # noqa: SLF001
    assert ws.sent == []  # no pair-pending
    assert client.pairing_window_open


async def _static_attempt_connection() -> tuple[SendspinConnection, _FakeWS]:
    """Build a connection whose client has a static pairing code configured and selected."""
    connection, ws = _static_pairing_code_connection()
    store = connection._client.pairing_store  # noqa: SLF001
    await store.set_static_pairing_code("12345678")
    config = await store.get_pairing_config()
    await store.store_pairing_config(replace(config, static_pairing_code_enabled=True))
    connection._selected_pairing = ActivatePairing(  # noqa: SLF001
        method=PairMethod.STATIC_PAIRING_CODE
    )
    return connection, ws


def _sibling_connection(connection: SendspinConnection) -> tuple[SendspinConnection, _FakeWS]:
    """Build another static-pairing-code connection of the same client."""
    sibling = SendspinConnection(connection._client)  # noqa: SLF001
    ws = _FakeWS()
    sibling._ws = ws  # type: ignore[assignment]  # noqa: SLF001
    sibling._server_id = "server-2"  # noqa: SLF001
    sibling._handshake_hash = b"\x00" * 32  # noqa: SLF001
    sibling._noise_psk = connection._noise_psk  # noqa: SLF001
    sibling._selected_pairing = connection._selected_pairing  # noqa: SLF001
    return sibling, ws


def _fake_static_runs(
    monkeypatch: pytest.MonkeyPatch, outcomes: list[BaseException | None]
) -> None:
    """Make each static-pairing-code exchange end with the next outcome (``None``: paired)."""

    async def fake_run(_ws: object, **_kwargs: object) -> None:
        outcome = outcomes.pop(0)
        if outcome is not None:
            raise outcome

    monkeypatch.setattr("aiosendspin.client.connection.run_static_pairing_code_client", fake_run)


async def _run_static_attempt(
    connection: SendspinConnection, ws: _FakeWS, outcome: BaseException | None
) -> None:
    if outcome is None:
        await connection._run_pairing_protocol(_as_ews(ws), 1)  # noqa: SLF001
        return
    with pytest.raises(type(outcome)):
        await connection._run_pairing_protocol(_as_ews(ws), 1)  # noqa: SLF001


async def test_static_pairing_code_attempt_under_an_open_window_closes_it_on_pairing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An attempt under an open window skips pair-pending; its pairing closes the window."""
    connection, ws = await _static_attempt_connection()
    client = connection._client  # noqa: SLF001
    client.open_pairing_window()
    _fake_static_runs(monkeypatch, [None])

    await _run_static_attempt(connection, ws, None)
    assert ws.sent == []  # no pair-pending
    assert not client.pairing_window_open


@pytest.mark.parametrize(
    "outcome",
    [
        pytest.param(LocalPairingAbortError(PairAbortReason.ATTEMPT_TIMEOUT), id="timed_out"),
        pytest.param(RemotePairingAbortError(PairAbortReason.USER_CANCELLED), id="cancelled"),
        pytest.param(
            RemotePairingAbortError(PairAbortReason.PAIRING_CODE_MISMATCH), id="server_kc_ok"
        ),
        pytest.param(asyncio.CancelledError(), id="abandoned"),
        pytest.param(PairingError("malformed"), id="protocol_error"),
    ],
)
async def test_static_attempt_ending_otherwise_keeps_the_window(
    monkeypatch: pytest.MonkeyPatch, outcome: BaseException
) -> None:
    """An attempt that ends without a pairing or a server_kc failure leaves the window open."""
    connection, ws = await _static_attempt_connection()
    client = connection._client  # noqa: SLF001
    client.open_pairing_window()
    _fake_static_runs(monkeypatch, [outcome] * 5)

    for _ in range(5):
        await _run_static_attempt(connection, ws, outcome)
    assert client.pairing_window_admits(connection)
    assert ws.sent == []  # each later attempt ran under the same window


async def test_fifth_server_kc_failure_closes_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Four failed attempts keep the window; the fifth closes it and gates the next attempt."""
    connection, ws = await _static_attempt_connection()
    client = connection._client  # noqa: SLF001
    client.open_pairing_window()
    mismatch = LocalPairingAbortError(PairAbortReason.PAIRING_CODE_MISMATCH)
    _fake_static_runs(monkeypatch, [mismatch] * 5)

    for _ in range(4):
        await _run_static_attempt(connection, ws, mismatch)
    assert client.pairing_window_open
    await _run_static_attempt(connection, ws, mismatch)
    assert not client.pairing_window_open

    client.open_pairing_window()  # a new window starts a fresh count
    _fake_static_runs(monkeypatch, [mismatch])
    await _run_static_attempt(connection, ws, mismatch)
    assert client.pairing_window_open
    assert ws.sent == []


async def test_window_admits_only_the_connection_of_its_first_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Another connection waits until the bound window closes and a new one opens."""
    first, first_ws = await _static_attempt_connection()
    second, second_ws = _sibling_connection(first)
    client = first._client  # noqa: SLF001
    client.open_pairing_window()
    timeout = LocalPairingAbortError(PairAbortReason.ATTEMPT_TIMEOUT)
    _fake_static_runs(monkeypatch, [timeout, None])
    await _run_static_attempt(first, first_ws, timeout)

    queue: asyncio.Queue[object] = asyncio.Queue()
    second_ws.receive = queue.get  # type: ignore[attr-defined]
    waiting = asyncio.create_task(
        second._run_pairing_protocol(_as_ews(second_ws), 1)  # noqa: SLF001
    )
    async with asyncio.timeout(1):
        while not second_ws.sent:  # noqa: ASYNC110
            await asyncio.sleep(0)
    assert ClientPairPendingMessage.from_json(second_ws.sent[0]).payload.pairing_index == 1
    client.open_pairing_window()  # no-op: the window is open, bound to the first connection
    await asyncio.sleep(0)
    assert not waiting.done()

    client.on_connection_closed(first)
    assert not client.pairing_window_open
    await asyncio.sleep(0)
    assert not waiting.done()

    client.open_pairing_window()
    await asyncio.wait_for(waiting, timeout=1)
    assert not client.pairing_window_open  # the second connection paired under it


async def test_close_pairing_window_ends_a_bound_window() -> None:
    """Operator cancellation closes the window and releases its connection."""
    connection, _ws = await _static_attempt_connection()
    sibling, _sibling_ws = _sibling_connection(connection)
    client = connection._client  # noqa: SLF001
    client.open_pairing_window()
    await client.await_pairing_window(connection)
    assert not client.pairing_window_admits(sibling)

    client.close_pairing_window()
    assert not client.pairing_window_open
    client.open_pairing_window()
    assert client.pairing_window_admits(sibling)


async def test_attempt_in_progress_outlives_window_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lifetime runs during an attempt, which completes; the next attempt is gated again."""
    monkeypatch.setattr("aiosendspin.client.client._PAIRING_WINDOW_LIFETIME_S", 0.05)
    connection, ws = await _static_attempt_connection()
    client = connection._client  # noqa: SLF001
    client.open_pairing_window()
    timeout = LocalPairingAbortError(PairAbortReason.ATTEMPT_TIMEOUT)

    async def slow_run(_ws: object, **_kwargs: object) -> None:
        await asyncio.sleep(0.1)
        raise timeout

    monkeypatch.setattr("aiosendspin.client.connection.run_static_pairing_code_client", slow_run)
    await _run_static_attempt(connection, ws, timeout)
    assert ws.sent == []
    assert not client.pairing_window_open

    queue: asyncio.Queue[object] = asyncio.Queue()
    ws.receive = queue.get  # type: ignore[attr-defined]
    gated = asyncio.create_task(
        connection._run_pairing_protocol(_as_ews(ws), 2)  # noqa: SLF001
    )
    async with asyncio.timeout(1):
        while not ws.sent:  # noqa: ASYNC110
            await asyncio.sleep(0)
    gated.cancel()
    with suppress(asyncio.CancelledError):
        await gated
    assert ClientPairPendingMessage.from_json(ws.sent[0]).payload.pairing_index == 2


async def test_open_pairing_window_is_noop_while_open() -> None:
    """Re-opening an open window does not extend its deadline."""
    client = make_sdk_client(client_name="C", roles=[Roles.CONTROLLER])
    client.open_pairing_window()
    deadline = client._pairing_window_deadline  # noqa: SLF001
    client.open_pairing_window()
    assert client._pairing_window_deadline == deadline  # noqa: SLF001


async def test_pairing_window_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    """A window closes silently after its lifetime."""
    monkeypatch.setattr("aiosendspin.client.client._PAIRING_WINDOW_LIFETIME_S", 0.01)
    client = make_sdk_client(client_name="C", roles=[Roles.CONTROLLER])
    client.open_pairing_window()
    assert client.pairing_window_open
    await asyncio.sleep(0.02)
    assert not client.pairing_window_open


async def test_await_pairing_window_prompts_for_gesture() -> None:
    """The wait shows the gesture prompt on entry and clears it once a window opens."""
    prompts: list[bool] = []

    async def prompt(active: bool) -> None:  # noqa: FBT001
        prompts.append(active)

    client = make_sdk_client(
        client_name="C",
        roles=[Roles.CONTROLLER],
        pairing_support=PairingSupport(gesture_prompt=prompt),
    )
    waiter = asyncio.ensure_future(client.await_pairing_window(SendspinConnection(client)))
    await asyncio.sleep(0)
    assert not waiter.done()
    assert prompts == [True]
    client.open_pairing_window()
    await asyncio.wait_for(waiter, timeout=1)
    assert prompts == [True, False]


async def test_await_pairing_window_clears_prompt_on_cancel() -> None:
    """A cancelled wait (the server ended the attempt) still clears the prompt."""
    prompts: list[bool] = []

    async def prompt(active: bool) -> None:  # noqa: FBT001
        prompts.append(active)

    client = make_sdk_client(
        client_name="C",
        roles=[Roles.CONTROLLER],
        pairing_support=PairingSupport(gesture_prompt=prompt),
    )
    waiter = asyncio.ensure_future(client.await_pairing_window(SendspinConnection(client)))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert prompts == [True, False]


async def test_declining_static_pairing_code_drops_it_from_implemented_methods() -> None:
    """A device with no per-device code opts out of static pairing code."""
    wired = make_sdk_client(
        client_name="C", roles=[Roles.CONTROLLER], pairing_support=PairingSupport()
    )
    declined = make_sdk_client(
        client_name="C",
        roles=[Roles.CONTROLLER],
        pairing_support=PairingSupport(offer_static_pairing_code=False),
    )
    assert PairMethod.STATIC_PAIRING_CODE in wired.implemented_pair_methods
    assert PairMethod.STATIC_PAIRING_CODE not in declined.implemented_pair_methods


async def test_hello_descriptors_carry_the_wired_channels_and_locations() -> None:
    """Out-channels follow the wired callbacks, and locations ride the static-secret methods."""

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
        pass

    async def speak(pairing_code: str | None, *, languages: tuple[str, ...]) -> None:
        pass

    client = make_sdk_client(
        client_name="C",
        roles=[Roles.CONTROLLER],
        pairing_support=PairingSupport(
            pairing_code_display=display,
            pairing_code_speaker=speak,
            secret_locations=("device", "leaflet"),
        ),
    )
    connection = SendspinConnection(client)
    connection._noise_psk = ResolvedPsk(  # noqa: SLF001
        "psk-id", b"\x00" * 32, PskCategory.SENTINEL
    )
    hello = await connection._build_client_hello()  # noqa: SLF001
    methods = hello.payload.supported_pair_methods
    assert methods is not None
    assert methods.dynamic_pairing_code is not None
    assert methods.dynamic_pairing_code.out_channels == ["display", "speaker"]
    assert methods.pairing_psk is not None
    assert methods.pairing_psk.locations == ["device", "leaflet"]


async def test_pairing_code_speaker_receives_the_server_hello_languages() -> None:
    """The server/hello language preferences reach the spoken channel, not the activation's."""
    spoken: list[tuple[str | None, tuple[str, ...]]] = []
    displayed: list[str | None] = []

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
        displayed.append(pairing_code)

    async def speak(pairing_code: str | None, *, languages: tuple[str, ...]) -> None:
        spoken.append((pairing_code, languages))

    client = make_sdk_client(
        client_name="C",
        roles=[Roles.CONTROLLER],
        pairing_support=PairingSupport(pairing_code_display=display, pairing_code_speaker=speak),
    )
    connection = SendspinConnection(client)
    connection._server_info = ServerInfo(  # noqa: SLF001
        server_id="server", name="Server", languages=("ca", "en")
    )
    connection._selected_pairing = ActivatePairing(  # noqa: SLF001
        method=PairMethod.DYNAMIC_PAIRING_CODE, format="digits", languages=["es"]
    )
    await connection._emit_pairing_code("123456", pairing_format=PairingCodeFormat.DIGITS)  # noqa: SLF001
    assert spoken == [("123456", ("ca", "en"))]
    assert displayed == ["123456"]


async def test_pairing_code_display_receives_the_grouped_code() -> None:
    """The display gets the raw code with its 3-3 grouping, and ``None`` for both on clear."""
    displayed: list[tuple[str | None, str | None]] = []

    async def display(pairing_code: str | None, *, grouped: str | None) -> None:
        displayed.append((pairing_code, grouped))

    connection, _ws = _pairing_connection(PairingSupport(pairing_code_display=display))
    await connection._emit_pairing_code("123456", pairing_format=PairingCodeFormat.DIGITS)  # noqa: SLF001
    await connection._emit_pairing_code(None, pairing_format=PairingCodeFormat.DIGITS)  # noqa: SLF001
    assert displayed == [("123456", "123-456"), (None, None)]


async def test_pairing_code_speaker_alone_enables_dynamic_pairing_code() -> None:
    """A speaker-only device offers dynamic pairing code, with no display wired."""

    async def speak(pairing_code: str | None, *, languages: tuple[str, ...]) -> None:
        pass

    client = make_sdk_client(
        client_name="C",
        roles=[Roles.CONTROLLER],
        pairing_support=PairingSupport(pairing_code_speaker=speak),
    )
    assert PairMethod.DYNAMIC_PAIRING_CODE in client.implemented_pair_methods
    assert client.pairing_code_out_channels == ("speaker",)


async def test_one_window_is_bound_to_the_first_waiting_connection() -> None:
    """One window releases the first connection's waits; others wait for its successor."""
    client = make_sdk_client(client_name="C", roles=[Roles.CONTROLLER])
    connection = SendspinConnection(client)
    other = SendspinConnection(client)
    first = asyncio.ensure_future(client.await_pairing_window(connection))
    second = asyncio.ensure_future(client.await_pairing_window(other))
    await asyncio.sleep(0)
    client.open_pairing_window()
    await asyncio.wait_for(first, timeout=1)
    await asyncio.sleep(0)
    assert not second.done()
    assert client.pairing_window_open
    await asyncio.wait_for(client.await_pairing_window(connection), timeout=1)
    client.close_pairing_window()
    client.open_pairing_window()
    await asyncio.wait_for(second, timeout=1)


async def test_overlapping_window_waits_share_the_prompt() -> None:
    """Overlapping waits prompt once; the prompt clears only when the last wait ends."""
    prompts: list[bool] = []

    async def prompt(active: bool) -> None:  # noqa: FBT001
        prompts.append(active)

    client = make_sdk_client(
        client_name="C",
        roles=[Roles.CONTROLLER],
        pairing_support=PairingSupport(gesture_prompt=prompt),
    )
    first = asyncio.ensure_future(client.await_pairing_window(SendspinConnection(client)))
    second = asyncio.ensure_future(client.await_pairing_window(SendspinConnection(client)))
    await asyncio.sleep(0)
    assert prompts == [True]
    first.cancel()  # a displaced connection's wait unwinding
    with pytest.raises(asyncio.CancelledError):
        await first
    assert prompts == [True]
    client.open_pairing_window()
    await asyncio.wait_for(second, timeout=1)
    assert prompts == [True, False]


async def test_await_pairing_window_resolves_on_explicit_open() -> None:
    """open_pairing_window (gesture handler or management) satisfies the wait directly."""
    client = make_sdk_client(client_name="C", roles=[Roles.CONTROLLER])
    connection = SendspinConnection(client)
    waiter = asyncio.ensure_future(client.await_pairing_window(connection))
    await asyncio.sleep(0)
    assert not waiter.done()
    client.open_pairing_window()
    await asyncio.wait_for(waiter, timeout=1)
    assert client.pairing_window_open
    assert not client.pairing_window_admits(SendspinConnection(client))


async def _cancel_time_task(connection: SendspinConnection) -> None:
    task = connection._time_task  # noqa: SLF001
    if task is not None:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


def _live_connection(
    category: PskCategory, pairing_support: PairingSupport | None = None
) -> tuple[SendspinConnection, EncryptedWebSocket]:
    """Build a live connection; the returned server end reads what the client sends."""
    client = make_sdk_client(
        client_name="C", roles=[Roles.CONTROLLER], pairing_support=pairing_support
    )
    connection = SendspinConnection(client)
    client_ews, server_ews, _client_raw, _server_raw = make_paired_encrypted_ws()
    connection._ws = client_ews  # noqa: SLF001
    connection._server_id = "server-1"  # noqa: SLF001
    connection._handshake_hash = b"\x00" * 32  # noqa: SLF001
    connection._noise_psk = ResolvedPsk("psk-id", b"\x00" * 32, category)  # noqa: SLF001
    connection._connected = True  # noqa: SLF001
    return connection, server_ews


def _pairing_activation(method: PairMethod) -> ServerActivatePayload:
    return ServerActivatePayload(
        activities=[Activity.PAIRING],
        active_roles=[],
        pairing=ActivatePairing(
            method=method,
            format="digits" if method is PairMethod.DYNAMIC_PAIRING_CODE else None,
        ),
    )


async def _received_types(server_ews: EncryptedWebSocket, count: int) -> list[str]:
    types = []
    async with asyncio.timeout(1):
        for _ in range(count):
            msg = await server_ews.receive()
            assert msg.type is WSMsgType.TEXT
            types.append(json.loads(msg.data)["type"])
    return types


async def test_each_pairing_activation_admits_a_new_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pairing activation mid-attempt abandons it and admits the next one."""
    connection, _server_ews = _live_connection(PskCategory.PAIRING)
    indexes: list[int] = []
    cancelled: list[int] = []

    async def fake_run(_ws: object, *, pairing_index: int, **_kwargs: object) -> None:
        indexes.append(pairing_index)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(pairing_index)
            raise

    monkeypatch.setattr("aiosendspin.client.connection.run_pairing_psk_client", fake_run)
    try:
        activation = _pairing_activation(PairMethod.PAIRING_PSK)
        await connection._handle_server_activate(activation)  # noqa: SLF001
        await asyncio.sleep(0)
        await connection._handle_server_activate(activation)  # noqa: SLF001
        await asyncio.sleep(0)
        assert indexes == [1, 2]
        assert cancelled == [1]
        assert connection._pairing_task is not None  # noqa: SLF001
    finally:
        await connection.disconnect()
    assert cancelled == [1, 2]


async def test_server_activate_mid_attempt_cancels_it_and_persists_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A leave activation after client/pair-finalize abandons the attempt without storing."""
    connection, server_ews = _live_connection(PskCategory.PAIRING)
    store = connection._client.pairing_store  # noqa: SLF001
    pairing = generate_psk()
    await store.set_pairing_psk(PairingPsk(psk_id=psk_id_for(pairing), psk=pairing))
    try:
        await connection._handle_server_activate(  # noqa: SLF001
            _pairing_activation(PairMethod.PAIRING_PSK)
        )
        assert await _received_types(server_ews, 2) == ["client/pair-init", "client/pair-finalize"]
        assert connection.pairing_attempt_in_progress

        await connection._handle_server_activate(  # noqa: SLF001
            ServerActivatePayload(activities=[], active_roles=[])
        )
        assert connection._pairing_task is None  # noqa: SLF001
        assert not connection.pairing_attempt_in_progress
        assert not connection.is_pairing

        # The ack the server sent before it saw nothing further is discarded quietly.
        with caplog.at_level(logging.DEBUG):
            await connection._handle_json_message(  # noqa: SLF001
                ServerPairFinalizeMessage().to_json()
            )
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert connection.connected
        assert await store.record_by_server_id("server-1") is None
    finally:
        await connection.disconnect()


async def test_finalize_ack_persists_before_the_reader_moves_on() -> None:
    """The reader hands server/pair-finalize over and waits until the record is stored."""
    connection, server_ews = _live_connection(PskCategory.PAIRING)
    store = connection._client.pairing_store  # noqa: SLF001
    pairing = generate_psk()
    await store.set_pairing_psk(PairingPsk(psk_id=psk_id_for(pairing), psk=pairing))
    try:
        await connection._handle_server_activate(  # noqa: SLF001
            _pairing_activation(PairMethod.PAIRING_PSK)
        )
        await _received_types(server_ews, 2)

        await connection._handle_json_message(  # noqa: SLF001
            ServerPairFinalizeMessage().to_json()
        )

        assert connection._pairing_task is None  # noqa: SLF001
        assert await store.record_by_server_id("server-1") is not None
    finally:
        await connection.disconnect()


async def test_malformed_pairing_message_fails_the_attempt_at_once() -> None:
    """A pairing message that does not parse reaches the attempt, which fails on it."""

    async def display(_pairing_code: str | None, **_kwargs: object) -> None:
        return

    connection, server_ews = _live_connection(
        PskCategory.SENTINEL, PairingSupport(pairing_code_display=display)
    )
    try:
        await connection._handle_server_activate(  # noqa: SLF001
            _pairing_activation(PairMethod.DYNAMIC_PAIRING_CODE)
        )
        assert await _received_types(server_ews, 1) == ["client/pair-init"]

        await connection._handle_json_message(  # noqa: SLF001
            json.dumps({"type": "server/pair-auth", "payload": {}})
        )
        async with asyncio.timeout(1):
            while connection._pairing_task is not None:  # noqa: ASYNC110, SLF001
                await asyncio.sleep(0)

        assert not connection.connected
    finally:
        await connection.disconnect()


async def test_attempt_runs_alongside_other_traffic(monkeypatch: pytest.MonkeyPatch) -> None:
    """During an attempt time sync flows both ways and pairing messages go to the attempt."""
    connection, server_ews = _live_connection(PskCategory.PAIRING)
    received: asyncio.Queue[str] = asyncio.Queue()

    async def fake_run(ws: EncryptedWebSocket, **_kwargs: object) -> None:
        msg = await ws.receive()
        await received.put(msg.data)
        await asyncio.Event().wait()

    monkeypatch.setattr("aiosendspin.client.connection.run_pairing_psk_client", fake_run)
    try:
        await connection._handle_server_activate(  # noqa: SLF001
            _pairing_activation(PairMethod.PAIRING_PSK)
        )
        connection._pairing_attempt_in_progress = True  # noqa: SLF001

        await connection._send_time_message()  # noqa: SLF001
        assert await _received_types(server_ews, 1) == ["client/time"]

        now_us = connection.now_us()
        time_reply = ServerTimeMessage(
            payload=ServerTimePayload(
                client_transmitted=now_us, server_received=now_us, server_transmitted=now_us
            )
        )
        await connection._handle_json_message(time_reply.to_json())  # noqa: SLF001
        assert connection._time_filter.count == 1  # noqa: SLF001

        abort = PairAbortMessage(payload=PairAbortPayload(reason=PairAbortReason.USER_CANCELLED))
        await connection._handle_json_message(abort.to_json())  # noqa: SLF001
        async with asyncio.timeout(1):
            assert await received.get() == abort.to_json()
    finally:
        await connection.disconnect()


async def test_remote_abort_leaves_the_connection_in_pairing() -> None:
    """A non-closing pair/abort ends the attempt only; later pairing frames are discarded."""
    connection, _server_ews = _live_connection(PskCategory.PAIRING)
    reasons: list[PairAbortReason] = []
    connection._client.add_pairing_abort_listener(reasons.append)  # noqa: SLF001
    pairing = generate_psk()
    store = connection._client.pairing_store  # noqa: SLF001
    await store.set_pairing_psk(PairingPsk(psk_id=psk_id_for(pairing), psk=pairing))
    try:
        await connection._handle_server_activate(  # noqa: SLF001
            _pairing_activation(PairMethod.PAIRING_PSK)
        )
        abort = PairAbortMessage(payload=PairAbortPayload(reason=PairAbortReason.USER_CANCELLED))
        await connection._handle_json_message(abort.to_json())  # noqa: SLF001
        task = connection._pairing_task  # noqa: SLF001
        if task is not None:
            await asyncio.wait((task,))

        assert reasons == [PairAbortReason.USER_CANCELLED]
        assert connection._pairing_task is None  # noqa: SLF001
        assert connection.is_pairing
        assert connection.connected
        await connection._handle_json_message(ServerPairFinalizeMessage().to_json())  # noqa: SLF001
        assert await store.record_by_server_id("server-1") is None
    finally:
        await connection.disconnect()


class _CancelObserver:
    """Records what a cancelled attempt does to the operator-facing callbacks."""

    def __init__(self) -> None:
        self.displayed: list[tuple[str | None, str | None]] = []
        self.suspended: list[bool] = []
        self.prompts: list[bool] = []
        self.aborts: list[PairAbortReason] = []

    async def display(self, pairing_code: str | None, *, grouped: str | None) -> None:
        self.displayed.append((pairing_code, grouped))

    async def suspend(self, active: bool) -> None:  # noqa: FBT001
        self.suspended.append(active)

    async def prompt(self, active: bool) -> None:  # noqa: FBT001
        self.prompts.append(active)

    def support(self) -> PairingSupport:
        return PairingSupport(
            gesture_prompt=self.prompt,
            pairing_code_display=self.display,
            out_channel_suspend=self.suspend,
        )


def _admitted_live_connection(
    observer: _CancelObserver,
) -> tuple[SendspinConnection, EncryptedWebSocket]:
    """Build a live Sentinel connection admitted by a client wired to ``observer``."""
    connection, server_ews = _live_connection(PskCategory.SENTINEL, observer.support())
    client = connection._client  # noqa: SLF001
    client._admitted_connection = connection  # noqa: SLF001
    client.add_pairing_abort_listener(observer.aborts.append)
    return connection, server_ews


async def _next_non_time_message(server_ews: EncryptedWebSocket) -> dict[str, Any]:
    """Return the next JSON message the client sent, skipping time sync."""
    async with asyncio.timeout(1):
        while True:
            msg = await server_ews.receive()
            assert msg.type is WSMsgType.TEXT
            message: dict[str, Any] = json.loads(msg.data)
            if message["type"] != "client/time":
                return message


async def _assert_cancelled(connection: SendspinConnection, server_ews: EncryptedWebSocket) -> None:
    """Cancel the attempt and check the abort, the open connection and the closed window."""
    client = connection._client  # noqa: SLF001
    # Concurrent cancellations send a single abort.
    await asyncio.gather(client.cancel_pairing(), client.cancel_pairing())

    assert await _next_non_time_message(server_ews) == {
        "type": "pair/abort",
        "payload": {"reason": "user_cancelled"},
    }
    assert connection._pairing_task is None  # noqa: SLF001
    assert connection.connected
    assert connection.is_pairing
    assert not client.pairing_window_open

    # A pairing message the server sent before seeing the abort is discarded.
    await connection._handle_json_message(ServerPairFinalizeMessage().to_json())  # noqa: SLF001
    assert connection.connected

    await connection.send_goodbye(GoodbyeReason.SHUTDOWN)
    assert (await _next_non_time_message(server_ews))["type"] == "client/goodbye"


async def test_cancel_pairing_ends_a_started_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    """A started attempt is aborted with user_cancelled and its displayed code is cleared."""
    observer = _CancelObserver()
    connection, server_ews = _admitted_live_connection(observer)
    client = connection._client  # noqa: SLF001
    shown = asyncio.Event()

    async def fake_run(_ws: object, *, pairing_code_emitter: object, **_kwargs: object) -> None:
        emit = cast("Callable[[str], Awaitable[None]]", pairing_code_emitter)
        await emit("123456")
        shown.set()
        await asyncio.Event().wait()

    monkeypatch.setattr("aiosendspin.client.connection.run_dynamic_pairing_code_client", fake_run)
    try:
        client.open_pairing_window()
        await connection._handle_server_activate(  # noqa: SLF001
            _pairing_activation(PairMethod.DYNAMIC_PAIRING_CODE)
        )
        async with asyncio.timeout(1):
            await shown.wait()

        await _assert_cancelled(connection, server_ews)

        assert observer.displayed == [("123456", "123-456"), (None, None)]
        assert observer.suspended == [True, False]
        assert observer.prompts == []
        assert observer.aborts == []
    finally:
        await connection.disconnect()


@pytest.mark.parametrize(
    "method", [PairMethod.DYNAMIC_PAIRING_CODE, PairMethod.STATIC_PAIRING_CODE]
)
async def test_cancel_pairing_ends_an_attempt_awaiting_a_window(method: PairMethod) -> None:
    """A gated attempt that sent client/pair-pending is aborted and its prompt cleared."""
    observer = _CancelObserver()
    connection, server_ews = _admitted_live_connection(observer)
    store = connection._client.pairing_store  # noqa: SLF001
    if method is PairMethod.STATIC_PAIRING_CODE:
        await store.set_static_pairing_code("12345678")
        config = await store.get_pairing_config()
        await store.store_pairing_config(
            replace(config, static_pairing_code_enabled=True, dynamic_pairing_code_enabled=False)
        )
    else:
        for _ in range(PAIRING_ROUND_LIMIT):
            await store.record_pairing_round()
    try:
        await connection._handle_server_activate(_pairing_activation(method))  # noqa: SLF001
        assert (await _next_non_time_message(server_ews))["type"] == "client/pair-pending"
        async with asyncio.timeout(1):
            while not observer.prompts:  # noqa: ASYNC110
                await asyncio.sleep(0)
        assert observer.prompts == [True]

        await _assert_cancelled(connection, server_ews)

        assert observer.prompts == [True, False]
        if method is PairMethod.DYNAMIC_PAIRING_CODE:
            assert observer.displayed == [(None, None)]
        else:
            assert observer.displayed == []
        assert observer.suspended == []
        assert observer.aborts == []
    finally:
        await connection.disconnect()


async def test_cancel_pairing_after_finalize_lets_the_attempt_complete() -> None:
    """Once client/pair-finalize is out the server may have stored the record, so it pairs."""
    observer = _CancelObserver()
    connection, server_ews = _admitted_live_connection(observer)
    connection._noise_psk = ResolvedPsk(  # noqa: SLF001
        "psk-id", b"\x00" * 32, PskCategory.PAIRING
    )
    client = connection._client  # noqa: SLF001
    store = client.pairing_store
    pairing = generate_psk()
    await store.set_pairing_psk(PairingPsk(psk_id=psk_id_for(pairing), psk=pairing))
    try:
        await connection._handle_server_activate(  # noqa: SLF001
            _pairing_activation(PairMethod.PAIRING_PSK)
        )
        assert (await _next_non_time_message(server_ews))["type"] == "client/pair-init"
        assert (await _next_non_time_message(server_ews))["type"] == "client/pair-finalize"

        await client.cancel_pairing()
        assert connection._pairing_task is not None  # noqa: SLF001

        await connection._handle_json_message(ServerPairFinalizeMessage().to_json())  # noqa: SLF001
        await connection.send_goodbye(GoodbyeReason.SHUTDOWN)

        assert connection._pairing_task is None  # noqa: SLF001
        assert await store.record_by_server_id("server-1") is not None
        # No pair/abort went out before the goodbye.
        assert (await _next_non_time_message(server_ews))["type"] == "client/goodbye"
    finally:
        await connection.disconnect()


async def test_cancel_pairing_while_releasing_the_code_lets_the_release_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An attempt the server ended is clearing its display; a cancel does not cut that short."""
    observer = _CancelObserver()
    clearing = asyncio.Event()
    release = asyncio.Event()

    async def display(pairing_code: str | None, *, grouped: str | None) -> None:
        if pairing_code is None:
            clearing.set()
            await release.wait()
        observer.displayed.append((pairing_code, grouped))

    monkeypatch.setattr(observer, "display", display)
    connection, server_ews = _admitted_live_connection(observer)
    client = connection._client  # noqa: SLF001

    async def fake_run(_ws: object, *, pairing_code_emitter: object, **_kwargs: object) -> None:
        emit = cast("Callable[[str], Awaitable[None]]", pairing_code_emitter)
        await emit("123456")
        raise RemotePairingAbortError(PairAbortReason.PAIRING_CODE_MISMATCH)

    monkeypatch.setattr("aiosendspin.client.connection.run_dynamic_pairing_code_client", fake_run)
    try:
        await connection._handle_server_activate(  # noqa: SLF001
            _pairing_activation(PairMethod.DYNAMIC_PAIRING_CODE)
        )
        async with asyncio.timeout(1):
            await clearing.wait()

        await client.cancel_pairing()
        release.set()
        task = connection._pairing_task  # noqa: SLF001
        if task is not None:
            await asyncio.wait((task,))
        await connection.send_goodbye(GoodbyeReason.SHUTDOWN)

        assert observer.displayed == [("123456", "123-456"), (None, None)]
        assert observer.suspended == [True, False]
        assert observer.aborts == [PairAbortReason.PAIRING_CODE_MISMATCH]
        assert (await _next_non_time_message(server_ews))["type"] == "client/goodbye"
    finally:
        await connection.disconnect()


async def test_cancel_pairing_without_an_attempt_is_a_noop() -> None:
    """With no connection, or no attempt on it, nothing is sent and nothing changes."""
    observer = _CancelObserver()
    connection, server_ews = _admitted_live_connection(observer)
    client = connection._client  # noqa: SLF001
    client._admitted_connection = None  # noqa: SLF001
    client.open_pairing_window()
    await client.cancel_pairing()

    client._admitted_connection = connection  # noqa: SLF001
    try:
        await client.cancel_pairing()
        await connection.send_goodbye(GoodbyeReason.SHUTDOWN)

        # The goodbye is the first frame the server sees: no pair/abort went before it.
        assert (await _next_non_time_message(server_ews))["type"] == "client/goodbye"
        assert client.pairing_window_open
        assert observer.displayed == []
        assert connection.connected
    finally:
        await connection.disconnect()


async def test_out_channel_resumes_when_releasing_the_code_fails() -> None:
    """A failing release of the out-channel still resumes the suspended output."""
    events: list[bool] = []

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
        if pairing_code is None:
            raise RuntimeError("display gone")

    async def suspend(active: bool) -> None:  # noqa: FBT001
        events.append(active)

    connection, _ws = _pairing_connection(
        PairingSupport(pairing_code_display=display, out_channel_suspend=suspend)
    )
    await connection._emit_pairing_code("123456", pairing_format=PairingCodeFormat.DIGITS)  # noqa: SLF001
    with pytest.raises(RuntimeError, match="display gone"):
        await connection._emit_pairing_code(None, pairing_format=PairingCodeFormat.DIGITS)  # noqa: SLF001

    assert events == [True, False]


async def test_out_channel_is_suspended_while_the_code_is_emitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The suspend hook brackets the dynamic pairing code's emission."""
    events: list[object] = []

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
        events.append(pairing_code)

    async def suspend(active: bool) -> None:  # noqa: FBT001
        events.append(active)

    connection, ws = _pairing_connection(
        PairingSupport(pairing_code_display=display, out_channel_suspend=suspend)
    )
    connection._selected_pairing = ActivatePairing(  # noqa: SLF001
        method=PairMethod.DYNAMIC_PAIRING_CODE, format="digits"
    )

    async def fake_run(_ws: object, *, pairing_code_emitter: object, **_kwargs: object) -> None:
        emit = cast("Callable[[str], Awaitable[None]]", pairing_code_emitter)
        await emit("123456")
        await emit("123456")  # the next round keeps the channel suspended
        raise RemotePairingAbortError(PairAbortReason.PAIRING_CODE_MISMATCH)

    monkeypatch.setattr("aiosendspin.client.connection.run_dynamic_pairing_code_client", fake_run)

    with pytest.raises(RemotePairingAbortError):
        await connection._run_pairing_protocol(_as_ews(ws), 1)  # noqa: SLF001

    assert events == [True, "123456", "123456", None, False]


async def test_leave_activate_resumes_time_sync() -> None:
    """A server/activate that returns the connection to normal service restarts time sync."""
    connection, _ws = _client_with(PskCategory.LONG_TERM)
    connection._connected = True  # noqa: SLF001
    assert connection._time_task is None  # noqa: SLF001

    try:
        await connection._handle_server_activate(  # noqa: SLF001
            ServerActivatePayload(activities=[], active_roles=[])
        )
        assert connection._time_task is not None  # noqa: SLF001
        assert not connection._time_task.done()  # noqa: SLF001
    finally:
        await _cancel_time_task(connection)


async def test_resolution_answers_within_the_declared_category() -> None:
    """One psk_id held under two categories resolves to the one the server declared.

    The store defines a record as taking precedence over a same-id Pairing PSK, so a
    reader that resolved first and checked the category afterwards would call a pairing
    handshake a lookup miss while holding the very credential it named.
    """
    psk = generate_psk()
    shared_id = psk_id_for(psk)
    store = InMemoryClientPairingStore()
    await store.set_pairing_psk(PairingPsk(psk_id=shared_id, psk=psk))
    await store.store_record(ClientPairingRecord(psk_id=shared_id, psk=psk, server_id="server-X"))

    client = make_sdk_client(client_name="C", roles=[Roles.CONTROLLER], pairing_store=store)
    connection = SendspinConnection(client)

    pairing = await connection._resolve_psk(shared_id, PskCategory.PAIRING)  # noqa: SLF001
    assert pairing is not None
    assert pairing.category is PskCategory.PAIRING

    long_term = await connection._resolve_psk(shared_id, PskCategory.LONG_TERM)  # noqa: SLF001
    assert long_term is not None
    assert long_term.category is PskCategory.LONG_TERM


async def test_post_pairing_activation_sends_stateless_initial_state() -> None:
    """Pairing that ends with only stateless roles active sends the initial client/state."""
    connection, ws = _client_with(PskCategory.LONG_TERM)
    connection._connected = True  # noqa: SLF001

    try:
        # The activation following the re-handshake onto the new record.
        await connection._handle_server_activate(  # noqa: SLF001
            ServerActivatePayload(activities=[], active_roles=[Roles.CONTROLLER.value]),
            resync=True,
        )
        states = [msg for msg in map(json.loads, ws.sent) if msg["type"] == "client/state"]
        assert [msg["payload"] for msg in states] == [{"available": True}]
    finally:
        await _cancel_time_task(connection)


async def _connection_offering_both_code_methods() -> SendspinConnection:
    """Build a connection whose config enables both pairing-code methods."""

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
        pass

    client = make_sdk_client(
        client_name="C",
        roles=[Roles.CONTROLLER],
        pairing_support=PairingSupport(pairing_code_display=display),
    )
    await client.pairing_store.set_static_pairing_code("12345678")
    config = await client.pairing_store.get_pairing_config()
    await client.pairing_store.store_pairing_config(
        replace(config, static_pairing_code_enabled=True, dynamic_pairing_code_enabled=True)
    )
    connection = SendspinConnection(client)
    connection._noise_psk = ResolvedPsk(  # noqa: SLF001
        "psk-id", b"\x00" * 32, PskCategory.SENTINEL
    )
    return connection


async def test_both_code_methods_wired_advertises_dynamic_only() -> None:
    """The client offers only one pairing-code method, and the per-session one wins."""
    connection = await _connection_offering_both_code_methods()

    hello = await connection._build_client_hello()  # noqa: SLF001

    methods = hello.payload.supported_pair_methods
    assert methods is not None
    assert methods.dynamic_pairing_code is not None
    assert methods.static_pairing_code is None
    assert methods.pairing_psk is not None


async def test_static_pairing_is_refused_once_it_is_no_longer_offered() -> None:
    """Dropping static from the advertisement also refuses a server that selects it."""
    connection = await _connection_offering_both_code_methods()
    connection._ws = _FakeWS()  # type: ignore[assignment]  # noqa: SLF001

    with pytest.raises(LocalPairingAbortError, match="method_not_supported"):
        await connection._validate_pairing(  # noqa: SLF001
            ActivatePairing(method=PairMethod.STATIC_PAIRING_CODE)
        )
