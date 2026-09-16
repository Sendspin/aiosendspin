"""Tests for the client's pair-method cross-check (spec server/hello enforcement)."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from dataclasses import replace
from typing import TYPE_CHECKING, cast

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

    async def display(pairing_code: str | None) -> None:
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


async def test_held_back_attempt_consumes_open_window_and_resets_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a window already open, a held-back attempt skips pair-pending, consumes it, resets."""
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
    assert not client.pairing_window_open  # consumed by the attempt
    assert await store.pairing_round_count() == 0  # the operator action resets the count


async def test_static_pairing_code_attempt_consumes_a_pre_open_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A static-pairing-code attempt spends a window that is already open."""
    connection, ws = _static_pairing_code_connection()
    client = connection._client  # noqa: SLF001
    await client.pairing_store.set_static_pairing_code("12345678")
    config = await client.pairing_store.get_pairing_config()
    await client.pairing_store.store_pairing_config(
        replace(config, static_pairing_code_enabled=True)
    )
    client.open_pairing_window()
    connection._selected_pairing = ActivatePairing(  # noqa: SLF001
        method=PairMethod.STATIC_PAIRING_CODE
    )

    async def fake_run(_ws: object, **kwargs: object) -> None:
        pass

    monkeypatch.setattr("aiosendspin.client.connection.run_static_pairing_code_client", fake_run)

    await connection._run_pairing_protocol(_as_ews(ws), 1)  # noqa: SLF001
    assert ws.sent == []  # no pair-pending
    assert not client.pairing_window_open


async def test_ungated_attempt_consumes_open_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ungated attempt still spends an open window: its lifetime ends at pair-init."""
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
    assert not client.pairing_window_open  # spent by the attempt


async def test_open_pairing_window_is_noop_while_open() -> None:
    """Re-opening an open window does not extend its deadline."""
    client = make_sdk_client(client_name="C", roles=[Roles.CONTROLLER])
    client.open_pairing_window()
    deadline = client._pairing_window_deadline  # noqa: SLF001
    client.open_pairing_window()
    assert client._pairing_window_deadline == deadline  # noqa: SLF001


async def test_pairing_window_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unconsumed window closes silently after its lifetime."""
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
    waiter = asyncio.ensure_future(client.await_pairing_window())
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
    waiter = asyncio.ensure_future(client.await_pairing_window())
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

    async def display(pairing_code: str | None) -> None:
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

    async def display(pairing_code: str | None) -> None:
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


async def test_one_window_admits_a_single_attempt() -> None:
    """One window releases one waiter, the rest wait for a fresh gesture."""
    client = make_sdk_client(client_name="C", roles=[Roles.CONTROLLER])
    first = asyncio.ensure_future(client.await_pairing_window())
    second = asyncio.ensure_future(client.await_pairing_window())
    await asyncio.sleep(0)
    client.open_pairing_window()
    await asyncio.wait_for(first, timeout=1)
    await asyncio.sleep(0)
    assert not second.done()
    assert not client.pairing_window_open
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
    first = asyncio.ensure_future(client.await_pairing_window())
    second = asyncio.ensure_future(client.await_pairing_window())
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
    waiter = asyncio.ensure_future(client.await_pairing_window())
    await asyncio.sleep(0)
    assert not waiter.done()
    client.open_pairing_window()
    await asyncio.wait_for(waiter, timeout=1)
    assert not client.pairing_window_open


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


async def test_out_channel_is_suspended_while_the_code_is_emitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The suspend hook brackets the dynamic pairing code's emission."""
    events: list[object] = []

    async def display(pairing_code: str | None) -> None:
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

    async def display(pairing_code: str | None) -> None:
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
