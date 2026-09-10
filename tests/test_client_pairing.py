"""Tests for the client's pair-method cross-check (spec server/hello enforcement)."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from dataclasses import replace

import pytest
from aiohttp import WSMessage, WSMsgType

from aiosendspin.client.connection import SendspinConnection
from aiosendspin.client.models import PairingSupport
from aiosendspin.models.core import ActivatePairing, ServerActivateMessage, ServerActivatePayload
from aiosendspin.models.types import (
    Activity,
    GoodbyeReason,
    MediaCommand,
    PairAbortReason,
    PairingCodeFormat,
    PairMethod,
    Roles,
)
from aiosendspin.noise.keys import b64url_encode
from aiosendspin.noise.models import (
    ClientPairPendingMessage,
    PairAbortMessage,
    ServerPairAuthMessage,
    ServerPairAuthPayload,
)
from aiosendspin.noise.pairing import LocalPairingAbortError, PairingError
from aiosendspin.noise.trust_store import (
    PAIRING_CODE_ESCALATION_THRESHOLD,
    PskCategory,
    ResolvedPsk,
)

from .conftest import make_sdk_client


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


async def test_pairing_window_tolerates_bare_leave_activate() -> None:
    """A bare leave server/activate during the window wait ends pairing, not the connection."""
    client = make_sdk_client(client_name="C", roles=[Roles.CONTROLLER])
    connection = SendspinConnection(client)
    ws = _FakeWS()
    leave = ServerActivateMessage(
        payload=ServerActivatePayload(activities=[], active_roles=[])
    ).to_json()

    async def receive() -> WSMessage:
        return WSMessage(WSMsgType.TEXT, leave, "")

    ws.receive = receive  # type: ignore[attr-defined]
    connection._ws = ws  # type: ignore[assignment]  # noqa: SLF001
    connection._server_id = "server-1"  # noqa: SLF001
    # Gesture-gated pairing runs over the Sentinel PSK.
    connection._noise_psk = ResolvedPsk(  # noqa: SLF001
        "psk-id", b"\x00" * 32, PskCategory.SENTINEL
    )

    frame = await connection._gate_on_pairing_window(1)  # noqa: SLF001

    # The bare leave activate is surfaced raw for downstream parsing, not raised.
    assert frame == leave
    assert ClientPairPendingMessage.from_json(ws.sent[0]).payload.pairing_index == 1


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


async def test_escalated_dynamic_attempt_is_gesture_gated() -> None:
    """Once escalated, even a dynamic pairing-code attempt signals pair-pending."""
    connection, ws = _dynamic_pairing_code_connection()
    store = connection._client.pairing_store  # noqa: SLF001
    for _ in range(PAIRING_CODE_ESCALATION_THRESHOLD):
        await store.record_pairing_code_failure()
    connection._selected_pairing = ActivatePairing(  # noqa: SLF001
        method=PairMethod.DYNAMIC_PAIRING_CODE, format="digits"
    )
    leave = ServerActivateMessage(
        payload=ServerActivatePayload(activities=[], active_roles=[])
    ).to_json()

    async def receive() -> WSMessage:
        return WSMessage(WSMsgType.TEXT, leave, "")

    ws.receive = receive  # type: ignore[attr-defined]

    frame = await connection._run_pairing_protocol()  # noqa: SLF001

    assert frame == leave
    assert ClientPairPendingMessage.from_json(ws.sent[0]).payload.pairing_index == 1


async def test_ungated_dynamic_attempt_starts_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unescalated dynamic attempt runs without pair-pending or a window."""
    connection, ws = _dynamic_pairing_code_connection()
    connection._selected_pairing = ActivatePairing(  # noqa: SLF001
        method=PairMethod.DYNAMIC_PAIRING_CODE, format="digits"
    )
    captured: dict[str, object] = {}

    async def fake_run(_ws: object, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("aiosendspin.client.connection.run_dynamic_pairing_code_client", fake_run)

    assert await connection._run_pairing_protocol() is None  # noqa: SLF001
    assert captured["pairing_format"] is PairingCodeFormat.DIGITS
    assert ws.sent == []  # no pair-pending


async def test_unrecognized_activation_format_aborts() -> None:
    """A format identifier from a newer spec revision is one this client does not offer."""
    connection, ws = _dynamic_pairing_code_connection()
    connection._selected_pairing = ActivatePairing(  # noqa: SLF001
        method=PairMethod.DYNAMIC_PAIRING_CODE, format="holographic"
    )

    with pytest.raises(PairingError):
        await connection._run_pairing_protocol()  # noqa: SLF001

    abort = PairAbortMessage.from_json(ws.sent[0])
    assert abort.payload.reason is PairAbortReason.METHOD_NOT_SUPPORTED


async def test_gated_attempt_consumes_open_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a window already open, a gated attempt skips pair-pending and consumes it."""
    connection, ws = _dynamic_pairing_code_connection()
    client = connection._client  # noqa: SLF001
    store = client.pairing_store
    for _ in range(PAIRING_CODE_ESCALATION_THRESHOLD):
        await store.record_pairing_code_failure()
    client.open_pairing_window()
    connection._selected_pairing = ActivatePairing(  # noqa: SLF001
        method=PairMethod.DYNAMIC_PAIRING_CODE, format="digits"
    )

    async def fake_run(_ws: object, **kwargs: object) -> None:
        pass

    monkeypatch.setattr("aiosendspin.client.connection.run_dynamic_pairing_code_client", fake_run)

    assert await connection._run_pairing_protocol() is None  # noqa: SLF001
    assert ws.sent == []  # no pair-pending
    assert not client.pairing_window_open  # consumed by the attempt


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

    assert await connection._run_pairing_protocol() is None  # noqa: SLF001
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

    assert await connection._run_pairing_protocol() is None  # noqa: SLF001
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


async def test_pairing_code_speaker_receives_the_activation_languages() -> None:
    """The activation's language preferences reach the spoken channel, which the display omits."""
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
    connection._selected_pairing = ActivatePairing(  # noqa: SLF001
        method=PairMethod.DYNAMIC_PAIRING_CODE, format="digits", languages=["ca", "en"]
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


async def test_leave_activate_redeclaring_pairing_runs_next_attempt() -> None:
    """A leave activate that declares pairing again immediately admits the next attempt."""
    connection, _ws = _client_with(PskCategory.LONG_TERM)
    connection._connected = True  # noqa: SLF001
    attempts = 0
    activates = iter(
        [
            ServerActivatePayload(
                activities=[Activity.PAIRING],
                active_roles=[],
                pairing=ActivatePairing(method=PairMethod.DYNAMIC_PAIRING_CODE, format="digits"),
            ),
            ServerActivatePayload(activities=[], active_roles=[]),
        ]
    )

    async def fake_protocol() -> str:
        nonlocal attempts
        attempts += 1
        return "leftover"

    async def resolve(leftover: str | None) -> ServerActivatePayload:  # noqa: ARG001
        return next(activates)

    connection._run_pairing_protocol = fake_protocol  # type: ignore[method-assign]  # noqa: SLF001
    connection._resolve_pairing_activate = resolve  # type: ignore[method-assign]  # noqa: SLF001

    try:
        await connection._pair()  # noqa: SLF001
        assert attempts == 2
        assert not connection.is_pairing
    finally:
        await _cancel_time_task(connection)


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
    """A client may offer only one pairing-code method, and the per-session one wins."""
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
