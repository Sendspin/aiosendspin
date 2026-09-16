"""Tests for :mod:`aiosendspin.noise.models`."""

from __future__ import annotations

import json

import pytest

from aiosendspin import noise
from aiosendspin.models.types import ServerErrorReason
from aiosendspin.noise.models import (
    ClientInitMessage,
    ClientInitPayload,
    ClientPairRetryMessage,
    NoiseHandshakeMessage,
    NoiseHandshakePayload,
    NoiseMsg1Payload,
    NoiseMsg2Payload,
    PairingMessage,
    ServerErrorMessage,
    ServerErrorPayload,
    ServerInitMessage,
    ServerInitPayload,
    ServerPairInitMessage,
    ServerPairInitPayload,
)
from aiosendspin.noise.trust_store import PskCategory


def test_client_init_message_round_trip() -> None:
    """ClientInitMessage serializes to spec-shaped JSON and roundtrips through from_json."""
    msg = ClientInitMessage(
        payload=ClientInitPayload(
            client_id="GFsV9tLaSQm9HcFWpKsgYQOr7wFTvNUtkmFwuVz3zoo",
            version=1,
            suite="25519_ChaChaPoly_SHA256",
        ),
    )
    raw = msg.to_json()
    assert '"type":"client/init"' in raw
    assert '"suite":"25519_ChaChaPoly_SHA256"' in raw
    parsed = ClientInitMessage.from_json(raw)
    assert parsed == msg


def test_server_init_has_no_suite_field() -> None:
    """ServerInitPayload only carries server_id and version (spec: no suite)."""
    msg = ServerInitMessage(
        payload=ServerInitPayload(
            server_id="GFsV9tLaSQm9HcFWpKsgYQOr7wFTvNUtkmFwuVz3zoo",
            version=1,
        ),
    )
    raw = msg.to_json()
    assert "suite" not in raw
    assert '"type":"server/init"' in raw


@pytest.mark.parametrize("reason", list(ServerErrorReason))
def test_server_error_message_round_trip(reason: ServerErrorReason) -> None:
    """ServerErrorMessage carries only the reason and roundtrips through from_json."""
    msg = ServerErrorMessage(payload=ServerErrorPayload(reason=reason))
    raw = msg.to_json()
    assert json.loads(raw) == {"type": "server/error", "payload": {"reason": reason.value}}
    assert ServerErrorMessage.from_json(raw) == msg


def test_server_error_reasons_match_the_spec() -> None:
    """ServerErrorReason holds exactly the spec's reasons."""
    assert {reason.value for reason in ServerErrorReason} == {
        "unsupported_version",
        "unsupported_suite",
        "malformed",
    }


def test_server_error_names_are_public() -> None:
    """The server/error models, reason and rejection are importable from the noise package."""
    for name in ("ServerErrorMessage", "ServerErrorPayload", "ServerErrorReason"):
        assert name in noise.__all__
    assert noise.ServerErrorMessage is ServerErrorMessage
    assert noise.ServerErrorPayload is ServerErrorPayload
    assert noise.ServerErrorReason is ServerErrorReason
    assert "InitRejectedError" in noise.__all__


def test_noise_integer_fields_serialize_as_integers() -> None:
    """Noise messages emit integer-typed wire fields."""
    msg = ClientInitMessage(
        payload=ClientInitPayload(
            client_id="GFsV9tLaSQm9HcFWpKsgYQOr7wFTvNUtkmFwuVz3zoo",
            version=1.0,
            suite="25519_ChaChaPoly_SHA256",
        ),
    )
    version = json.loads(msg.to_json())["payload"]["version"]
    assert version == 1
    assert type(version) is int


def test_noise_handshake_message_round_trip() -> None:
    """NoiseHandshakeMessage carries a base64url ``data`` field."""
    msg = NoiseHandshakeMessage(payload=NoiseHandshakePayload(data="aGVsbG8"))
    parsed = NoiseHandshakeMessage.from_json(msg.to_json())
    assert parsed.payload.data == "aGVsbG8"
    assert parsed.type == "noise/handshake"


def test_noise_msg1_payload_carries_psk_id_and_category() -> None:
    """The encrypted msg-1 inner payload exposes ``psk_id`` and the PSK's category code."""
    payload = NoiseMsg1Payload(
        psk_id="GFsV9tLaSQm9HcFWpKsgYQOr7wFTvNUtkmFwuVz3zoo",
        psk_category=PskCategory.SENTINEL.code,
    )
    raw = payload.to_json()
    assert raw == ('{"psk_id":"GFsV9tLaSQm9HcFWpKsgYQOr7wFTvNUtkmFwuVz3zoo","psk_category":"sn"}')


def test_noise_msg1_payload_requires_a_category() -> None:
    """The category is not optional: a payload without it does not parse."""
    with pytest.raises(Exception, match="psk_category"):
        NoiseMsg1Payload.from_json('{"psk_id":"GFsV9tLaSQm9HcFWpKsgYQOr7wFTvNUtkmFwuVz3zoo"}')


def test_noise_msg1_category_codes_share_one_length() -> None:
    """Equal-length codes keep the encrypted payload's length independent of the category."""
    assert len({len(category.code) for category in PskCategory}) == 1


def test_noise_msg2_payload_serializes_to_empty_object() -> None:
    """The encrypted msg-2 inner payload is the literal empty object ``{}`` per spec."""
    assert NoiseMsg2Payload().to_json() == "{}"
    # And empty input roundtrips cleanly.
    assert NoiseMsg2Payload.from_json("{}") == NoiseMsg2Payload()


@pytest.mark.parametrize(
    ("payload", "wire"),
    [
        pytest.param(
            ServerPairInitPayload(nonce_A="A" * 43), {"nonce_A": "A" * 43}, id="first-round"
        ),
        pytest.param(ServerPairInitPayload(), {}, id="later-round"),
    ],
)
def test_server_pair_init_omits_absent_nonce(
    payload: ServerPairInitPayload, wire: dict[str, str]
) -> None:
    """server/pair-init carries nonce_A only when set and round-trips either way."""
    msg = ServerPairInitMessage(payload=payload)
    raw = msg.to_json()
    assert json.loads(raw) == {"type": "server/pair-init", "payload": wire}
    assert PairingMessage.from_json(raw) == msg


def test_client_pair_retry_round_trip() -> None:
    """client/pair-retry is an empty-payload pairing message."""
    raw = ClientPairRetryMessage().to_json()
    assert json.loads(raw) == {"type": "client/pair-retry", "payload": {}}
    assert PairingMessage.from_json(raw) == ClientPairRetryMessage()
