"""Tests for player model payloads."""

from __future__ import annotations

import pytest

from aiosendspin.models.player import (
    PLAYER_AUDIO_HEADER_SIZE,
    SEND_AHEAD_MAX,
    ClientHelloPlayerSupport,
    PlayerAudioHeader,
    PlayerCommandPayload,
    PlayerStatePayload,
    compute_send_ahead,
    pack_player_audio_header,
    unpack_player_audio_header,
)
from aiosendspin.models.types import BinaryMessageType, PlayerCommand


def test_player_state_output_delay_serializes_when_set() -> None:
    """output_delay_ms is serialized when explicitly set."""
    payload = PlayerStatePayload(output_delay_ms=0)
    data = payload.to_dict()
    assert "output_delay_ms" in data
    assert data["output_delay_ms"] == 0


def test_player_state_output_delay_omitted_when_unset() -> None:
    """output_delay_ms is omitted when not provided so partial deltas don't reset it."""
    payload = PlayerStatePayload()
    data = payload.to_dict()
    assert "output_delay_ms" not in data


def test_player_state_output_delay_range_valid() -> None:
    """Maximum value 5000 is accepted."""
    payload = PlayerStatePayload(output_delay_ms=5000)
    assert payload.output_delay_ms == 5000


def test_player_state_output_delay_range_invalid() -> None:
    """Values above 5000 are rejected."""
    with pytest.raises(ValueError, match="output_delay_ms"):
        PlayerStatePayload(output_delay_ms=5001)


def test_player_state_output_delay_negative_invalid() -> None:
    """Negative values are rejected."""
    with pytest.raises(ValueError, match="output_delay_ms"):
        PlayerStatePayload(output_delay_ms=-1)


def test_player_state_supported_commands_serializes() -> None:
    """supported_commands serializes enum values as strings."""
    payload = PlayerStatePayload(supported_commands=[PlayerCommand.SET_OUTPUT_DELAY])
    data = payload.to_dict()
    assert data["supported_commands"] == ["set_output_delay"]


def test_player_state_backward_compat_no_delay() -> None:
    """Omitted output_delay_ms parses as None so server treats it as 'unchanged'."""
    data = '{"volume": 50}'
    payload = PlayerStatePayload.from_json(data)
    assert payload.output_delay_ms is None


def test_player_state_timing_defaults_to_none() -> None:
    """Omitted timing fields parse as None so server treats them as 'unchanged'."""
    payload = PlayerStatePayload.from_json('{"volume": 50}')
    assert payload.required_lead_time_ms is None
    assert payload.min_buffer_ms is None


def test_player_state_timing_serializes() -> None:
    """Timing fields are always serialized (not omitted)."""
    data = PlayerStatePayload(required_lead_time_ms=80, min_buffer_ms=1200).to_dict()
    assert data["required_lead_time_ms"] == 80
    assert data["min_buffer_ms"] == 1200


def test_player_state_required_lead_time_out_of_range() -> None:
    """required_lead_time_ms above 30000 is rejected."""
    with pytest.raises(ValueError, match="required_lead_time_ms"):
        PlayerStatePayload(required_lead_time_ms=30001)


def test_player_state_min_buffer_negative_invalid() -> None:
    """Negative min_buffer_ms is rejected."""
    with pytest.raises(ValueError, match="min_buffer_ms"):
        PlayerStatePayload(min_buffer_ms=-1)


def test_player_command_set_output_delay_valid() -> None:
    """SET_OUTPUT_DELAY command accepts valid delay value."""
    cmd = PlayerCommandPayload(command=PlayerCommand.SET_OUTPUT_DELAY, output_delay_ms=300)
    assert cmd.output_delay_ms == 300


def test_player_command_set_output_delay_missing() -> None:
    """SET_OUTPUT_DELAY command requires output_delay_ms."""
    with pytest.raises(ValueError, match="output_delay_ms must be provided"):
        PlayerCommandPayload(command=PlayerCommand.SET_OUTPUT_DELAY)


def test_player_command_set_output_delay_out_of_range() -> None:
    """SET_OUTPUT_DELAY command rejects out-of-range values."""
    with pytest.raises(ValueError, match="output_delay_ms"):
        PlayerCommandPayload(command=PlayerCommand.SET_OUTPUT_DELAY, output_delay_ms=6000)


def test_player_command_volume_rejects_output_delay() -> None:
    """VOLUME command rejects output_delay_ms parameter."""
    with pytest.raises(ValueError, match="output_delay_ms should not"):
        PlayerCommandPayload(command=PlayerCommand.VOLUME, volume=50, output_delay_ms=100)


def test_player_state_accepts_full_supported_command_set() -> None:
    """State-level supported_commands accepts volume, mute and set_output_delay."""
    payload = PlayerStatePayload.from_dict(
        {"supported_commands": ["volume", "mute", "set_output_delay"]}
    )
    assert payload.supported_commands == [
        PlayerCommand.VOLUME,
        PlayerCommand.MUTE,
        PlayerCommand.SET_OUTPUT_DELAY,
    ]


def test_player_state_rejects_unknown_supported_command() -> None:
    """An unknown state-level command fails to parse."""
    with pytest.raises(ValueError, match="supported_commands"):
        PlayerStatePayload.from_dict({"supported_commands": ["reboot"]})


def test_player_state_empty_supported_commands_round_trips() -> None:
    """An empty supported_commands list is serialized, not omitted."""
    data = PlayerStatePayload(supported_commands=[]).to_dict()
    assert data["supported_commands"] == []
    assert PlayerStatePayload.from_dict(data).supported_commands == []


def test_player_state_supported_commands_omitted_when_unset() -> None:
    """supported_commands is omitted when unset so incremental updates leave it unchanged."""
    assert "supported_commands" not in PlayerStatePayload().to_dict()


def _hello_support(**extra: object) -> dict[str, object]:
    return {
        "supported_formats": [
            {"codec": "pcm", "channels": 2, "sample_rate": 48000, "bit_depth": 16}
        ],
        "buffer_capacity": 100_000,
        **extra,
    }


def test_hello_player_support_parses_without_supported_commands() -> None:
    """A spec-shaped player support object without supported_commands parses."""
    support = ClientHelloPlayerSupport.from_dict(_hello_support())
    assert support.supported_commands is None
    assert "supported_commands" not in support.to_dict()


# DEPRECATED(spec-pr-177): remove in aiosendspin <version>
def test_hello_player_support_accepts_legacy_supported_commands() -> None:
    """A pre-#177 hello-level supported_commands list is still parsed."""
    support = ClientHelloPlayerSupport.from_dict(_hello_support(supported_commands=["volume"]))
    assert support.supported_commands == [PlayerCommand.VOLUME]


# DEPRECATED(spec-pr-177): remove in aiosendspin <version>
def test_hello_player_support_rejects_legacy_delay_command() -> None:
    """The legacy hello-level list only ever held volume and mute."""
    with pytest.raises(ValueError, match="Invalid hello supported_commands"):
        ClientHelloPlayerSupport.from_dict(_hello_support(supported_commands=["set_output_delay"]))


def test_player_state_accepts_pre_rename_delay_key() -> None:
    """static_delay_ms is rewritten to output_delay_ms and recorded for the role to flag."""
    payload = PlayerStatePayload.from_dict({"static_delay_ms": 250})
    assert payload.output_delay_ms == 250
    assert payload.legacy_delay_key == "static_delay_ms"


def test_player_state_current_key_wins_over_legacy() -> None:
    """When both keys are present, the current output_delay_ms value is kept."""
    payload = PlayerStatePayload.from_dict({"static_delay_ms": 250, "output_delay_ms": 400})
    assert payload.output_delay_ms == 400
    assert payload.legacy_delay_key == "static_delay_ms"


def test_player_state_current_delay_key_not_flagged_as_legacy() -> None:
    """output_delay_ms alone leaves legacy_delay_key unset."""
    payload = PlayerStatePayload.from_dict({"output_delay_ms": 250})
    assert payload.legacy_delay_key is None


def test_player_state_accepts_pre_rename_command_name() -> None:
    """set_static_delay is still a valid state-level supported_commands entry."""
    payload = PlayerStatePayload(supported_commands=[PlayerCommand.SET_STATIC_DELAY])
    assert payload.supported_commands == [PlayerCommand.SET_STATIC_DELAY]


def test_player_command_set_static_delay_serializes_pre_rename_wire_shape() -> None:
    """Constructing with SET_STATIC_DELAY addresses a client that only declared that name."""
    cmd = PlayerCommandPayload(command=PlayerCommand.SET_STATIC_DELAY, output_delay_ms=300)
    data = cmd.to_dict()
    assert data == {"command": "set_static_delay", "static_delay_ms": 300}


def test_player_command_set_static_delay_requires_output_delay_ms() -> None:
    """SET_STATIC_DELAY command requires output_delay_ms same as SET_OUTPUT_DELAY."""
    with pytest.raises(ValueError, match="output_delay_ms must be provided"):
        PlayerCommandPayload(command=PlayerCommand.SET_STATIC_DELAY)


def test_player_command_accepts_pre_rename_delay_key() -> None:
    """A pre-rename server's set_static_delay command parses into output_delay_ms."""
    cmd = PlayerCommandPayload.from_dict({"command": "set_static_delay", "static_delay_ms": 300})
    assert cmd.command == PlayerCommand.SET_STATIC_DELAY
    assert cmd.output_delay_ms == 300


def test_player_command_pre_rename_round_trips() -> None:
    """SET_STATIC_DELAY survives the pre-rename wire shape in both directions."""
    cmd = PlayerCommandPayload(command=PlayerCommand.SET_STATIC_DELAY, output_delay_ms=300)
    restored = PlayerCommandPayload.from_dict(cmd.to_dict())
    assert restored == cmd


def test_player_command_accepts_current_delay_key() -> None:
    """The current set_output_delay spelling still parses."""
    cmd = PlayerCommandPayload.from_dict({"command": "set_output_delay", "output_delay_ms": 300})
    assert cmd.command == PlayerCommand.SET_OUTPUT_DELAY
    assert cmd.output_delay_ms == 300


def test_player_command_current_delay_key_wins_over_legacy() -> None:
    """When both delay keys are present, the current output_delay_ms value is kept."""
    cmd = PlayerCommandPayload.from_dict(
        {"command": "set_static_delay", "static_delay_ms": 250, "output_delay_ms": 400}
    )
    assert cmd.output_delay_ms == 400


def test_player_state_legacy_delay_key_cannot_be_spoofed() -> None:
    """A client sending legacy_delay_key on the wire is not flagged for it."""
    payload = PlayerStatePayload.from_dict(
        {"output_delay_ms": 250, "legacy_delay_key": "static_delay_ms"}
    )
    assert payload.output_delay_ms == 250
    assert payload.legacy_delay_key is None


def test_player_state_from_dict_does_not_mutate_input() -> None:
    """Parsing a pre-rename payload leaves the caller's dict untouched."""
    raw = {"static_delay_ms": 250}
    PlayerStatePayload.from_dict(raw)
    assert raw == {"static_delay_ms": 250}


def test_player_audio_header_round_trips() -> None:
    """The 13-byte audio header packs big-endian and unpacks to the same fields."""
    header = pack_player_audio_header(0x0102030405060708, 0x0A0B0C0D)

    assert header == b"\x04\x01\x02\x03\x04\x05\x06\x07\x08\x0a\x0b\x0c\x0d"
    assert PLAYER_AUDIO_HEADER_SIZE == 13
    assert unpack_player_audio_header(header + b"audio") == PlayerAudioHeader(
        message_type=BinaryMessageType.AUDIO_CHUNK.value,
        timestamp_us=0x0102030405060708,
        send_ahead=0x0A0B0C0D,
    )


def test_unpack_player_audio_header_rejects_short_data() -> None:
    """Data shorter than the header raises ValueError."""
    with pytest.raises(ValueError, match="at least 13 bytes"):
        unpack_player_audio_header(bytes(12))


@pytest.mark.parametrize(
    ("timestamp_us", "now_us", "expected"),
    [
        (1_250_000, 1_000_000, 250_000),
        (1_000_000, 1_000_000, 0),
        (1_000_000, 1_000_001, 0),
        (-5_000_000, 1_000_000, 0),
        (1_000_000 + SEND_AHEAD_MAX, 1_000_000, SEND_AHEAD_MAX),
        (1_000_000 + SEND_AHEAD_MAX + 1, 1_000_000, SEND_AHEAD_MAX),
    ],
)
def test_compute_send_ahead_saturates(timestamp_us: int, now_us: int, expected: int) -> None:
    """send_ahead is the lead in hand, clamped to 0 and to the uint32 maximum."""
    assert compute_send_ahead(timestamp_us, now_us) == expected
