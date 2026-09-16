"""Tests for ArtworkV1Role (v1) implementation."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image

from aiosendspin.models.artwork import (
    ArtworkChannel,
    ClientHelloArtworkSupport,
    ClientStateArtwork,
    StreamRequestFormatArtwork,
)
from aiosendspin.models.core import (
    ClientStatePayload,
    StreamEndMessage,
    StreamRequestFormatPayload,
    StreamStartMessage,
)
from aiosendspin.models.types import ArtworkSource, PictureFormat
from aiosendspin.server.roles.artwork.group import ArtworkGroupRole
from aiosendspin.server.roles.artwork.v1 import ArtworkV1Role

_ALBUM = ArtworkChannel(
    source=ArtworkSource.ALBUM, format=PictureFormat.JPEG, width=300, height=300
)
_ARTIST = ArtworkChannel(
    source=ArtworkSource.ARTIST, format=PictureFormat.PNG, width=400, height=200
)
_NONE = ArtworkChannel(source=ArtworkSource.NONE)

_ALBUM_WIRE = {"source": "album", "format": "jpeg", "width": 300, "height": 300}
_ARTIST_WIRE = {"source": "artist", "format": "png", "width": 400, "height": 200}
_NONE_WIRE = {"source": "none"}


def _make_client_stub() -> MagicMock:
    """Create a mock client for testing."""
    client = MagicMock()
    client.group = MagicMock()
    client.group.group_role.return_value = None
    client.info = MagicMock()
    client.info.artwork_support = None
    client.send_message = MagicMock()
    client.send_role_message = MagicMock()
    client.send_binary = MagicMock(return_value=True)
    client._logger = MagicMock()  # noqa: SLF001
    return client


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def _make_legacy_client_stub(*channels: ArtworkChannel) -> MagicMock:
    """Mock client whose hello declares artwork channels."""
    client = _make_client_stub()
    client.info.artwork_support = ClientHelloArtworkSupport(channels=list(channels or [_ALBUM]))
    return client


def _state(*channels: ArtworkChannel) -> ClientStatePayload:
    return ClientStatePayload(available=True, artwork=ClientStateArtwork(channels=list(channels)))


def _record(client: MagicMock, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Attach a group role with current images and record what the role sends, in order."""
    group = MagicMock()
    group._server.clock.now_us.return_value = 1_000_000  # noqa: SLF001
    group_role = ArtworkGroupRole(group)
    group_role._current_artwork = {  # noqa: SLF001
        ArtworkSource.ALBUM: Image.new("RGB", (10, 10)),
        ArtworkSource.ARTIST: Image.new("RGB", (10, 10)),
    }
    client.group.group_role.return_value = group_role

    events: list[Any] = []

    def _message(_role: str, message: object) -> None:
        if isinstance(message, StreamStartMessage):
            assert message.payload.artwork is not None
            events.append(("start", message.payload.artwork.to_dict()["channels"]))
        else:
            events.append(type(message).__name__)

    def _binary(data: bytes, **kwargs: Any) -> None:
        events.append(("binary", kwargs["message_type"] - 8, len(data)))

    def _schedule(_role: object, _image: object, channel: int, _config: object) -> None:
        events.append(("image", channel))

    client.send_role_message.side_effect = _message
    client.send_binary.side_effect = _binary
    monkeypatch.setattr(group_role, "_schedule_send_artwork", _schedule)
    return events


def test_artwork_role_has_role_id() -> None:
    """ArtworkV1Role has role_id of 'artwork@v1'."""
    client = _make_client_stub()
    role = ArtworkV1Role(client=client)
    assert role.role_id == "artwork@v1"


def test_artwork_role_has_role_family() -> None:
    """ArtworkV1Role has role_family of 'artwork'."""
    client = _make_client_stub()
    role = ArtworkV1Role(client=client)
    assert role.role_family == "artwork"


def test_artwork_role_requires_client() -> None:
    """ArtworkV1Role raises ValueError if no client provided."""
    with pytest.raises(ValueError, match="requires a client"):
        ArtworkV1Role(client=None)


def test_artwork_role_has_no_audio_requirements() -> None:
    """ArtworkV1Role does not receive audio."""
    client = _make_client_stub()
    role = ArtworkV1Role(client=client)
    assert role.get_audio_requirements() is None


def test_artwork_role_on_connect_subscribes_to_group_role() -> None:
    """on_connect() subscribes to ArtworkGroupRole."""
    client = _make_client_stub()
    group_role = MagicMock()
    client.group.group_role.return_value = group_role

    role = ArtworkV1Role(client=client)
    role.on_connect()

    client.group.group_role.assert_called_with("artwork")
    group_role.subscribe.assert_called_once_with(role)


def test_artwork_role_on_disconnect_unsubscribes_from_group_role() -> None:
    """on_disconnect() unsubscribes from ArtworkGroupRole."""
    client = _make_client_stub()
    group_role = MagicMock()
    client.group.group_role.return_value = group_role

    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_disconnect()

    group_role.unsubscribe.assert_called_once_with(role)


def test_artwork_role_on_connect_waits_for_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the client/state artwork object, nothing is streamed."""
    client = _make_client_stub()
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)

    role.on_connect()
    role.on_client_state(ClientStatePayload(available=True))

    assert events == []
    assert role.get_channel_configs() == {}


@pytest.mark.asyncio
async def test_artwork_update_before_state_sends_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A group artwork update reaches the client only once its channels are declared."""
    client = _make_client_stub()
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)
    role.on_connect()
    group_role = client.group.group_role.return_value

    await group_role.set_album_artwork(Image.new("RGB", (10, 10)))
    assert events == []

    role.on_client_state(_state(_ALBUM))
    events.clear()
    await group_role.set_album_artwork(Image.new("RGB", (10, 10)))
    assert [event[:2] for event in events] == [("binary", 0)]


@pytest.mark.parametrize(
    ("channels", "expected"),
    [
        ([_ALBUM], [_ALBUM_WIRE]),
        ([_ALBUM, _NONE, _ARTIST, _NONE], [_ALBUM_WIRE, _NONE_WIRE, _ARTIST_WIRE]),
        ([_NONE, _ARTIST], [_NONE_WIRE, _ARTIST_WIRE]),
        ([_NONE], [_NONE_WIRE]),
        ([_NONE, _NONE, _NONE, _NONE], [_NONE_WIRE]),
    ],
)
def test_artwork_state_starts_truncated_stream(
    monkeypatch: pytest.MonkeyPatch,
    channels: list[ArtworkChannel],
    expected: list[dict[str, object]],
) -> None:
    """stream/start matches the state, truncated after the last streamed channel."""
    client = _make_client_stub()
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)
    role.on_connect()

    role.on_client_state(_state(*channels))

    assert events[0] == ("start", expected)


def test_artwork_state_start_is_followed_by_current_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After stream/start, the current image is sent for each streamed channel."""
    client = _make_client_stub()
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)
    role.on_connect()

    role.on_client_state(_state(_ALBUM, _NONE, _ARTIST))

    assert events == [
        ("start", [_ALBUM_WIRE, _NONE_WIRE, _ARTIST_WIRE]),
        ("image", 0),
        ("image", 2),
    ]
    assert role.get_channel_configs() == {0: _ALBUM, 2: _ARTIST}


def test_artwork_unchanged_state_sends_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A state repeating the current channels produces no messages."""
    client = _make_client_stub()
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_client_state(_state(_ALBUM, _NONE))
    events.clear()

    stray_none = ArtworkChannel(
        source=ArtworkSource.NONE, format=PictureFormat.PNG, width=1, height=1
    )
    role.on_client_state(_state(_ALBUM, stray_none))
    role.on_client_state(_state(_ALBUM))

    assert events == []


def test_artwork_state_change_clears_restarts_and_resends_changed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed state clears disabled channels first and re-sends only changed channels."""
    client = _make_client_stub()
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_client_state(_state(_ALBUM, _NONE, _ARTIST))
    events.clear()

    role.on_client_state(_state(_NONE, _ARTIST, _ARTIST))

    assert events == [
        ("binary", 0, 9),
        ("start", [_NONE_WIRE, _ARTIST_WIRE, _ARTIST_WIRE]),
        ("image", 1),
    ]


def test_artwork_state_format_change_resends_that_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing one channel's format re-announces the stream and re-sends that channel only."""
    client = _make_client_stub()
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_client_state(_state(_ALBUM, _ARTIST))
    events.clear()

    png_album = ArtworkChannel(
        source=ArtworkSource.ALBUM, format=PictureFormat.PNG, width=300, height=300
    )
    role.on_client_state(_state(png_album, _ARTIST))

    assert events == [
        ("start", [{**_ALBUM_WIRE, "format": "png"}, _ARTIST_WIRE]),
        ("image", 0),
    ]


def test_artwork_all_none_state_keeps_stream_active(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disabling every channel keeps the stream, so a later state can enable one again."""
    client = _make_client_stub()
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_client_state(_state(_ALBUM))
    events.clear()

    role.on_client_state(_state(_NONE))
    role.on_client_state(_state(_ALBUM))

    assert events == [
        ("binary", 0, 9),
        ("start", [_NONE_WIRE]),
        ("start", [_ALBUM_WIRE]),
        ("image", 0),
    ]


def test_artwork_role_on_deactivate_sends_stream_end_when_started() -> None:
    """on_deactivate() ends the artwork stream and forgets its channels."""
    client = _make_client_stub()
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_client_state(_state(_ALBUM))
    client.send_role_message.reset_mock()

    role.on_deactivate()

    sent = [call.args[1] for call in client.send_role_message.call_args_list]
    assert any(isinstance(m, StreamEndMessage) and m.payload.roles == ["artwork"] for m in sent)
    assert role._stream_started is False  # noqa: SLF001
    assert role.get_channel_configs() == {}


def test_artwork_role_on_deactivate_noop_without_stream() -> None:
    """on_deactivate() sends nothing when no stream/start was ever sent."""
    client = _make_client_stub()
    role = ArtworkV1Role(client=client)
    role.on_connect()
    client.send_role_message.reset_mock()

    role.on_deactivate()

    client.send_role_message.assert_not_called()


def test_artwork_reconnect_waits_for_state_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """After a reconnect the stream restarts only once that connection's state arrives."""
    client = _make_client_stub()
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_client_state(_state(_ALBUM))
    role.on_disconnect()
    events.clear()

    role.on_connect()
    assert events == []
    assert role.get_channel_configs() == {}

    role.on_client_state(_state(_ALBUM))
    assert events == [("start", [_ALBUM_WIRE]), ("image", 0)]


def test_artwork_role_send_artwork() -> None:
    """send_artwork() sends binary message with header and image data."""
    client = _make_client_stub()
    role = ArtworkV1Role(client=client)
    role._client.connection = MagicMock()  # noqa: SLF001
    role.on_client_state(_state(_ALBUM))

    role.send_artwork(channel=0, image_data=b"image", timestamp_us=1000)

    client.send_binary.assert_called_once()
    kwargs = client.send_binary.call_args.kwargs
    assert kwargs["role_family"] == "artwork"
    assert kwargs["timestamp_us"] == 1000
    assert kwargs["message_type"] == 8  # ARTWORK_CHANNEL_0


def test_artwork_role_send_artwork_cleared() -> None:
    """send_artwork_cleared() sends empty binary message."""
    client = _make_client_stub()
    role = ArtworkV1Role(client=client)
    role._client.connection = MagicMock()  # noqa: SLF001
    role.on_client_state(_state(_NONE, _ARTIST))

    role.send_artwork_cleared(channel=1, timestamp_us=2000)

    client.send_binary.assert_called_once()
    kwargs = client.send_binary.call_args.kwargs
    assert kwargs["message_type"] == 9  # ARTWORK_CHANNEL_1


def test_artwork_role_send_artwork_skips_unstreamed_channels() -> None:
    """No artwork binary goes out before stream/start or for a channel that is not streamed."""
    client = _make_client_stub()
    role = ArtworkV1Role(client=client)
    role._client.connection = MagicMock()  # noqa: SLF001

    role.send_artwork(channel=0, image_data=b"image", timestamp_us=1000)
    role.on_client_state(_state(_ALBUM, _NONE))
    role.send_artwork(channel=1, image_data=b"image", timestamp_us=1000)
    role.send_artwork_cleared(channel=2, timestamp_us=1000)

    client.send_binary.assert_not_called()


def test_artwork_role_send_artwork_noop_without_transport() -> None:
    """send_artwork() is a no-op when no transport."""
    client = _make_client_stub()
    role = ArtworkV1Role(client=client)
    role.on_client_state(_state(_ALBUM))
    role._client.connection = None  # noqa: SLF001

    role.send_artwork(channel=0, image_data=b"image", timestamp_us=1000)

    client.send_binary.assert_not_called()


def test_artwork_initial_state_without_artwork_is_a_deviation() -> None:
    """An initial client/state must carry the artwork object."""
    client = _make_client_stub()
    role = ArtworkV1Role(client=client)

    assert role.initial_state_deviations(ClientStatePayload(available=True)) == [
        "has an active artwork role but no artwork state"
    ]
    assert role.initial_state_deviations(_state(_ALBUM)) == []


def test_artwork_state_on_superseded_wire_is_a_deviation() -> None:
    """A state artwork object using 'bmp' or media_width/media_height is reported."""
    client = _make_client_stub()
    role = ArtworkV1Role(client=client)
    payload = ClientStatePayload.from_dict(
        {
            "available": True,
            "artwork": {
                "channels": [
                    {"source": "album", "format": "bmp", "media_width": 8, "media_height": 8}
                ]
            },
        }
    )

    assert role.client_state_deviations(payload) == [
        "artwork used pre-rename dimension keys: media_height, media_width",
        "artwork declared the removed 'bmp' format",
    ]
    assert role.client_state_deviations(_state(_ALBUM)) == []


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_legacy_initial_state_without_artwork_is_not_a_deviation() -> None:
    """A client configured by its hello needs no artwork object in the initial state."""
    client = _make_legacy_client_stub()
    role = ArtworkV1Role(client=client)

    assert role.initial_state_deviations(ClientStatePayload(available=True)) == []


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_legacy_hello_channels_start_stream_on_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hello declaring artwork channels starts the stream on connect, then sends images."""
    client = _make_legacy_client_stub(_ALBUM, _ARTIST)
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)

    role.on_connect()

    assert events == [("start", [_ALBUM_WIRE, _ARTIST_WIRE]), ("image", 0), ("image", 1)]
    assert role.get_channel_configs() == {0: _ALBUM, 1: _ARTIST}


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_legacy_state_replaces_hello_channels(monkeypatch: pytest.MonkeyPatch) -> None:
    """A client/state artwork object replaces the channels the hello declared."""
    client = _make_legacy_client_stub(_ALBUM)
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)
    role.on_connect()
    events.clear()

    role.on_client_state(_state(_NONE, _ARTIST))

    assert events == [
        ("binary", 0, 9),
        ("start", [_NONE_WIRE, _ARTIST_WIRE]),
        ("image", 1),
    ]
    assert role.get_channel_configs() == {1: _ARTIST}


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_legacy_request_format_resends_changed_channel_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stream/request-format re-announces the stream and re-sends only its channel."""
    client = _make_legacy_client_stub(_ALBUM, _ARTIST)
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)
    role.on_connect()
    events.clear()

    role.on_stream_request_format(
        StreamRequestFormatPayload(artwork=StreamRequestFormatArtwork(channel=1, width=800))
    )
    role.on_stream_request_format(
        StreamRequestFormatPayload(artwork=StreamRequestFormatArtwork(channel=1, width=800))
    )

    assert events == [
        ("start", [_ALBUM_WIRE, {**_ARTIST_WIRE, "width": 800}]),
        ("image", 1),
    ]


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_legacy_request_format_enables_declared_none_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request can enable a hello channel declared as none, keeping its declared size."""
    declared_none = ArtworkChannel(
        source=ArtworkSource.NONE, format=PictureFormat.PNG, width=64, height=64
    )
    client = _make_legacy_client_stub(_ALBUM, declared_none)
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)
    role.on_connect()
    assert events[0] == ("start", [_ALBUM_WIRE])
    events.clear()

    role.on_stream_request_format(
        StreamRequestFormatPayload(
            artwork=StreamRequestFormatArtwork(channel=1, source=ArtworkSource.ARTIST)
        )
    )

    assert events == [
        ("start", [_ALBUM_WIRE, {"source": "artist", "format": "png", "width": 64, "height": 64}]),
        ("image", 1),
    ]


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_artwork_role_flags_nonpositive_request_dimensions() -> None:
    """A stream/request-format with non-positive dimensions is flagged, not applied."""
    client = _make_legacy_client_stub()
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_stream_request_format(
        StreamRequestFormatPayload(artwork=StreamRequestFormatArtwork(channel=0, width=-10))
    )
    client.flag_noncompliance.assert_called_once()


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_artwork_role_applies_and_flags_pre_rename_request_dimensions() -> None:
    """A stream/request-format phrased as media_width/media_height resizes, and is flagged."""
    client = _make_legacy_client_stub()
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_stream_request_format(
        StreamRequestFormatPayload(
            artwork=StreamRequestFormatArtwork.from_dict(
                {"channel": 0, "media_width": 800, "media_height": 480}
            )
        )
    )
    config = role.get_channel_configs()[0]
    assert (config.width, config.height) == (800, 480)
    assert "media_width" in client.flag_noncompliance.call_args.args[0]


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_artwork_role_flags_bmp_request_format() -> None:
    """A stream/request-format asking for the removed 'bmp' format is flagged, not rejected."""
    client = _make_legacy_client_stub()
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_stream_request_format(
        StreamRequestFormatPayload(
            artwork=StreamRequestFormatArtwork(channel=0, format=PictureFormat.BMP)
        )
    )
    assert "bmp" in client.flag_noncompliance.call_args.args[0]
    assert role.get_channel_configs()[0].format == PictureFormat.BMP


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_artwork_role_no_flag_for_valid_request_dimensions() -> None:
    """A stream/request-format with positive dimensions is not flagged."""
    client = _make_legacy_client_stub()
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_stream_request_format(
        StreamRequestFormatPayload(
            artwork=StreamRequestFormatArtwork(channel=0, width=100, height=100)
        )
    )
    client.flag_noncompliance.assert_not_called()


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_artwork_role_flags_unknown_request_channel() -> None:
    """A stream/request-format for a channel with no config is flagged."""
    client = _make_legacy_client_stub()
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_stream_request_format(
        StreamRequestFormatPayload(artwork=StreamRequestFormatArtwork(channel=1))
    )
    client.flag_noncompliance.assert_called_once()


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_artwork_partial_format_request_preserves_unchanged_fields() -> None:
    """A partial stream/request-format only overwrites fields the client included."""
    client = _make_legacy_client_stub()
    role = ArtworkV1Role(client=client)
    role.on_connect()

    payload = StreamRequestFormatPayload(
        artwork=StreamRequestFormatArtwork(channel=0, format=PictureFormat.PNG),
    )
    role.on_stream_request_format(payload)

    configs = role.get_channel_configs()
    assert configs[0].format == PictureFormat.PNG
    assert configs[0].source == ArtworkSource.ALBUM
    assert configs[0].width == 300
    assert configs[0].height == 300
