"""Tests for ArtworkV1Role (v1) implementation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aiosendspin.models.artwork import ArtworkChannel, StreamRequestFormatArtwork
from aiosendspin.models.core import (
    StreamEndMessage,
    StreamRequestFormatPayload,
    StreamStartMessage,
)
from aiosendspin.models.types import ArtworkSource, PictureFormat
from aiosendspin.server.roles.artwork.group import ArtworkGroupRole
from aiosendspin.server.roles.artwork.v1 import ArtworkV1Role


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


def _make_client_stub_with_channel() -> MagicMock:
    """Mock client whose hello advertises a single artwork channel."""
    client = _make_client_stub()
    client.info.artwork_support = MagicMock()
    client.info.artwork_support.channels = [
        ArtworkChannel(
            source=ArtworkSource.ALBUM,
            format=PictureFormat.JPEG,
            width=300,
            height=300,
        )
    ]
    return client


def test_artwork_role_has_role_id() -> None:
    """ArtworkV1Role has role_id of 'artwork@v1'."""
    client = _make_client_stub()
    role = ArtworkV1Role(client=client)
    assert role.role_id == "artwork@v1"


def test_artwork_role_flags_nonpositive_request_dimensions() -> None:
    """A stream/request-format with non-positive dimensions is flagged, not applied."""
    client = _make_client_stub_with_channel()
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_stream_request_format(
        StreamRequestFormatPayload(artwork=StreamRequestFormatArtwork(channel=0, width=-10))
    )
    client.flag_noncompliance.assert_called_once()


def test_artwork_role_no_flag_for_valid_request_dimensions() -> None:
    """A stream/request-format with positive dimensions is not flagged."""
    client = _make_client_stub_with_channel()
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_stream_request_format(
        StreamRequestFormatPayload(
            artwork=StreamRequestFormatArtwork(channel=0, width=100, height=100)
        )
    )
    client.flag_noncompliance.assert_not_called()


def test_artwork_role_flags_unknown_request_channel() -> None:
    """A stream/request-format for a channel with no config is flagged."""
    client = _make_client_stub_with_channel()
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_stream_request_format(
        StreamRequestFormatPayload(artwork=StreamRequestFormatArtwork(channel=1))
    )
    client.flag_noncompliance.assert_called_once()


def test_artwork_role_has_role_family() -> None:
    """ArtworkV1Role has role_family of 'artwork'."""
    client = _make_client_stub()
    role = ArtworkV1Role(client=client)
    assert role.role_family == "artwork"


def test_artwork_role_requires_client() -> None:
    """ArtworkV1Role raises ValueError if no client provided."""
    with pytest.raises(ValueError, match="requires a client"):
        ArtworkV1Role(client=None)


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


def test_artwork_role_on_deactivate_sends_stream_end_when_started() -> None:
    """on_deactivate() ends the artwork stream once a stream/start was sent."""
    client = _make_client_stub_with_channel()
    role = ArtworkV1Role(client=client)
    role.on_connect()
    client.send_role_message.reset_mock()

    role.on_deactivate()

    sent = [call.args[1] for call in client.send_role_message.call_args_list]
    assert any(isinstance(m, StreamEndMessage) and m.payload.roles == ["artwork"] for m in sent)


def test_artwork_role_on_deactivate_clears_stream_started() -> None:
    """on_deactivate() resets _stream_started so a reused instance holds no stale flag."""
    client = _make_client_stub_with_channel()
    role = ArtworkV1Role(client=client)
    role.on_connect()

    role.on_deactivate()

    assert role._stream_started is False  # noqa: SLF001


def test_artwork_role_on_deactivate_noop_without_stream() -> None:
    """on_deactivate() sends nothing when no stream/start was ever sent."""
    client = _make_client_stub()  # no artwork_support -> no stream/start
    role = ArtworkV1Role(client=client)
    role.on_connect()
    client.send_role_message.reset_mock()

    role.on_deactivate()

    client.send_role_message.assert_not_called()


def test_artwork_role_init_channel_configs_from_support() -> None:
    """on_connect() initializes channel configs from client hello."""
    client = _make_client_stub()
    support = MagicMock()
    support.channels = [
        ArtworkChannel(
            source=ArtworkSource.ALBUM,
            format=PictureFormat.JPEG,
            width=300,
            height=300,
        ),
        ArtworkChannel(
            source=ArtworkSource.ARTIST,
            format=PictureFormat.PNG,
            width=400,
            height=400,
        ),
    ]
    client.info.artwork_support = support

    role = ArtworkV1Role(client=client)
    role.on_connect()

    configs = role.get_channel_configs()
    assert len(configs) == 2
    assert configs[0].source == ArtworkSource.ALBUM
    assert configs[1].source == ArtworkSource.ARTIST


def test_artwork_role_sends_stream_start_on_connect_with_transport() -> None:
    """on_connect() sends stream/start when transport is attached."""
    client = _make_client_stub()
    support = MagicMock()
    support.channels = [
        ArtworkChannel(
            source=ArtworkSource.ALBUM,
            format=PictureFormat.JPEG,
            width=300,
            height=300,
        ),
    ]
    client.info.artwork_support = support

    role = ArtworkV1Role(client=client)
    role._client.connection = MagicMock()  # noqa: SLF001
    role.on_connect()

    client.send_role_message.assert_called()
    _role, msg = client.send_role_message.call_args.args
    assert isinstance(msg, StreamStartMessage)
    assert msg.payload.artwork is not None
    assert len(msg.payload.artwork.channels) == 1


def test_artwork_role_send_artwork() -> None:
    """send_artwork() sends binary message with header and image data."""
    client = _make_client_stub()
    role = ArtworkV1Role(client=client)
    role._client.connection = MagicMock()  # noqa: SLF001

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

    role.send_artwork_cleared(channel=1, timestamp_us=2000)

    client.send_binary.assert_called_once()
    kwargs = client.send_binary.call_args.kwargs
    assert kwargs["message_type"] == 9  # ARTWORK_CHANNEL_1


def test_artwork_role_send_artwork_noop_without_transport() -> None:
    """send_artwork() is a no-op when no transport."""
    client = _make_client_stub()
    role = ArtworkV1Role(client=client)
    role._client.connection = None  # noqa: SLF001

    role.send_artwork(channel=0, image_data=b"image", timestamp_us=1000)

    client.send_binary.assert_not_called()


def test_artwork_role_has_no_audio_requirements() -> None:
    """ArtworkV1Role does not receive audio."""
    client = _make_client_stub()
    role = ArtworkV1Role(client=client)
    assert role.get_audio_requirements() is None


def test_artwork_role_on_connect_schedules_artwork_once_per_channel() -> None:
    """on_connect() schedules a single artwork snapshot per configured channel."""
    from PIL import Image  # noqa: PLC0415

    client = _make_client_stub()
    support = MagicMock()
    support.channels = [
        ArtworkChannel(
            source=ArtworkSource.ALBUM,
            format=PictureFormat.JPEG,
            width=300,
            height=300,
        ),
        ArtworkChannel(
            source=ArtworkSource.ARTIST,
            format=PictureFormat.PNG,
            width=400,
            height=400,
        ),
    ]
    client.info.artwork_support = support
    client.connection = MagicMock()

    group = MagicMock()
    group._server = MagicMock()  # noqa: SLF001
    group._server.clock.now_us.return_value = 1_000_000  # noqa: SLF001
    group_role = ArtworkGroupRole(group)
    group_role._current_artwork = {  # noqa: SLF001
        ArtworkSource.ALBUM: Image.new("RGB", (10, 10)),
        ArtworkSource.ARTIST: Image.new("RGB", (10, 10)),
    }
    client.group.group_role.return_value = group_role

    role = ArtworkV1Role(client=client)
    with patch.object(group_role, "_schedule_send_artwork") as schedule:
        role.on_connect()

    channels_sent = [call.args[2] for call in schedule.call_args_list]
    assert sorted(channels_sent) == [0, 1]


def test_artwork_partial_format_request_preserves_unchanged_fields() -> None:
    """A partial stream/request-format only overwrites fields the client included."""
    client = _make_client_stub()
    support = MagicMock()
    support.channels = [
        ArtworkChannel(
            source=ArtworkSource.ALBUM,
            format=PictureFormat.JPEG,
            width=300,
            height=300,
        ),
    ]
    client.info.artwork_support = support

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
