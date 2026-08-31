"""ArtworkV1Role implementation (v1).

This role handles artwork binary streaming to display clients:
- Sends stream/start with channel configs on connect
- Sends binary artwork messages (types 8-11) when artwork changes
- Handles stream/request-format to change channel preferences
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiosendspin.models import BinaryMessageType, pack_binary_header_raw
from aiosendspin.models.artwork import (
    ArtworkChannel,
    StreamArtworkChannelConfig,
    StreamRequestFormatArtwork,
    StreamStartArtwork,
)
from aiosendspin.models.core import (
    StreamEndMessage,
    StreamEndPayload,
    StreamRequestFormatPayload,
    StreamStartMessage,
    StreamStartPayload,
)
from aiosendspin.models.types import ArtworkSource
from aiosendspin.server.roles.artwork.group import ArtworkGroupRole
from aiosendspin.server.roles.base import Role

if TYPE_CHECKING:
    from aiosendspin.server.client import SendspinClient


class ArtworkV1Role(Role):
    """Role implementation for artwork display.

    Manages artwork binary streaming. Unlike player, artwork streams are
    independent of playback - they start on connect and don't clear on pause/stop.
    """

    def __init__(self, client: SendspinClient | None = None) -> None:
        """Initialize ArtworkV1Role.

        Args:
            client: The owning SendspinClient.
        """
        if client is None:
            msg = "ArtworkV1Role requires a client"
            raise ValueError(msg)
        self._client = client
        self._stream_started = False
        self._buffer_tracker = None
        self._group_role: ArtworkGroupRole | None = None
        self._channel_configs: dict[int, ArtworkChannel] = {}

    @property
    def role_id(self) -> str:
        """Versioned role identifier."""
        return "artwork@v1"

    @property
    def role_family(self) -> str:
        """Role family name for protocol messages."""
        return "artwork"

    def requires_initial_state(self) -> bool:
        """Artwork receives server binary, gated on the client's initial state."""
        return True

    def on_connect(self) -> None:
        """Initialize channel configs from client hello and subscribe to group."""
        self._init_channel_configs()
        if self._channel_configs:
            # Reannounce stream config first so follow-up artwork snapshot is interpretable.
            self._send_stream_start()
        # Subscribe after stream/start so the on_member_join artwork snapshot lands second.
        self._subscribe_to_group_role()

    def on_deactivate(self) -> None:
        """End the artwork stream when the role is deactivated while still connected."""
        if self._stream_started:
            self.send_message(StreamEndMessage(payload=StreamEndPayload(roles=["artwork"])))
            self._stream_started = False
        super().on_deactivate()

    def on_disconnect(self) -> None:
        """Unsubscribe from ArtworkGroupRole."""
        self._unsubscribe_from_group_role()
        self._channel_configs.clear()
        self._stream_started = False

    def _init_channel_configs(self) -> None:
        """Initialize channel configs from client hello artwork support."""
        support = self._client.info.artwork_support
        if support is None:
            return

        for i, channel in enumerate(support.channels):
            self._channel_configs[i] = channel

    def _send_stream_start(self) -> None:
        """Send stream/start message with artwork channel configs."""
        # TODO: duplicate guard? on_connect already checks _channel_configs
        if not self._channel_configs:
            return

        stream_channels = []
        for channel_num in sorted(self._channel_configs.keys()):
            channel = self._channel_configs[channel_num]
            stream_channels.append(
                StreamArtworkChannelConfig(
                    source=channel.source,
                    format=channel.format,
                    width=channel.width,
                    height=channel.height,
                )
            )

        stream_start = StreamStartMessage(
            payload=StreamStartPayload(artwork=StreamStartArtwork(channels=stream_channels))
        )
        self.send_message(stream_start)
        self._stream_started = True

    def get_channel_configs(self) -> dict[int, ArtworkChannel]:
        """Return current channel configurations."""
        return self._channel_configs

    def send_artwork(self, channel: int, image_data: bytes, timestamp_us: int) -> None:
        """Send artwork binary message for a channel.

        Args:
            channel: Channel number (0-3).
            image_data: Encoded image bytes.
            timestamp_us: Timestamp in microseconds.
        """
        # TODO: should we raise instead of swallowing when no transport?
        if not self.has_connection():
            return

        message_type = BinaryMessageType.ARTWORK_CHANNEL_0.value + channel
        header = pack_binary_header_raw(message_type, timestamp_us)

        self._client.send_binary(
            header + image_data,
            role_family=self.role_family,
            timestamp_us=timestamp_us,
            message_type=message_type,
        )

    def send_artwork_cleared(self, channel: int, timestamp_us: int) -> None:
        """Send empty artwork binary message to clear a channel.

        Args:
            channel: Channel number (0-3).
            timestamp_us: Timestamp in microseconds.
        """
        if not self.has_connection():
            return

        message_type = BinaryMessageType.ARTWORK_CHANNEL_0.value + channel
        header = pack_binary_header_raw(message_type, timestamp_us)

        self._client.send_binary(
            header,
            role_family=self.role_family,
            timestamp_us=timestamp_us,
            message_type=message_type,
        )

    def on_stream_request_format(
        self,
        payload: StreamRequestFormatPayload,
    ) -> None:
        """Handle stream/request-format for artwork channels."""
        artwork_request = payload.artwork
        if artwork_request is None:
            return

        if artwork_request.channel not in self._channel_configs:
            self._client.flag_noncompliance(
                f"stream/request-format targeted unknown artwork channel {artwork_request.channel}"
            )
            return

        invalid_dims = [
            name
            for name, value in (
                ("width", artwork_request.width),
                ("height", artwork_request.height),
            )
            if value is not None and value <= 0
        ]
        if invalid_dims:
            # Skip the update: non-positive dims would otherwise raise out of
            # ArtworkChannel and tear the connection down.
            self._client.flag_noncompliance(
                "stream/request-format artwork dimensions must be positive: "
                + ", ".join(invalid_dims)
            )
            return

        self._update_channel_config(artwork_request)

    def _update_channel_config(self, request: StreamRequestFormatArtwork) -> None:
        """Update channel config from a request and send updated stream/start."""
        current = self._channel_configs[request.channel]

        updated = ArtworkChannel(
            source=request.source if request.source is not None else current.source,
            format=request.format if request.format is not None else current.format,
            width=request.width if request.width is not None else current.width,
            height=request.height if request.height is not None else current.height,
        )

        self._channel_configs[request.channel] = updated
        self._send_stream_start()

        # TODO: refactor to guard clause: if source == NONE or _group_role is None: return
        if updated.source != ArtworkSource.NONE and isinstance(self._group_role, ArtworkGroupRole):
            group_role = self._group_role
            if updated.source == ArtworkSource.ALBUM:
                artwork = group_role.get_album_artwork()
            else:
                artwork = group_role.get_artist_artwork()

            if artwork is not None:
                group_role._schedule_send_artwork(  # noqa: SLF001
                    self, artwork, request.channel, updated
                )
