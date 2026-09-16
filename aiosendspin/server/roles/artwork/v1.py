"""ArtworkV1Role implementation (v1).

This role handles artwork binary streaming to display clients:
- Sends stream/start with the channel configs the client declares in client/state
- Sends binary artwork messages (types 8-11) when artwork changes
- Re-announces the stream when the client changes its channel configs
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
    ClientStatePayload,
    StreamEndMessage,
    StreamEndPayload,
    StreamRequestFormatPayload,
    StreamStartMessage,
    StreamStartPayload,
)
from aiosendspin.models.types import ArtworkSource, PictureFormat
from aiosendspin.server.roles.artwork.group import ArtworkGroupRole
from aiosendspin.server.roles.base import Role

if TYPE_CHECKING:
    from aiosendspin.server.client import SendspinClient

MAX_ARTWORK_CHANNELS = 4


class ArtworkV1Role(Role):
    """Role implementation for artwork display.

    Manages artwork binary streaming. Unlike player, artwork streams are
    independent of playback - they start once the client declares its channels
    and don't clear on pause/stop.
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
        # Channels of the active stream, positional; an index past the end is not streamed.
        self._channels: list[ArtworkChannel] = []

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
        """Subscribe to the group; the stream starts once the client declares its channels."""
        # DEPRECATED(spec-pr-195): remove in aiosendspin <version>
        support = self._client.info.artwork_support
        if support is not None:
            # Reannounce stream config first so follow-up artwork snapshot is interpretable.
            self._apply_channels(list(support.channels))
        # Subscribe after stream/start so the on_member_join artwork snapshot lands second.
        self._subscribe_to_group_role()

    def on_deactivate(self) -> None:
        """End the artwork stream when the role is deactivated while still connected."""
        if self._stream_started:
            self.send_message(StreamEndMessage(payload=StreamEndPayload(roles=["artwork"])))
        self._reset_stream()
        super().on_deactivate()

    def on_disconnect(self) -> None:
        """Unsubscribe from ArtworkGroupRole."""
        self._unsubscribe_from_group_role()
        self._reset_stream()

    def initial_state_deviations(self, payload: ClientStatePayload) -> list[str]:
        """Report an initial client/state that does not declare the artwork channels."""
        # DEPRECATED(spec-pr-195): remove in aiosendspin <version>
        if payload.artwork is None and self._client.info.artwork_support is None:
            return ["has an active artwork role but no artwork state"]
        return []

    def client_state_deviations(self, payload: ClientStatePayload) -> list[str]:
        """Report artwork channels in a client/state phrased on a superseded wire."""
        if payload.artwork is None:
            return []
        channels = payload.artwork.channels
        reasons: list[str] = []
        legacy_keys = sorted(
            {key for channel in channels for key in channel.legacy_dimension_keys or ()}
        )
        if legacy_keys:
            reasons.append("artwork used pre-rename dimension keys: " + ", ".join(legacy_keys))
        if any(channel.format is PictureFormat.BMP for channel in channels):
            reasons.append("artwork declared the removed 'bmp' format")
        return reasons

    def on_client_state(self, payload: ClientStatePayload) -> None:
        """Start or update the artwork stream from the declared channels."""
        if payload.artwork is not None:
            self._apply_channels(payload.artwork.channels)

    def get_channel_configs(self) -> dict[int, ArtworkChannel]:
        """Return the configurations of the channels currently streamed, by channel number."""
        return {
            channel_num: channel
            for channel_num, channel in enumerate(self._channels)
            if channel.source is not ArtworkSource.NONE
        }

    def send_artwork(self, channel: int, image_data: bytes, timestamp_us: int) -> None:
        """Send artwork binary message for a channel.

        Does nothing when the channel is not currently streamed.

        Args:
            channel: Channel number (0-3).
            image_data: Encoded image bytes.
            timestamp_us: Timestamp in microseconds.
        """
        # TODO: should we raise instead of swallowing when no transport?
        if not self.has_connection() or channel not in self.get_channel_configs():
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

        Does nothing when the channel is not currently streamed.

        Args:
            channel: Channel number (0-3).
            timestamp_us: Timestamp in microseconds.
        """
        if not self.has_connection() or channel not in self.get_channel_configs():
            return

        message_type = BinaryMessageType.ARTWORK_CHANNEL_0.value + channel
        header = pack_binary_header_raw(message_type, timestamp_us)

        self._client.send_binary(
            header,
            role_family=self.role_family,
            timestamp_us=timestamp_us,
            message_type=message_type,
        )

    # DEPRECATED(spec-pr-195): remove in aiosendspin <version>
    def on_stream_request_format(
        self,
        payload: StreamRequestFormatPayload,
    ) -> None:
        """Apply a pre-#195 artwork channel request onto the current channels."""
        artwork_request = payload.artwork
        if artwork_request is None:
            return

        if artwork_request.channel >= len(self._channels):
            self._client.flag_noncompliance(
                f"stream/request-format targeted unknown artwork channel {artwork_request.channel}"
            )
            return

        self._flag_legacy_artwork_wire(artwork_request)

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

        channels = list(self._channels)
        current = channels[artwork_request.channel]
        channels[artwork_request.channel] = ArtworkChannel(
            source=artwork_request.source if artwork_request.source is not None else current.source,
            format=artwork_request.format if artwork_request.format is not None else current.format,
            width=artwork_request.width if artwork_request.width is not None else current.width,
            height=artwork_request.height if artwork_request.height is not None else current.height,
        )
        self._apply_channels(channels)

    # DEPRECATED(spec-pr-195): remove in aiosendspin <version>
    def _flag_legacy_artwork_wire(self, request: StreamRequestFormatArtwork) -> None:
        """Flag a request phrased on the wire the spec superseded."""
        if request.legacy_dimension_keys:
            self._client.flag_noncompliance(
                "stream/request-format artwork used pre-rename dimension keys: "
                + ", ".join(request.legacy_dimension_keys)
            )
        if request.format is PictureFormat.BMP:
            self._client.flag_noncompliance(
                "stream/request-format artwork requested the removed 'bmp' format"
            )

    def _apply_channels(self, channels: list[ArtworkChannel]) -> None:
        """Stream `channels`, re-announcing the stream when its configuration changed."""
        new_configs = _stream_configs(channels)
        if self._stream_started:
            old_configs = _stream_configs(self._channels)
            if old_configs == new_configs:
                self._channels = channels
                return
            # Queued images may be encoded for the old configuration; the current image of
            # every streamed channel is re-sent below.
            self._client.drop_pending_binary([self.role_family])
            # A channel must be cleared before the stream/start that stops streaming it.
            now_us = self._client._server.clock.now_us()  # noqa: SLF001
            for channel_num, config in enumerate(new_configs):
                if config.source is ArtworkSource.NONE and old_configs[channel_num] != config:
                    self.send_artwork_cleared(channel_num, now_us)

        self._channels = channels
        self._send_stream_start(new_configs)
        if self._group_role is not None:
            self._group_role.send_current_artwork(self)

    def _send_stream_start(self, configs: list[StreamArtworkChannelConfig]) -> None:
        """Send stream/start with `configs` truncated after the last streamed channel."""
        streamed = [
            i for i, config in enumerate(configs) if config.source is not ArtworkSource.NONE
        ]
        # With no channel streamed, keep one entry: the stream stays active so a later
        # client/state can enable a channel, which it could not do after a stream/end.
        stream_channels = configs[: streamed[-1] + 1] if streamed else configs[:1]
        stream_start = StreamStartMessage(
            payload=StreamStartPayload(artwork=StreamStartArtwork(channels=stream_channels))
        )
        self.send_message(stream_start)
        self._stream_started = True

    def _reset_stream(self) -> None:
        """Forget the stream so the next activation waits for the client's channels again."""
        self._stream_started = False
        self._channels = []


def _stream_configs(channels: list[ArtworkChannel]) -> list[StreamArtworkChannelConfig]:
    """Return the stream/start config of every channel number, uncovered ones as `none`."""
    configs = [
        StreamArtworkChannelConfig(source=ArtworkSource.NONE)
        if channel.source is ArtworkSource.NONE
        else StreamArtworkChannelConfig(
            source=channel.source,
            format=channel.format,
            width=channel.width,
            height=channel.height,
        )
        for channel in channels
    ]
    missing = MAX_ARTWORK_CHANNELS - len(configs)
    return configs + [StreamArtworkChannelConfig(source=ArtworkSource.NONE)] * missing
