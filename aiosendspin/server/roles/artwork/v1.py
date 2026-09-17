"""ArtworkV1Role implementation (v1).

This role handles artwork binary streaming to display clients:
- Sends stream/start with the channel configs the client declares in client/state
- Transfers each image as an announce followed by its parts, one transfer at a time
- Re-announces the stream when the client changes its channel configs
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import TYPE_CHECKING

from aiosendspin.models import pack_binary_header_raw
from aiosendspin.models.artwork import (
    ArtworkChannel,
    StreamArtworkChannelConfig,
    StreamRequestFormatArtwork,
    StreamStartArtwork,
    artwork_message_type,
    pack_artwork_announce,
    pack_artwork_cancel,
    pack_artwork_parts,
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
from aiosendspin.util import create_task

if TYPE_CHECKING:
    from aiosendspin.server.client import SendspinClient

MAX_ARTWORK_CHANNELS = 4
# A scheduled image is announced at most this long before its timestamp.
MAX_ANNOUNCE_LEAD_US = 20_000_000


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
        # Images waiting for their transfer: channel -> [(image, timestamp_us)], a current
        # image before a scheduled one. The first channel whose next image is due to be
        # announced is sent next.
        self._queued: dict[int, list[tuple[bytes, int]]] = {}
        # Channel and timestamp of the transfer announced and not yet fully sent.
        self._in_flight: int | None = None
        self._in_flight_timestamp_us = 0
        # Channels whose last announced image was scheduled and may be pending on the client.
        self._scheduled_announced: set[int] = set()
        self._transfer_task: asyncio.Task[None] | None = None
        self._queue_changed = asyncio.Event()

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

    def requires_activation_state(self) -> bool:
        """Artwork waits for its client/state channels unless the hello declared them."""
        # DEPRECATED(spec-pr-195): remove in aiosendspin <version>
        return self._client.info.artwork_support is None

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
            self._cancel_in_flight()
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
        """
        Send an image for a channel.

        An image with a future timestamp replaces only a scheduled image still queued or
        in flight for the channel; any other image replaces every one. Does nothing when
        the channel is not currently streamed.

        Args:
            channel: Channel number (0-3).
            image_data: Encoded image bytes; empty clears the channel.
            timestamp_us: Server time in microseconds when the image should be displayed.
        """
        # TODO: should we raise instead of swallowing when no transport?
        if not self.has_connection() or channel not in self.get_channel_configs():
            return
        now_us = self._client._server.clock.now_us()  # noqa: SLF001
        # DEPRECATED(spec-pr-188): remove in aiosendspin <version>
        if self.uses_single_message_framing():
            self._queued.pop(channel, None)
            if timestamp_us - MAX_ANNOUNCE_LEAD_US <= now_us:
                self._send_single_message(channel, image_data, timestamp_us)
            else:
                self._queued[channel] = [(image_data, timestamp_us)]
                self._start_transfers()
            return
        scheduled = timestamp_us > now_us
        queued = self._current_queued(channel, now_us) if scheduled else []
        queued.append((image_data, timestamp_us))
        if self._discard_client_scheduled(channel, now_us, keep_current=scheduled):
            self._queued.pop(channel, None)
            self._queued = {channel: queued, **self._queued}
        else:
            self._queued[channel] = queued
        self._start_transfers()

    def send_artwork_cleared(self, channel: int, timestamp_us: int) -> None:
        """
        Clear a channel by sending it an empty image.

        Does nothing when the channel is not currently streamed.

        Args:
            channel: Channel number (0-3).
            timestamp_us: Server time in microseconds when the channel should clear.
        """
        self.send_artwork(channel, b"", timestamp_us)

    def cancel_scheduled_artwork(self, channel: int) -> bool:
        """
        Discard the channel's scheduled image, keeping its current one.

        Returns False when the client can only drop a scheduled image by receiving the
        current image again, which the caller then sends.
        """
        if not self.has_connection() or channel not in self.get_channel_configs():
            return True
        # DEPRECATED(spec-pr-188): remove in aiosendspin <version>
        if self.uses_single_message_framing():
            self._queued.pop(channel, None)
            return False
        now_us = self._client._server.clock.now_us()  # noqa: SLF001
        if queued := self._current_queued(channel, now_us):
            self._queued[channel] = queued
        else:
            self._queued.pop(channel, None)
        if self._discard_client_scheduled(channel, now_us, keep_current=True):
            self._start_transfers()
        return True

    # DEPRECATED(spec-pr-188): remove in aiosendspin <version>
    def uses_single_message_framing(self) -> bool:
        """Whether the client predates transfers, having declared its channels in the hello."""
        return self._client.info.artwork_support is not None

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
            self._cancel_in_flight()
            self._queued.clear()
            # A channel must be cleared before the stream/start that stops streaming it.
            now_us = self._client._server.clock.now_us()  # noqa: SLF001
            for channel_num, config in enumerate(new_configs):
                if config.source is ArtworkSource.NONE and old_configs[channel_num] != config:
                    self._send_clear_now(channel_num, now_us)

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
        self._stop_transfer_task()
        self._queued.clear()
        self._in_flight = None
        self._scheduled_announced.clear()
        self._stream_started = False
        self._channels = []

    # DEPRECATED(spec-pr-188): remove in aiosendspin <version>
    def _send_single_message(self, channel: int, image_data: bytes, timestamp_us: int) -> None:
        """Send `image_data` as one `[type][timestamp][image]` message."""
        message_type = artwork_message_type(channel)
        self._client.send_binary(
            pack_binary_header_raw(message_type, timestamp_us) + image_data,
            role_family=self.role_family,
            timestamp_us=timestamp_us,
            message_type=message_type,
        )

    def _send_clear_now(self, channel: int, timestamp_us: int) -> None:
        """Enqueue a clear for `channel` at once; no transfer may be in flight."""
        # DEPRECATED(spec-pr-188): remove in aiosendspin <version>
        if self.uses_single_message_framing():
            self._send_single_message(channel, b"", timestamp_us)
            return
        self._send_transfer_message(channel, pack_artwork_announce(channel, timestamp_us, 0))

    def _send_transfer_message(
        self, channel: int, data: bytes, *, epoch_exempt: bool = False
    ) -> None:
        """Enqueue a transfer message in FIFO order with the role's other messages."""
        self._client.send_binary(
            data,
            role_family=self.role_family,
            timestamp_us=0,
            message_type=artwork_message_type(channel),
            epoch_exempt=epoch_exempt,
        )

    def _current_queued(self, channel: int, now_us: int) -> list[tuple[bytes, int]]:
        """Return the channel's latest queued image that is already due, if any."""
        return [entry for entry in self._queued.get(channel, []) if entry[1] <= now_us][-1:]

    def _discard_client_scheduled(self, channel: int, now_us: int, *, keep_current: bool) -> bool:
        """
        Make the client discard the channel's scheduled image.

        Cancels the channel's transfer in flight unless `keep_current` and it carries a
        current image. Returns whether a transfer was cancelled.
        """
        if channel == self._in_flight:
            if keep_current and self._in_flight_timestamp_us <= now_us:
                return False
            self._cancel_in_flight()
            return True
        if channel in self._scheduled_announced:
            self._scheduled_announced.discard(channel)
            # Must survive a later cancel dropping the role's queued binary.
            self._send_transfer_message(channel, pack_artwork_cancel(channel), epoch_exempt=True)
        return False

    def _start_transfers(self) -> None:
        """Run the transfer task, or wake it to reconsider the queue."""
        if self._transfer_task is None or self._transfer_task.done():
            self._transfer_task = create_task(self._run_transfers())
        else:
            self._queue_changed.set()

    def _stop_transfer_task(self) -> None:
        """Stop the transfer task, leaving a transfer it started in flight."""
        if self._transfer_task is not None:
            self._transfer_task.cancel()
            self._transfer_task = None

    def _cancel_in_flight(self) -> None:
        """Stop the transfer task, drop queued artwork binary, and cancel a transfer in flight."""
        self._stop_transfer_task()
        self._client.drop_pending_binary([self.role_family])
        if self._in_flight is not None:
            # Must reach the client even when a stream/end follows and drops the role's binary.
            self._send_transfer_message(
                self._in_flight, pack_artwork_cancel(self._in_flight), epoch_exempt=True
            )
            self._scheduled_announced.discard(self._in_flight)
            self._in_flight = None

    async def _run_transfers(self) -> None:
        """Transfer the queued images one at a time, each part once the previous was sent."""
        clock = self._client._server.clock  # noqa: SLF001
        while self._queued:
            self._queue_changed.clear()
            now_us = clock.now_us()
            due = [
                (queued[0][1] - MAX_ANNOUNCE_LEAD_US, channel)
                for channel, queued in self._queued.items()
            ]
            channel = next((channel for announce_us, channel in due if announce_us <= now_us), None)
            if channel is None:
                wait_s = (min(due)[0] - now_us) / 1_000_000
                with suppress(TimeoutError):
                    await asyncio.wait_for(self._queue_changed.wait(), wait_s)
                continue
            queued = self._queued[channel]
            image, timestamp_us = queued.pop(0)
            if not queued:
                del self._queued[channel]
            # DEPRECATED(spec-pr-188): remove in aiosendspin <version>
            if self.uses_single_message_framing():
                self._send_single_message(channel, image, timestamp_us)
                continue
            self._in_flight = channel
            self._in_flight_timestamp_us = timestamp_us
            if timestamp_us > clock.now_us():
                self._scheduled_announced.add(channel)
            else:
                self._scheduled_announced.discard(channel)
            self._send_transfer_message(
                channel, pack_artwork_announce(channel, timestamp_us, len(image))
            )
            for part in pack_artwork_parts(channel, image):
                await self._client.wait_role_drained(self.role_family)
                self._send_transfer_message(channel, part)
            await self._client.wait_role_drained(self.role_family)
            self._in_flight = None


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
