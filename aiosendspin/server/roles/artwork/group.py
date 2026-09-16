"""ArtworkGroupRole - group-level artwork coordination."""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from io import BytesIO
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

from PIL import Image

from aiosendspin.models import BinaryMessageType, pack_binary_header_raw
from aiosendspin.models.artwork import ArtworkChannel
from aiosendspin.models.types import ArtworkSource, PictureFormat
from aiosendspin.server.roles.artwork.events import ArtworkClearedEvent, ArtworkUpdatedEvent
from aiosendspin.server.roles.artwork.types import ArtworkRoleProtocol
from aiosendspin.server.roles.base import GroupRole, Role
from aiosendspin.server.roles.scheduled_state import ScheduledRoleState
from aiosendspin.util import create_task

if TYPE_CHECKING:
    from aiosendspin.server.group import SendspinGroup


logger = logging.getLogger(__name__)


class ArtworkGroupRole(GroupRole):
    """Coordinate artwork across a group.

    Stores current raw artwork images and pushes encoded images to subscribed
    ArtworkRoles based on their channel preferences.
    """

    role_family = "artwork"

    def __init__(self, group: SendspinGroup) -> None:
        """Initialize ArtworkGroupRole."""
        super().__init__(group)
        self._artwork: dict[ArtworkSource, ScheduledRoleState[Image.Image]] = {}
        # Serialize sends per role channel so a current image never lands after a scheduled
        # one. Kept across a warm reconnect, which reuses the role, so work started before
        # it still runs first.
        self._send_locks: WeakKeyDictionary[ArtworkRoleProtocol, dict[int, asyncio.Lock]] = (
            WeakKeyDictionary()
        )
        self._replay_tasks: dict[ArtworkRoleProtocol, dict[int, asyncio.Task[None]]] = {}

    def on_member_join(self, role: Role) -> None:
        """Send current artwork to newly joined member."""
        if isinstance(role, ArtworkRoleProtocol):
            self.send_current_artwork(role)

    def on_member_leave(self, role: Role) -> None:
        """Stop sending artwork to the departing member."""
        if isinstance(role, ArtworkRoleProtocol):
            for task in self._replay_tasks.pop(role, {}).values():
                task.cancel()

    def send_current_artwork(self, role: ArtworkRoleProtocol) -> None:
        """Schedule the current, then any scheduled, image for each channel the role streams."""
        now_us = self._now_us()
        for channel_num, channel_config in role.get_channel_configs().items():
            if channel_config.source == ArtworkSource.NONE:
                continue
            state = self._artwork.get(channel_config.source)
            if state is None or (
                state.current(now_us) is None and state.pending_timestamp_us is None
            ):
                continue
            self._schedule_replay(role, channel_num, channel_config)

    def _schedule_replay(
        self, role: ArtworkRoleProtocol, channel: int, channel_config: ArtworkChannel
    ) -> None:
        """Replay the channel's artwork in the background, replacing a replay in progress."""
        tasks = self._replay_tasks.setdefault(role, {})
        if (previous := tasks.get(channel)) is not None:
            previous.cancel()
        task = create_task(self._replay_artwork(role, channel, channel_config))
        tasks[channel] = task
        task.add_done_callback(partial(_forget_task, tasks, channel))

    async def _replay_artwork(
        self,
        role: ArtworkRoleProtocol,
        channel: int,
        channel_config: ArtworkChannel,
    ) -> None:
        """Send the current, then any scheduled, image of the channel's source."""
        async with self._send_lock(role, channel):
            state = self._artwork[channel_config.source]
            now_us = self._now_us()
            current = state.current(now_us)
            scheduled = state.pending
            scheduled_us = state.pending_timestamp_us
            if current is not None:
                await self._encode_and_send(role, current, channel, channel_config, now_us)
            if scheduled_us is not None:
                await self._encode_and_send(role, scheduled, channel, channel_config, scheduled_us)

    async def _send_artwork_to_role_channel(
        self,
        role: ArtworkRoleProtocol,
        image: Image.Image | None,
        channel: int,
        channel_config: ArtworkChannel,
        timestamp_us: int,
    ) -> None:
        """Send artwork to a specific role channel."""
        async with self._send_lock(role, channel):
            await self._encode_and_send(role, image, channel, channel_config, timestamp_us)

    async def _encode_and_send(
        self,
        role: ArtworkRoleProtocol,
        image: Image.Image | None,
        channel: int,
        channel_config: ArtworkChannel,
        timestamp_us: int,
    ) -> None:
        """Encode `image` for the channel and send it, or clear the channel for None."""
        try:
            if image is None:
                role.send_artwork_cleared(channel, timestamp_us)
                return
            # ArtworkChannel requires these for every source but none, which is never sent.
            assert channel_config.width is not None
            assert channel_config.height is not None
            assert channel_config.format is not None
            img_data = await asyncio.to_thread(
                self._process_and_encode_image,
                # Pillow images are not safe to share across concurrent encode tasks.
                image.copy(),
                channel_config.width,
                channel_config.height,
                channel_config.format,
            )
            current = role.get_channel_configs().get(channel)
            if current is None or _encoding(current) != _encoding(channel_config):
                # Reconfigured while encoding; the new configuration triggers its own send.
                return
            role.send_artwork(channel, img_data, timestamp_us)
        except Exception:
            logger.exception("Failed to send artwork update")

    def _send_lock(self, role: ArtworkRoleProtocol, channel: int) -> asyncio.Lock:
        return self._send_locks.setdefault(role, {}).setdefault(channel, asyncio.Lock())

    def get_album_artwork(self) -> Image.Image | None:
        """Return current album artwork, or None if not set."""
        return self._current_artwork(ArtworkSource.ALBUM)

    def get_artist_artwork(self) -> Image.Image | None:
        """Return current artist artwork, or None if not set."""
        return self._current_artwork(ArtworkSource.ARTIST)

    async def set_album_artwork(
        self, image: Image.Image | None, *, timestamp_us: int | None = None
    ) -> None:
        """Set or clear album artwork.

        A future `timestamp_us` schedules the artwork, or its clear, to take effect then,
        replacing artwork already scheduled. It is sent to clients at most 20 seconds
        ahead, and the artwork event fires now, carrying that timestamp. Otherwise the
        artwork applies at once and cancels scheduled artwork. To show two images in
        sequence, schedule the second only after the first took effect.

        Args:
            image: The artwork image to set, or None to clear.
            timestamp_us: Server time in microseconds the artwork takes effect.
        """
        await self._set_artwork(ArtworkSource.ALBUM, image, timestamp_us=timestamp_us)

    async def set_artist_artwork(
        self, image: Image.Image | None, *, timestamp_us: int | None = None
    ) -> None:
        """Set or clear artist artwork.

        A future `timestamp_us` schedules the artwork, or its clear, to take effect then,
        replacing artwork already scheduled. It is sent to clients at most 20 seconds
        ahead, and the artwork event fires now, carrying that timestamp. Otherwise the
        artwork applies at once and cancels scheduled artwork. To show two images in
        sequence, schedule the second only after the first took effect.

        Args:
            image: The artwork image to set, or None to clear.
            timestamp_us: Server time in microseconds the artwork takes effect.
        """
        await self._set_artwork(ArtworkSource.ARTIST, image, timestamp_us=timestamp_us)

    async def _set_artwork(
        self,
        source: ArtworkSource,
        image: Image.Image | None,
        *,
        timestamp_us: int | None = None,
    ) -> None:
        """Set, schedule or clear artwork for a source type."""
        now_us = self._now_us()
        state = self._artwork.setdefault(source, ScheduledRoleState())
        state.current(now_us)
        if timestamp_us is not None and timestamp_us > now_us:
            event_timestamp_us = timestamp_us
            state.schedule(image, event_timestamp_us)
        else:
            event_timestamp_us = now_us if timestamp_us is None else timestamp_us
            state.apply(image, event_timestamp_us)

        await asyncio.gather(
            *(
                self._send_artwork_to_role_channel(
                    role, image, channel_num, channel_config, event_timestamp_us
                )
                for role, channel_num, channel_config in self._member_channels(source)
            )
        )

        if image is None:
            self.emit_group_event(
                ArtworkClearedEvent(source=source, timestamp_us=event_timestamp_us)
            )
            return
        self.emit_group_event(
            ArtworkUpdatedEvent(
                source=source,
                timestamp_us=event_timestamp_us,
                width=image.width,
                height=image.height,
            )
        )

    async def cancel_scheduled_artwork(self, source: ArtworkSource) -> None:
        """Cancel the artwork scheduled for a source type, if any, keeping the current one."""
        state = self._artwork.get(source)
        if state is None:
            return
        now_us = self._now_us()
        current = state.current(now_us)
        if state.pending_timestamp_us is None:
            return
        state.apply(current, now_us)
        await asyncio.gather(
            *(
                self._cancel_role_channel(role, current, channel_num, channel_config, now_us)
                for role, channel_num, channel_config in self._member_channels(source)
            )
        )

    async def _cancel_role_channel(
        self,
        role: ArtworkRoleProtocol,
        current: Image.Image | None,
        channel: int,
        channel_config: ArtworkChannel,
        timestamp_us: int,
    ) -> None:
        """Cancel the channel's scheduled image, re-sending the current one where required."""
        async with self._send_lock(role, channel):
            if not role.cancel_scheduled_artwork(channel):
                await self._encode_and_send(role, current, channel, channel_config, timestamp_us)

    def _current_artwork(self, source: ArtworkSource) -> Image.Image | None:
        state = self._artwork.get(source)
        return None if state is None else state.current(self._now_us())

    def _member_channels(
        self, source: ArtworkSource
    ) -> list[tuple[ArtworkRoleProtocol, int, ArtworkChannel]]:
        """Return each member channel streaming `source`."""
        return [
            (role, channel_num, channel_config)
            for role in self._members
            if isinstance(role, ArtworkRoleProtocol)
            for channel_num, channel_config in role.get_channel_configs().items()
            if channel_config.source == source
        ]

    def _letterbox_image(
        self, image: Image.Image, target_width: int, target_height: int
    ) -> Image.Image:
        """Resize image to fit within target dimensions while preserving aspect ratio."""
        image_aspect = image.width / image.height
        target_aspect = target_width / target_height

        if image_aspect > target_aspect:
            new_width = target_width
            new_height = int(target_width / image_aspect)
        else:
            new_height = target_height
            new_width = int(target_height * image_aspect)

        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        letterboxed = Image.new("RGB", (target_width, target_height), (0, 0, 0))
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        letterboxed.paste(resized, (x_offset, y_offset))

        return letterboxed

    def _process_and_encode_image(
        self,
        image: Image.Image,
        width: int,
        height: int,
        art_format: PictureFormat,
    ) -> bytes:
        """Process and encode image for client."""
        resized_image = self._letterbox_image(image, width, height)

        with BytesIO() as img_bytes:
            if art_format == PictureFormat.JPEG:
                resized_image.save(img_bytes, format="JPEG", quality=85)
            elif art_format == PictureFormat.PNG:
                resized_image.save(img_bytes, format="PNG", compress_level=6)
            elif art_format == PictureFormat.BMP:
                resized_image.save(img_bytes, format="BMP")
            else:
                raise NotImplementedError(f"Unsupported artwork format: {art_format}")
            img_bytes.seek(0)
            return img_bytes.read()

    def get_binary_message_type(self, channel: int) -> int:
        """Get the binary message type for an artwork channel."""
        return BinaryMessageType.ARTWORK_CHANNEL_0.value + channel

    def pack_artwork_header(self, channel: int, timestamp_us: int) -> bytes:
        """Pack binary header for artwork message."""
        message_type = self.get_binary_message_type(channel)
        return pack_binary_header_raw(message_type, timestamp_us)


def _forget_task(
    tasks: dict[int, asyncio.Task[None]], channel: int, task: asyncio.Task[None]
) -> None:
    """Remove a finished replay task unless a newer one replaced it."""
    if tasks.get(channel) is task:
        del tasks[channel]


def _encoding(channel: ArtworkChannel) -> tuple[object, ...]:
    """Return the fields an encoded image depends on."""
    return (channel.source, channel.format, channel.width, channel.height)
