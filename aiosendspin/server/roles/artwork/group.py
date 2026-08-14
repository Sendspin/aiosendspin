"""ArtworkGroupRole - group-level artwork coordination."""

from __future__ import annotations

import asyncio
import logging
from io import BytesIO
from typing import TYPE_CHECKING

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
        self._artwork: dict[ArtworkSource, ScheduledRoleState[Image.Image, None]] = {}
        self._send_locks: dict[tuple[int, int], asyncio.Lock] = {}

    def on_member_join(self, role: Role) -> None:
        """Send current artwork to newly joined member."""
        if isinstance(role, ArtworkRoleProtocol):
            self.send_current_artwork(role)

    def send_current_artwork(self, role: ArtworkRoleProtocol) -> None:
        """Schedule the current image for each channel the role streams."""
        for channel_num, channel_config in role.get_channel_configs().items():
            if channel_config.source == ArtworkSource.NONE:
                continue
            if self._has_replayable_artwork(channel_config.source):
                self._schedule_artwork_replay(role, channel_num, channel_config)

    def _schedule_artwork_replay(
        self,
        role: ArtworkRoleProtocol,
        channel: int,
        channel_config: ArtworkChannel,
    ) -> None:
        create_task(self._send_artwork_replay(role, channel, channel_config))

    async def _send_artwork_replay(
        self,
        role: ArtworkRoleProtocol,
        channel: int,
        channel_config: ArtworkChannel,
    ) -> None:
        async with self._send_lock(role, channel):
            state = self._artwork_state(channel_config.source)
            current = state.current(self._now_us())
            pending = state.pending
            pending_effective_us = state.pending_effective_us
            if current is not None:
                await self._encode_and_send_artwork(
                    role,
                    current.copy(),
                    channel,
                    channel_config,
                    self._now_us(),
                )
            elif pending_effective_us is not None:
                await self._encode_and_send_artwork(
                    role,
                    None,
                    channel,
                    channel_config,
                    self._now_us(),
                )
            if pending_effective_us is not None:
                await self._encode_and_send_artwork(
                    role,
                    pending.copy() if pending is not None else None,
                    channel,
                    channel_config,
                    pending_effective_us,
                )

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
            await self._encode_and_send_artwork(role, image, channel, channel_config, timestamp_us)

    def _send_lock(self, role: ArtworkRoleProtocol, channel: int) -> asyncio.Lock:
        return self._send_locks.setdefault((id(role), channel), asyncio.Lock())

    async def _encode_and_send_artwork(
        self,
        role: ArtworkRoleProtocol,
        image: Image.Image | None,
        channel: int,
        channel_config: ArtworkChannel,
        timestamp_us: int,
    ) -> None:
        try:
            # ArtworkChannel requires these for every source but none, which is never sent.
            assert channel_config.width is not None
            assert channel_config.height is not None
            assert channel_config.format is not None
            timestamp_us = self._group._server.clock.now_us()  # noqa: SLF001
            img_data = await asyncio.to_thread(
                self._process_and_encode_image,
                image,
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

    async def _set_artwork(
        self,
        source: ArtworkSource,
        image: Image.Image | None,
        *,
        timestamp_us: int | None = None,
    ) -> None:
        """Set or clear artwork for a source type."""
        now_us = self._now_us()
        state = self._artwork_state(source)
        state.promote_due(now_us)
        event_timestamp_us = now_us if timestamp_us is None else timestamp_us

        if event_timestamp_us > now_us:
            state.schedule(image, None, event_timestamp_us)
        else:
            state.apply(image, event_timestamp_us)

        send_tasks = []
        for role in self._members:
            if not isinstance(role, ArtworkRoleProtocol):
                continue
            channel_configs = role.get_channel_configs()
            if not channel_configs:
                continue
            for channel_num, channel_config in channel_configs.items():
                if channel_config.source == source:
                    send_tasks.append(
                        self._send_artwork_to_role_channel(
                            role,
                            image.copy() if image is not None else None,
                            channel_num,
                            channel_config,
                            event_timestamp_us,
                        )
                    )

        if send_tasks:
            await asyncio.gather(*send_tasks)

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


def _encoding(channel: ArtworkChannel) -> tuple[object, ...]:
    """Return the fields an encoded image depends on."""
    return (channel.source, channel.format, channel.width, channel.height)
