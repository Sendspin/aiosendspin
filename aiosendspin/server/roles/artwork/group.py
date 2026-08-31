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
        self._current_artwork: dict[ArtworkSource, Image.Image] = {}

    def on_member_join(self, role: Role) -> None:
        """Send current artwork to newly joined member."""
        self._send_artwork_to_role(role)

    def _send_artwork_to_role(self, role: Role) -> None:
        """Send current artwork for all channels to a role."""
        if not isinstance(role, ArtworkRoleProtocol):
            return
        channel_configs = role.get_channel_configs()
        if not channel_configs:
            return

        for channel_num, channel_config in channel_configs.items():
            if channel_config.source == ArtworkSource.NONE:
                continue
            artwork = self._current_artwork.get(channel_config.source)
            if artwork is not None:
                self._schedule_send_artwork(role, artwork, channel_num, channel_config)

    def _schedule_send_artwork(
        self,
        role: ArtworkRoleProtocol,
        image: Image.Image,
        channel: int,
        channel_config: ArtworkChannel,
    ) -> None:
        """Schedule artwork send as a background task."""
        # Pillow images are not safe to share across concurrent encode tasks.
        create_task(self._send_artwork_to_role_channel(role, image.copy(), channel, channel_config))

    async def _send_artwork_to_role_channel(
        self,
        role: ArtworkRoleProtocol,
        image: Image.Image,
        channel: int,
        channel_config: ArtworkChannel,
    ) -> None:
        """Send artwork to a specific role channel."""
        try:
            timestamp_us = self._group._server.clock.now_us()  # noqa: SLF001
            img_data = await asyncio.to_thread(
                self._process_and_encode_image,
                image,
                channel_config.width,
                channel_config.height,
                channel_config.format,
            )
            role.send_artwork(channel, img_data, timestamp_us)
        except Exception:
            logger.exception("Failed to send artwork update")

    def get_album_artwork(self) -> Image.Image | None:
        """Return current album artwork, or None if not set."""
        return self._current_artwork.get(ArtworkSource.ALBUM)

    def get_artist_artwork(self) -> Image.Image | None:
        """Return current artist artwork, or None if not set."""
        return self._current_artwork.get(ArtworkSource.ARTIST)

    async def set_album_artwork(self, image: Image.Image | None) -> None:
        """Set or clear album artwork.

        Args:
            image: The artwork image to set, or None to clear.
        """
        await self._set_artwork(ArtworkSource.ALBUM, image)

    async def set_artist_artwork(self, image: Image.Image | None) -> None:
        """Set or clear artist artwork.

        Args:
            image: The artwork image to set, or None to clear.
        """
        await self._set_artwork(ArtworkSource.ARTIST, image)

    async def _set_artwork(self, source: ArtworkSource, image: Image.Image | None) -> None:
        """Set or clear artwork for a source type."""
        event_timestamp_us = self._group._server.clock.now_us()  # noqa: SLF001
        if image is None:
            self._current_artwork.pop(source, None)
        else:
            self._current_artwork[source] = image

        send_tasks = []
        for role in self._members:
            if not isinstance(role, ArtworkRoleProtocol):
                continue
            channel_configs = role.get_channel_configs()
            if not channel_configs:
                continue
            for channel_num, channel_config in channel_configs.items():
                if channel_config.source == source:
                    if image is None:
                        timestamp_us = self._group._server.clock.now_us()  # noqa: SLF001
                        role.send_artwork_cleared(channel_num, timestamp_us)
                    else:
                        # Pillow images are not safe to share across concurrent encode tasks.
                        send_tasks.append(
                            self._send_artwork_to_role_channel(
                                role, image.copy(), channel_num, channel_config
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
