"""Shared data structures for the Sendspin client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from aiosendspin.models.types import SECRET_LOCATIONS, AudioCodec

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

# Longest operator message a client/pair-pending may carry.
_PAIR_PENDING_MESSAGE_MAX_LENGTH = 200

# Renders a dynamic pairing token as a QR code, cleared by a ``None`` call.
type QRCodeDisplay = Callable[[str | None], Awaitable[None]]


class PairingCodeDisplay(Protocol):
    """Shows a derived dynamic pairing code on the device's visual out-channel."""

    def __call__(self, pairing_code: str | None, *, grouped: str | None) -> Awaitable[None]:
        """Show ``pairing_code``, or clear the display when it is ``None``.

        ``grouped`` is the same code in its presentation form (``123-456``), which the
        display should show; it is ``None`` exactly when ``pairing_code`` is.
        """


class PairingCodeSpeaker(Protocol):
    """Speaks a derived dynamic pairing code through the device's audio out-channel."""

    def __call__(self, pairing_code: str | None, *, languages: tuple[str, ...]) -> Awaitable[None]:
        """Speak ``pairing_code``, or stop speaking when it is ``None``.

        Return once emission has started rather than awaiting its completion, since the
        pairing exchange is blocked meanwhile. ``languages`` holds the operator's BCP 47
        preferences in descending order, and is empty when the server declared none.
        """


@dataclass(frozen=True, slots=True)
class PairingSupport:
    """Operator wiring for pairing-code pairing: an operator can perform the pairing gesture here.

    The gesture itself is reported by calling ``SendspinClient.open_pairing_window``.
    Its presence enables offering ``static_pairing_code``, unless
    ``offer_static_pairing_code`` declines it. Any pairing-code out-channel enables
    ``dynamic_pairing_code``, which supersedes ``static_pairing_code``: the client offers
    only one pairing-code method.
    """

    gesture_prompt: Callable[[bool], Awaitable[None]] | None = None
    """Optional operator prompt: awaited with ``True`` when a gated attempt starts
    waiting for a pairing window, and with ``False`` when the wait ends."""
    pairing_code_display: PairingCodeDisplay | None = None
    """Visual out-channel for the derived dynamic pairing code (``digits`` format).

    Called with the same code and its grouped presentation form at the start of every round,
    and with ``None`` for both when the pairing exchange ends so the channel can clear.
    """
    pairing_code_speaker: PairingCodeSpeaker | None = None
    """Spoken out-channel for the derived dynamic pairing code, which also receives the operator's
    language preferences. Called with the same code at the start of every round."""
    qr_code_display: QRCodeDisplay | None = None
    """Display able to render the dynamic pairing token as a QR code (``qr_code`` format).

    Its presence offers the ``qr_code`` emission format. Called with the same token at the
    start of every round, and with ``None`` when the pairing exchange ends so the display
    can clear.
    """
    offer_static_pairing_code: bool = True
    """Whether to offer ``static_pairing_code`` for a device without a per-device code."""
    secret_locations: tuple[str, ...] = ()
    """Where the operator finds a configured static secret, from ``SECRET_LOCATIONS``.

    Applies to every static-secret method the client offers.
    """
    out_channel_suspend: Callable[[bool], Awaitable[None]] | None = None
    """Optional hook for an out-channel that is also a role's output (the speaker playing the
    stream, the display showing artwork): awaited with ``True`` before a dynamic pairing code is
    first emitted, and with ``False`` when the attempt ends.

    Suspend that output locally meanwhile. Streams stay open and their timeline runs on, so a
    player discards the audio scheduled while suspended and resumes in sync.
    """

    pair_pending_message: str | None = None
    """Optional short plain-text sentence for the operator sent in ``client/pair-pending``,
    such as naming the pairing gesture, at most 200 characters.

    The server shows it as unauthenticated text attributed to the device, never as markup.
    """

    def __post_init__(self) -> None:
        """Reject a secret location the descriptor cannot carry, or an overlong message."""
        unknown = sorted(set(self.secret_locations) - SECRET_LOCATIONS)
        if unknown:
            names = ", ".join(unknown)
            raise ValueError(f"unknown secret_locations: {names}")
        if (
            self.pair_pending_message is not None
            and len(self.pair_pending_message) > _PAIR_PENDING_MESSAGE_MAX_LENGTH
        ):
            raise ValueError(
                f"pair_pending_message exceeds {_PAIR_PENDING_MESSAGE_MAX_LENGTH} characters"
            )


@dataclass(slots=True)
class PCMFormat:
    """PCM audio format description."""

    sample_rate: int
    """Sample rate in Hz (e.g., 48000, 44100)."""
    channels: int
    """Number of audio channels (1=mono, 2=stereo)."""
    bit_depth: int
    """Bits per sample (e.g., 16, 24, 32)."""

    def __post_init__(self) -> None:
        """Validate the provided PCM audio format."""
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels not in (1, 2):
            raise ValueError("channels must be 1 or 2")
        if self.bit_depth not in (16, 24, 32):
            raise ValueError("bit_depth must be 16, 24, or 32")

    @property
    def frame_size(self) -> int:
        """Return bytes per PCM frame."""
        return self.channels * (self.bit_depth // 8)


@dataclass(slots=True)
class AudioFormat:
    """Audio format description including codec type."""

    codec: AudioCodec
    """Audio codec used for encoding."""
    pcm_format: PCMFormat
    """Format of decoded PCM audio."""
    codec_header: bytes | None = None
    """Optional codec-specific header bytes (e.g., FLAC streaminfo)."""


@dataclass(slots=True)
class ServerInfo:
    """Information about the connected server."""

    server_id: str
    name: str
    languages: tuple[str, ...] = ()
    """The operator's BCP 47 language preferences in descending order, empty when undeclared."""
    source_codecs: frozenset[AudioCodec] | None = None
    """Codecs the server accepts from a source, or None when server/hello listed none."""
