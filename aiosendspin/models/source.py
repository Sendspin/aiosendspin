"""Source role protocol messages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .base import SendspinConfig, SendspinModel
from .types import AudioCodec, ClientMessage, SignalState


# Client -> Server: client/hello source@v1 support object
@dataclass
class ClientHelloSourceFeatures(SendspinModel):
    """Optional feature hints for a source client."""

    line_sense: bool | None = None
    """True if the source reports signal/line-sense presence."""

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class ClientHelloSourceSupport(SendspinModel):
    """Source support configuration."""

    features: ClientHelloSourceFeatures | None = None
    """Optional feature hints."""

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True


# Client -> Server: client/state source object
@dataclass
class SourceStatePayload(SendspinModel):
    """Source state reported in client/state."""

    signal: SignalState | None = None
    """Signal/line-sense presence, only if 'line_sense' is supported."""

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True


# Server -> Client: server/command source object
@dataclass
class SourceCommandServerPayload(SendspinModel):
    """Source object in server/command message (server-requested streaming change)."""

    command: Literal["start", "stop"]
    """'start' requests the source begin streaming, 'stop' requests it stop."""

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True


# Client -> Server: client-stream/start source object
@dataclass
class ClientStreamStartSource(SendspinModel):
    """Source object in client-stream/start message."""

    codec: AudioCodec
    """Codec of the input stream."""
    channels: int
    sample_rate: int
    bit_depth: int
    codec_header: str | None = None
    """Standard Base64 codec header when required."""

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class ClientStreamStartPayload(SendspinModel):
    """Payload for client-stream/start message."""

    source: ClientStreamStartSource

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class ClientStreamStartMessage(ClientMessage):
    """Message sent by a source client to announce the active input stream format."""

    payload: ClientStreamStartPayload
    type: Literal["client-stream/start", "client_stream/start"] = "client-stream/start"


# Client -> Server: client-stream/end
@dataclass
class ClientStreamEndMessage(ClientMessage):
    """Message sent by a source client to end the current input stream."""

    type: Literal["client-stream/end", "client_stream/end"] = "client-stream/end"
