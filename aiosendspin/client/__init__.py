"""Public interface for the Sendspin client package."""

from aiosendspin.models.types import SECRET_LOCATIONS
from aiosendspin.noise.pairing_code import format_pairing_code

from .client import (
    AudioChunkCallback,
    DisconnectCallback,
    GroupUpdateCallback,
    MetadataCallback,
    OutputDelayCallback,
    ScheduledColorCallback,
    ScheduledMetadataCallback,
    SendspinClient,
    StreamEndCallback,
    StreamStartCallback,
    VisualizerCallback,
)
from .listener import ClientListener
from .models import (
    AudioFormat,
    PairingCodeDisplay,
    PairingCodeSpeaker,
    PairingSupport,
    PCMFormat,
    QRCodeDisplay,
    ServerInfo,
)
from .source import SourceCapture
from .time_sync import SendspinTimeFilter

__all__ = [
    "SECRET_LOCATIONS",
    "AudioChunkCallback",
    "AudioFormat",
    "ClientListener",
    "DisconnectCallback",
    "GroupUpdateCallback",
    "MetadataCallback",
    "OutputDelayCallback",
    "PCMFormat",
    "PairingCodeDisplay",
    "PairingCodeSpeaker",
    "PairingSupport",
    "QRCodeDisplay",
    "ScheduledColorCallback",
    "ScheduledMetadataCallback",
    "SendspinClient",
    "SendspinTimeFilter",
    "ServerInfo",
    "SourceCapture",
    "StreamEndCallback",
    "StreamStartCallback",
    "VisualizerCallback",
    "format_pairing_code",
]
