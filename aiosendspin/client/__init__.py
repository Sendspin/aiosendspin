"""Public interface for the Sendspin client package."""

from aiosendspin.models.types import SECRET_LOCATIONS

from .client import (
    AudioChunkCallback,
    DisconnectCallback,
    GroupUpdateCallback,
    MetadataCallback,
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
    "PCMFormat",
    "PairingCodeDisplay",
    "PairingCodeSpeaker",
    "PairingSupport",
    "QRCodeDisplay",
    "SendspinClient",
    "SendspinTimeFilter",
    "ServerInfo",
    "SourceCapture",
    "StreamEndCallback",
    "StreamStartCallback",
    "VisualizerCallback",
]
