"""Sendspin Noise Protocol constants and identifiers."""

from __future__ import annotations

import hashlib
from typing import Final

PROTOCOL_VERSION: Final[int] = 1

# Published constant PSK used when no other PSK applies; public, so authenticates nothing.
SENTINEL_PSK: Final[bytes] = hashlib.sha256(b"sendspin-sentinel-psk-v1").digest()

PSK_ID_LABEL: Final[bytes] = b"sendspin-psk-id-v1"

# Cleartext init / handshake JSON message-type tags.
INIT_TYPE_CLIENT: Final[str] = "client/init"
INIT_TYPE_SERVER: Final[str] = "server/init"
ERROR_TYPE_SERVER: Final[str] = "server/error"
HANDSHAKE_TYPE: Final[str] = "noise/handshake"

# Transport-mode binary message type at byte 0 of decrypted plaintext.
MSG_TYPE_JSON_BODY: Final[int] = 0

# Fragment frame types (bit 0 is the last-fragment flag).
MSG_TYPE_FRAGMENT_MORE: Final[int] = 2
MSG_TYPE_FRAGMENT_END: Final[int] = 3

# Noise's 65535-byte transport message limit minus the 16-byte AEAD tag.
MAX_TRANSPORT_PLAINTEXT: Final[int] = 65535 - 16
