"""JSON dataclasses for the cleartext init exchange and Noise handshake frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mashumaro.types import Discriminator

from aiosendspin.models.base import SendspinConfig, SendspinModel
from aiosendspin.models.types import PairAbortReason, ServerMessage


@dataclass
class ClientInitPayload(SendspinModel):
    """Cleartext ``client/init`` payload — the client opens the conversation."""

    client_id: str
    """Client's static public key, 43-char base64url X25519 with no padding."""
    version: int
    """Core protocol version. Must be ``1`` for this implementation."""
    suite: str
    """The Noise cipher suite the client picked (see ``NoiseCipherSuite``)."""


@dataclass
class ClientInitMessage(SendspinModel):
    """Envelope for ``ClientInitPayload``."""

    payload: ClientInitPayload
    type: Literal["client/init"] = "client/init"


@dataclass
class ServerInitPayload(SendspinModel):
    """Cleartext ``server/init`` payload — server's response with its identity."""

    server_id: str
    """Server's static public key, 43-char base64url X25519 with no padding."""
    version: int
    """Core protocol version. Must be ``1`` for this implementation."""


@dataclass
class ServerInitMessage(SendspinModel):
    """Envelope for ``ServerInitPayload``."""

    payload: ServerInitPayload
    type: Literal["server/init"] = "server/init"


@dataclass
class NoiseHandshakePayload(SendspinModel):
    """Carries one Noise handshake message (base64url-encoded ciphertext)."""

    data: str
    """Base64url-encoded Noise handshake bytes, no padding."""


@dataclass
class NoiseHandshakeMessage(ServerMessage):
    """Envelope for ``NoiseHandshakePayload``. Used for both Noise messages 1 and 2."""

    payload: NoiseHandshakePayload
    type: Literal["noise/handshake"] = "noise/handshake"


@dataclass
class NoiseMsg1Payload(SendspinModel):
    """Plaintext payload encrypted *inside* Noise handshake message 1 (server → client)."""

    psk_id: str
    """43-char base64url SHA-256 of the PSK (see ``psk_id_for``)."""
    psk_category: str
    """Category the server is using the referenced PSK as, as a ``PskCategory`` code."""


@dataclass
class NoiseMsg2Payload(SendspinModel):
    """Plaintext payload encrypted *inside* Noise handshake message 2 (client → server)."""


@dataclass
class PairingMessage(SendspinModel):
    """Discriminated base for pairing envelopes, keyed on the ``type`` field."""

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        discriminator = Discriminator(field="type", include_subtypes=True)


@dataclass
class ClientPairFinalizePayload(SendspinModel):
    """Encrypted ``client/pair-finalize`` payload — the client delivers the long-term PSK."""

    long_term_psk: str | None = None
    """43-char base64url 32-byte PSK, sent directly. Pairing PSK flow only."""
    wrapped_psk: str | None = None
    """The PSK sealed under the CPace-derived wrap key (64-char base64url).

    Present only in pairing-code flows.
    """

    class Config(SendspinConfig):
        """Omit the field the flow doesn't use."""

        omit_none = True


@dataclass
class ClientPairFinalizeMessage(PairingMessage):
    """Envelope for ``ClientPairFinalizePayload``."""

    payload: ClientPairFinalizePayload
    type: Literal["client/pair-finalize"] = "client/pair-finalize"


@dataclass
class ServerPairFinalizePayload(SendspinModel):
    """Encrypted ``server/pair-finalize`` payload — empty ack that the server persisted."""


@dataclass
class ServerPairFinalizeMessage(PairingMessage):
    """Envelope for ``ServerPairFinalizePayload``."""

    payload: ServerPairFinalizePayload = field(default_factory=ServerPairFinalizePayload)
    type: Literal["server/pair-finalize"] = "server/pair-finalize"


@dataclass
class PairAbortPayload(SendspinModel):
    """``pair/abort`` payload — aborts a pairing attempt; sender closes after sending."""

    reason: PairAbortReason
    """Why the pairing attempt was aborted."""


@dataclass
class PairAbortMessage(PairingMessage):
    """Envelope for ``PairAbortPayload`` (sent by either side)."""

    payload: PairAbortPayload
    type: Literal["pair/abort"] = "pair/abort"


@dataclass
class ClientPairPendingPayload(SendspinModel):
    """``client/pair-pending`` payload — a gesture-gated attempt awaits a pairing window."""

    pairing_index: int
    """Pairing ``server/activate`` message count received since the last Noise handshake."""


@dataclass
class ClientPairPendingMessage(PairingMessage):
    """Envelope for ``ClientPairPendingPayload``."""

    payload: ClientPairPendingPayload
    type: Literal["client/pair-pending"] = "client/pair-pending"


@dataclass
class ClientPairInitPayload(SendspinModel):
    """``client/pair-init`` payload — signals readiness for the pairing code-pairing flow."""

    pairing_index: int
    """Pairing ``server/activate`` message count received since the last Noise handshake."""
    commit_B: str | None = None  # noqa: N815 - spec wire field name
    """``SHA-256("sendspin-pair-commit-v1" || nonce_B)`` (43-char base64url).

    Present in dynamic pairing code and absent in static pairing code.
    """

    class Config(SendspinConfig):
        """Omit the optional commitment when absent (static pairing code)."""

        omit_none = True


@dataclass
class ClientPairInitMessage(PairingMessage):
    """Envelope for ``ClientPairInitPayload``."""

    payload: ClientPairInitPayload
    type: Literal["client/pair-init"] = "client/pair-init"


@dataclass
class ServerPairInitPayload(SendspinModel):
    """``server/pair-init`` payload — the server's nonce contribution (dynamic pairing code)."""

    nonce_A: str  # noqa: N815 - spec wire field name
    """32 bytes from a CSPRNG, base64url-encoded (43 chars)."""


@dataclass
class ServerPairInitMessage(PairingMessage):
    """Envelope for ``ServerPairInitPayload``."""

    payload: ServerPairInitPayload
    type: Literal["server/pair-init"] = "server/pair-init"


@dataclass
class ServerPairAuthPayload(SendspinModel):
    """``server/pair-auth`` payload — the server's CPace public share."""

    pake_msg_1: str
    """CPace public share ``Ya`` (43-char base64url)."""


@dataclass
class ServerPairAuthMessage(PairingMessage):
    """Envelope for ``ServerPairAuthPayload``."""

    payload: ServerPairAuthPayload
    type: Literal["server/pair-auth"] = "server/pair-auth"


@dataclass
class ClientPairAuthPayload(SendspinModel):
    """``client/pair-auth`` payload — the client's CPace public share."""

    pake_msg_2: str
    """CPace public share ``Yb`` (43-char base64url)."""


@dataclass
class ClientPairAuthMessage(PairingMessage):
    """Envelope for ``ClientPairAuthPayload``."""

    payload: ClientPairAuthPayload
    type: Literal["client/pair-auth"] = "client/pair-auth"


@dataclass
class ServerPairConfirmPayload(SendspinModel):
    """``server/pair-confirm`` payload — the server's MCF confirmation tag."""

    server_kc: str
    """CPace MCF tag ``Ta`` (HMAC-SHA-512, 86-char base64url)."""


@dataclass
class ServerPairConfirmMessage(PairingMessage):
    """Envelope for ``ServerPairConfirmPayload``."""

    payload: ServerPairConfirmPayload
    type: Literal["server/pair-confirm"] = "server/pair-confirm"


@dataclass
class ClientPairConfirmPayload(SendspinModel):
    """``client/pair-confirm`` payload — the client's MCF tag and wrapped commitment opening."""

    client_kc: str
    """CPace MCF tag ``Tb`` (HMAC-SHA-512, 86-char base64url)."""
    wrapped_nonce_B: str | None = None  # noqa: N815 - spec wire field name
    """48-byte wrapping of the ``commit_B`` preimage (64-char base64url). Dynamic only."""

    class Config(SendspinConfig):
        """Omit the optional wrapped nonce opening when absent (static pairing code)."""

        omit_none = True


@dataclass
class ClientPairConfirmMessage(PairingMessage):
    """Envelope for ``ClientPairConfirmPayload``."""

    payload: ClientPairConfirmPayload
    type: Literal["client/pair-confirm"] = "client/pair-confirm"
