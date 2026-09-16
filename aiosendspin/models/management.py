"""Management command messages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .base import SendspinConfig, SendspinModel
from .core import UnpairedAccess
from .types import (
    ClientMessage,
    ManagementResult,
    ServerMessage,
)


# Server -> Client: server/unpair
@dataclass
class ServerUnpairPayload(SendspinModel):
    """Empty ``server/unpair`` payload."""


@dataclass
class ServerUnpairMessage(ServerMessage):
    """Tells the client to drop this server's pairing record and close (no payload fields)."""

    payload: ServerUnpairPayload = field(default_factory=ServerUnpairPayload)
    type: Literal["server/unpair"] = "server/unpair"


# Server -> Client: management/list-records
@dataclass
class ManagementListRecordsPayload(SendspinModel):
    """Empty ``management/list-records`` payload."""


@dataclass
class ManagementListRecordsMessage(ServerMessage):
    """Requests the client's pairing records (no payload fields)."""

    payload: ManagementListRecordsPayload = field(default_factory=ManagementListRecordsPayload)
    type: Literal["management/list-records"] = "management/list-records"


# Server -> Client: management/add-record
@dataclass
class ManagementAddRecordPayload(SendspinModel):
    """A pairing record to add directly."""

    psk: str
    """43-char base64url 32-byte Sendspin PSK (no padding)."""
    server_id: str | None = None
    """Present for stored-pubkey records, absent for shared-PSK records."""

    class Config(SendspinConfig):
        """Omit the absent server_id for shared-PSK records."""

        omit_none = True


@dataclass
class ManagementAddRecordMessage(ServerMessage):
    """Adds a pairing record to the client."""

    payload: ManagementAddRecordPayload
    type: Literal["management/add-record"] = "management/add-record"


# Server -> Client: management/remove-record
@dataclass
class ManagementRemoveRecordPayload(SendspinModel):
    """Identifies the record to remove."""

    psk_id: str


@dataclass
class ManagementRemoveRecordMessage(ServerMessage):
    """Removes a pairing record from the client."""

    payload: ManagementRemoveRecordPayload
    type: Literal["management/remove-record"] = "management/remove-record"


# Pairing config (shared wire shapes for get/set-pairing-config)
@dataclass
class RecordModeConfig(SendspinModel):
    """The client-wide record mode on the wire."""

    psk_id: str
    """Shared-PSK record used as the storage-exhaustion fallback when pairing."""


# Server -> Client: management/get-pairing-config
@dataclass
class ManagementGetPairingConfigPayload(SendspinModel):
    """Empty ``management/get-pairing-config`` payload."""


@dataclass
class ManagementGetPairingConfigMessage(ServerMessage):
    """Requests the client's pairing configuration (no payload fields)."""

    payload: ManagementGetPairingConfigPayload = field(
        default_factory=ManagementGetPairingConfigPayload
    )
    type: Literal["management/get-pairing-config"] = "management/get-pairing-config"


# Server -> Client: management/set-pairing-config
@dataclass
class SetPairingPskConfig(SendspinModel):
    """Patch for the Pairing PSK method; absent fields are left unchanged."""

    enabled: bool | None = None
    psk: str | None = None
    """43-char base64url 32-byte PSK; replaces the configured Pairing PSK."""

    class Config(SendspinConfig):
        """Absent (omitted) fields mean 'leave unchanged'."""

        omit_none = True


@dataclass
class SetStaticPairingCodeConfig(SendspinModel):
    """Patch for the static-pairing-code method; absent fields are left unchanged."""

    enabled: bool | None = None
    code: str | None = None
    """8 decimal digits; replaces the configured static pairing code."""

    class Config(SendspinConfig):
        """Absent (omitted) fields mean 'leave unchanged'."""

        omit_none = True


@dataclass
class SetDynamicPairingCodeConfig(SendspinModel):
    """Patch for the dynamic pairing-code method; absent fields are left unchanged."""

    enabled: bool | None = None

    class Config(SendspinConfig):
        """Absent (omitted) fields mean 'leave unchanged'."""

        omit_none = True


@dataclass
class SetUnpairedAccessConfig(SendspinModel):
    """Patch for unpaired access; absent fields are left unchanged."""

    enabled: bool | None = None

    class Config(SendspinConfig):
        """Absent (omitted) fields mean 'leave unchanged'."""

        omit_none = True


@dataclass
class ManagementSetPairingConfigPayload(SendspinModel):
    """Partial patch over the client's pairing config; absent method objects are unchanged."""

    pairing_psk: SetPairingPskConfig | None = None
    static_pairing_code: SetStaticPairingCodeConfig | None = None
    dynamic_pairing_code: SetDynamicPairingCodeConfig | None = None
    record_mode: RecordModeConfig | None = None
    unpaired_access: SetUnpairedAccessConfig | None = None

    class Config(SendspinConfig):
        """Absent (omitted) objects mean 'leave unchanged'."""

        omit_none = True


@dataclass
class ManagementSetPairingConfigMessage(ServerMessage):
    """Modifies the client's pairing config as a patch."""

    payload: ManagementSetPairingConfigPayload
    type: Literal["management/set-pairing-config"] = "management/set-pairing-config"


# Server -> Client: management/open-pairing-window
@dataclass
class ManagementOpenPairingWindowPayload(SendspinModel):
    """Empty ``management/open-pairing-window`` payload."""


@dataclass
class ManagementOpenPairingWindowMessage(ServerMessage):
    """Opens a pairing window on the client in place of the operator gesture."""

    payload: ManagementOpenPairingWindowPayload = field(
        default_factory=ManagementOpenPairingWindowPayload
    )
    type: Literal["management/open-pairing-window"] = "management/open-pairing-window"


# Client -> Server: management/result
@dataclass(kw_only=True)
class RecordSummary(SendspinModel):
    """One entry in a list-records result."""

    psk_id: str
    server_id: str | None = None
    """Present for stored-pubkey records, absent for shared-PSK records."""
    used: bool
    """``True`` once a server has authenticated a session with this record's PSK."""

    class Config(SendspinConfig):
        """Omit the absent server_id for shared-PSK records."""

        omit_none = True


@dataclass
class PairingMethodConfig(SendspinModel):
    """A method's config in a get-pairing-config result."""

    enabled: bool
    escalated: bool | None = None
    """For dynamic pairing code only: ``true`` while the round limit holds attempts back."""

    class Config(SendspinConfig):
        """Omit method-specific fields where they do not apply."""

        omit_none = True


@dataclass
class ManagementResultData(SendspinModel):
    """Operation-specific data for a management/result; present only on ``ok``."""

    records: list[RecordSummary] | None = None
    """Present for list-records."""
    pairing_psk: PairingMethodConfig | None = None
    """Present for get-pairing-config."""
    static_pairing_code: PairingMethodConfig | None = None
    """Present for get-pairing-config if the client implements static pairing code."""
    dynamic_pairing_code: PairingMethodConfig | None = None
    """Present for get-pairing-config if the client implements dynamic pairing code."""
    record_mode: RecordModeConfig | None = None
    """Present for get-pairing-config."""
    unpaired_access: UnpairedAccess | None = None
    """Present for get-pairing-config."""

    class Config(SendspinConfig):
        """Omit fields not relevant to the answered request."""

        omit_none = True


@dataclass
class StorageAccounting(SendspinModel):
    """Record-storage accounting on a management/result.

    ``free`` is always present; ``capacity`` and the per-kind costs accompany it only on
    list-records and get-pairing-config results.
    """

    free: int
    capacity: int | None = None
    cost_individual: int | None = None
    cost_shared: int | None = None

    class Config(SendspinConfig):
        """Omit the static fields on results that carry only ``free``."""

        omit_none = True


@dataclass
class ManagementResultPayload(SendspinModel):
    """Result code and optional data for a management request."""

    result: ManagementResult
    data: ManagementResultData | None = None
    storage: StorageAccounting | None = None

    class Config(SendspinConfig):
        """Omit absent optional fields (data, storage)."""

        omit_none = True


@dataclass
class ManagementResultMessage(ClientMessage):
    """Reply to any management/* request."""

    payload: ManagementResultPayload
    type: Literal["management/result"] = "management/result"
