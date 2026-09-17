"""Round-trip and discriminator tests for the management command messages."""
# DEPRECATED(spec-pr-183): remove in aiosendspin <version>

from __future__ import annotations

import orjson
import pytest

from aiosendspin.models.management import (
    ManagementAddRecordMessage,
    ManagementAddRecordPayload,
    ManagementGetPairingConfigMessage,
    ManagementListRecordsMessage,
    ManagementOpenPairingWindowMessage,
    ManagementRemoveRecordMessage,
    ManagementRemoveRecordPayload,
    ManagementResultData,
    ManagementResultMessage,
    ManagementResultPayload,
    ManagementSetPairingConfigMessage,
    ManagementSetPairingConfigPayload,
    PairingMethodConfig,
    RecordModeConfig,
    RecordSummary,
    ServerUnpairMessage,
    SetDynamicPairingCodeConfig,
    SetPairingPskConfig,
    StorageAccounting,
    UnpairedAccess,
)
from aiosendspin.models.types import (
    ClientMessage,
    ManagementResult,
    ServerMessage,
)

_PSK_B64 = "a" * 43


@pytest.mark.parametrize(
    "message",
    [
        ManagementListRecordsMessage(),
        ManagementAddRecordMessage(ManagementAddRecordPayload(psk=_PSK_B64)),
        ManagementAddRecordMessage(ManagementAddRecordPayload(psk=_PSK_B64, server_id="srv1")),
        ManagementRemoveRecordMessage(ManagementRemoveRecordPayload(psk_id="p1")),
        ManagementGetPairingConfigMessage(),
        ManagementSetPairingConfigMessage(
            ManagementSetPairingConfigPayload(
                pairing_psk=SetPairingPskConfig(enabled=False),
                dynamic_pairing_code=SetDynamicPairingCodeConfig(enabled=True),
                record_mode=RecordModeConfig(psk_id="p1"),
            )
        ),
        ManagementOpenPairingWindowMessage(),
    ],
)
def test_server_request_round_trips(message: ServerMessage) -> None:
    """Each server→client request round-trips through the ServerMessage discriminator."""
    decoded = ServerMessage.from_json(message.to_json())
    assert type(decoded) is type(message)
    assert decoded == message


@pytest.mark.parametrize(
    "message",
    [
        ServerUnpairMessage(),
        ManagementListRecordsMessage(),
        ManagementGetPairingConfigMessage(),
        ManagementOpenPairingWindowMessage(),
    ],
)
def test_empty_payload_messages_send_payload_key(message: ServerMessage) -> None:
    """Requests without payload fields still serialize an empty payload object."""
    assert orjson.loads(message.to_json())["payload"] == {}


def test_client_responses_round_trip() -> None:
    """The client→server response type resolves via the ClientMessage discriminator."""
    result = ManagementResultMessage(
        ManagementResultPayload(
            result=ManagementResult.OK,
            data=ManagementResultData(
                records=[
                    RecordSummary(psk_id="p1", used=True, server_id="srv1"),
                    RecordSummary(psk_id="p2", used=False),
                ]
            ),
        )
    )
    assert ClientMessage.from_json(result.to_json()) == result


def test_add_record_omits_absent_server_id() -> None:
    """A shared-PSK add-record (no server_id) omits the field on the wire."""
    message = ManagementAddRecordMessage(ManagementAddRecordPayload(psk=_PSK_B64))
    assert "server_id" not in orjson.loads(message.to_json())["payload"]


def test_result_omits_data_when_absent() -> None:
    """A non-ok result carries no data field."""
    message = ManagementResultMessage(ManagementResultPayload(result=ManagementResult.NOT_FOUND))
    assert "data" not in orjson.loads(message.to_json())["payload"]


def test_record_summary_omits_server_id_for_shared() -> None:
    """A shared-PSK record summary omits server_id."""
    summary = RecordSummary(psk_id="p2", used=False)
    assert "server_id" not in orjson.loads(summary.to_json())


def test_storage_free_only_omits_statics() -> None:
    """A free-only storage object (non-read result) omits capacity and the per-kind costs."""
    message = ManagementResultMessage(
        ManagementResultPayload(result=ManagementResult.OK, storage=StorageAccounting(free=3))
    )
    assert orjson.loads(message.to_json())["payload"]["storage"] == {"free": 3}


def test_storage_full_round_trips() -> None:
    """A full storage object (list-records / get-pairing-config) round-trips with every field."""
    message = ManagementResultMessage(
        ManagementResultPayload(
            result=ManagementResult.OK,
            storage=StorageAccounting(free=2, capacity=8, cost_individual=2, cost_shared=1),
        )
    )
    assert ClientMessage.from_json(message.to_json()) == message


def test_result_omits_storage_when_absent() -> None:
    """A result with no storage accounting omits the field (e.g. permission_denied)."""
    message = ManagementResultMessage(
        ManagementResultPayload(result=ManagementResult.PERMISSION_DENIED)
    )
    assert "storage" not in orjson.loads(message.to_json())["payload"]


def test_set_config_patch_omits_absent_fields() -> None:
    """An absent patch field / method object is omitted on the wire (means 'unchanged')."""
    payload = ManagementSetPairingConfigPayload(pairing_psk=SetPairingPskConfig(enabled=True))
    raw = orjson.loads(ManagementSetPairingConfigMessage(payload).to_json())["payload"]
    assert raw == {"pairing_psk": {"enabled": True}}


def test_get_config_data_round_trips() -> None:
    """A get-pairing-config result round-trips and omits escalated for non-dynamic methods."""
    data = ManagementResultData(
        pairing_psk=PairingMethodConfig(enabled=True),
        dynamic_pairing_code=PairingMethodConfig(enabled=False, escalated=True),
        record_mode=RecordModeConfig(psk_id="p1"),
        unpaired_access=UnpairedAccess(enabled=False),
    )
    message = ManagementResultMessage(
        ManagementResultPayload(result=ManagementResult.OK, data=data)
    )
    assert ClientMessage.from_json(message.to_json()) == message
    raw = orjson.loads(message.to_json())["payload"]["data"]
    assert "escalated" not in raw["pairing_psk"]
    assert raw["dynamic_pairing_code"]["escalated"] is True
