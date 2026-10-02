"""Pairing-store helpers shared across the noise/client/integration suites."""

from __future__ import annotations

from datetime import UTC, datetime

from aiosendspin.noise.keys import generate_psk, psk_id_for
from aiosendspin.noise.trust_store import (
    ClientPairingRecord,
    ClientPairingStore,
)


async def seed_used_client_records(
    store: ClientPairingStore, count: int
) -> list[ClientPairingRecord]:
    """Store ``count`` per-server records for ``server-<i>``, oldest last use first."""
    records = []
    for i in range(count):
        psk = generate_psk()
        record = ClientPairingRecord(
            psk_id=psk_id_for(psk),
            psk=psk,
            server_id=f"server-{i}",
            last_used_at=datetime(2026, 1, 1, i, tzinfo=UTC),
        )
        await store.store_record(record)
        records.append(record)
    return records
