"""Constrained pairing-store fakes shared across the noise/client/integration suites."""

from __future__ import annotations

from datetime import UTC, datetime

from aiosendspin.noise.keys import generate_psk, psk_id_for
from aiosendspin.noise.trust_store import (
    ClientPairingRecord,
    ClientPairingStore,
    InMemoryClientPairingStore,
    StorageReport,
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


# DEPRECATED(spec-pr-183): remove in aiosendspin <version>
class ExhaustedClientStore(InMemoryClientPairingStore):
    """Client store that cannot persist new records (exercises the shared-PSK fallback)."""

    async def can_store_record(self) -> bool:
        """Refuse: there is no capacity for a new record."""
        return False


# DEPRECATED(spec-pr-183): remove in aiosendspin <version>
class BoundedClientStore(InMemoryClientPairingStore):
    """Client store with a fixed four-slot record budget (one slot per record)."""

    async def storage_accounting(self) -> StorageReport:
        """Report a four-slot budget, one slot consumed per stored record."""
        used = len(await self.list_records())
        return StorageReport(capacity=4, free=4 - used, cost_individual=1, cost_shared=1)
