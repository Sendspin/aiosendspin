"""Benchmark how many simultaneous player clients the server can stream to."""

# ruff: noqa

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import math
import multiprocessing as mp
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from aiohttp import ClientSession, TCPConnector

from aiosendspin.client import SendspinClient
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import AudioCodec, PlayerCommand, Roles
from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.push_stream import DEFAULT_INITIAL_DELAY_US, PushStream
from aiosendspin.server.server import SendspinServer

logger = logging.getLogger(__name__)


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark simultaneous player clients streaming with clock sync. "
            "Runs a local server and connects N clients in-process."
        )
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host to bind")
    parser.add_argument("--port", type=int, default=19375, help="Server port to bind")
    parser.add_argument("--clients", type=int, default=10_000, help="Number of clients")
    parser.add_argument("--duration-s", type=int, default=60, help="Stream duration")
    parser.add_argument("--chunk-ms", type=int, default=100, help="PCM chunk size in ms")
    parser.add_argument("--sample-rate", type=int, default=48_000, help="PCM sample rate")
    parser.add_argument("--channels", type=int, default=2, help="PCM channel count")
    parser.add_argument("--bit-depth", type=int, default=16, help="PCM bit depth")
    parser.add_argument(
        "--ramp-rate",
        type=float,
        default=1_000.0,
        help="Client connect rate (clients/sec)",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=1_000,
        help="Max concurrent connect tasks",
    )
    parser.add_argument(
        "--connect-timeout-s",
        type=float,
        default=10.0,
        help="Per-client connect timeout",
    )
    parser.add_argument(
        "--connect-phase-timeout-s",
        type=float,
        default=30.0,
        help="Total timeout for the connect phase (0 to disable)",
    )
    parser.add_argument(
        "--sync-sample-interval-s",
        type=float,
        default=5.0,
        help="How often to sample time sync stats",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker processes (0 = single-process mode)",
    )
    parser.add_argument(
        "--servers",
        type=int,
        default=1,
        help="Number of server instances (each in separate process when > 1)",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=19375,
        help="Starting port for multi-server mode",
    )
    parser.add_argument(
        "--clients-per-worker",
        type=int,
        default=100,
        help="Clients per worker process in distributed mode",
    )
    parser.add_argument(
        "--codec",
        choices=["pcm", "opus", "flac"],
        default="pcm",
        help="Audio codec for clients to request",
    )
    parser.add_argument(
        "--opus-bitrate",
        type=int,
        default=32000,
        help="Opus bitrate in bps (e.g., 6000 for 6kbps, 32000 for 32kbps)",
    )
    return parser.parse_args(argv)


def _make_player_support(
    sample_rate: int,
    channels: int,
    bit_depth: int,
    codec: str = "pcm",
    opus_bitrate: int = 32000,
) -> ClientHelloPlayerSupport:
    codec_enum = {
        "pcm": AudioCodec.PCM,
        "opus": AudioCodec.OPUS,
        "flac": AudioCodec.FLAC,
    }[codec]

    fmt = SupportedAudioFormat(
        codec=codec_enum,
        channels=channels,
        sample_rate=sample_rate,
        bit_depth=bit_depth,
    )

    # Add codec-specific options
    if codec == "opus":
        fmt.options = {"bitrate": str(opus_bitrate)}

    return ClientHelloPlayerSupport(
        supported_formats=[fmt],
        buffer_capacity=512 * 1024,
        supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE],
    )


def _make_audio_format(sample_rate: int, channels: int, bit_depth: int) -> AudioFormat:
    return AudioFormat(sample_rate=sample_rate, bit_depth=bit_depth, channels=channels)


def _make_silence_chunk(fmt: AudioFormat, chunk_ms: int) -> bytes:
    frames = int(fmt.sample_rate * (chunk_ms / 1000.0))
    frame_bytes = fmt.channels * (fmt.bit_depth // 8)
    return b"\x00" * (frames * frame_bytes)


@dataclass(slots=True)
class AudioStats:
    chunks: int = 0
    bytes: int = 0
    empty_chunks: int = 0
    bad_sizes: int = 0
    out_of_order: int = 0
    gaps: int = 0
    gap_us_total: int = 0
    gap_us_max: int = 0
    last_ts_us: int | None = None
    last_duration_us: int | None = None
    first_ts_us: int | None = None


@dataclass(slots=True)
class SyncStats:
    synchronized: int = 0
    total: int = 0
    error_us: list[int] | None = None
    offset_us: list[int] | None = None


# --- IPC message types for multiprocess mode ---


@dataclass
class WorkerReady:
    worker_id: int
    connected: int
    failed: int


@dataclass
class WorkerStats:
    worker_id: int
    audio_stats_json: dict[str, dict[str, Any]]
    sync_stats_json: dict[str, Any]


@dataclass
class WorkerError:
    worker_id: int
    error: str


# --- IPC message types for server processes (distributed mode) ---


@dataclass
class ServerListening:
    """Sent when server is listening and ready to accept connections."""

    server_id: int
    port: int


@dataclass
class ServerReady:
    """Sent when server has clients connected and is ready to stream."""

    server_id: int
    port: int
    client_count: int


@dataclass
class ServerCheckClients:
    """Command for server to count clients and report ready."""

    pass


@dataclass
class ServerStart:
    initial_play_start_us: int


@dataclass
class ServerDone:
    server_id: int
    chunks_sent: int


@dataclass
class ServerError:
    server_id: int
    error: str


def _audio_stats_to_dict(stats: AudioStats) -> dict[str, Any]:
    """Convert AudioStats to dict for serialization (slots=True blocks pickle)."""
    return {
        "chunks": stats.chunks,
        "bytes": stats.bytes,
        "empty_chunks": stats.empty_chunks,
        "bad_sizes": stats.bad_sizes,
        "out_of_order": stats.out_of_order,
        "gaps": stats.gaps,
        "gap_us_total": stats.gap_us_total,
        "gap_us_max": stats.gap_us_max,
        "last_ts_us": stats.last_ts_us,
        "last_duration_us": stats.last_duration_us,
        "first_ts_us": stats.first_ts_us,
    }


def _dict_to_audio_stats(d: dict[str, Any]) -> AudioStats:
    """Convert dict back to AudioStats."""
    stats = AudioStats()
    stats.chunks = d["chunks"]
    stats.bytes = d["bytes"]
    stats.empty_chunks = d["empty_chunks"]
    stats.bad_sizes = d["bad_sizes"]
    stats.out_of_order = d["out_of_order"]
    stats.gaps = d["gaps"]
    stats.gap_us_total = d["gap_us_total"]
    stats.gap_us_max = d["gap_us_max"]
    stats.last_ts_us = d["last_ts_us"]
    stats.last_duration_us = d["last_duration_us"]
    stats.first_ts_us = d.get("first_ts_us")
    return stats


def _sync_stats_to_dict(stats: SyncStats) -> dict[str, Any]:
    """Convert SyncStats to dict for serialization."""
    return {
        "synchronized": stats.synchronized,
        "total": stats.total,
        "error_us": stats.error_us,
        "offset_us": stats.offset_us,
    }


def _dict_to_sync_stats(d: dict[str, Any]) -> SyncStats:
    """Convert dict back to SyncStats."""
    return SyncStats(
        synchronized=d["synchronized"],
        total=d["total"],
        error_us=d["error_us"],
        offset_us=d["offset_us"],
    )


def _distribute_clients(total: int, workers: int) -> list[tuple[int, int]]:
    """Distribute clients across workers. Returns list of (start_index, count)."""
    if workers <= 0:
        return [(0, total)]
    base_count = total // workers
    remainder = total % workers
    result: list[tuple[int, int]] = []
    start = 0
    for i in range(workers):
        count = base_count + (1 if i < remainder else 0)
        result.append((start, count))
        start += count
    return result


def _summarize(values: list[int]) -> str:
    if not values:
        return "n=0"
    values_sorted = sorted(values)
    n = len(values_sorted)
    p50 = values_sorted[n // 2]
    p95 = values_sorted[min(n - 1, math.ceil(n * 0.95) - 1)]
    p99 = values_sorted[min(n - 1, math.ceil(n * 0.99) - 1)]
    vmin = values_sorted[0]
    vmax = values_sorted[-1]
    return f"n={n} min={vmin} p50={p50} p95={p95} p99={p99} max={vmax}"


async def _connect_clients(
    *,
    url: str | list[str],
    count: int,
    ramp_rate: float,
    max_in_flight: int,
    connect_timeout_s: float,
    connect_phase_timeout_s: float,
    session: ClientSession,
    sample_rate: int,
    channels: int,
    bit_depth: int,
    start_index: int = 0,
    codec: str = "pcm",
    opus_bitrate: int = 32000,
) -> tuple[list[SendspinClient], dict[str, AudioStats]]:
    # Normalize urls to a list for round-robin distribution
    urls = [url] if isinstance(url, str) else url

    player_support = _make_player_support(sample_rate, channels, bit_depth, codec, opus_bitrate)

    semaphore = asyncio.Semaphore(max_in_flight)
    tasks: list[asyncio.Task[SendspinClient | None]] = []
    connected_count = 0

    audio_stats: dict[str, AudioStats] = {}

    async def _connect_one(index: int) -> SendspinClient | None:
        nonlocal connected_count
        client_id = f"bench-{start_index + index:05d}"
        # Round-robin across URLs based on global client index
        target_url = urls[(start_index + index) % len(urls)]
        client = SendspinClient(
            client_id=client_id,
            client_name=client_id,
            roles=[Roles.PLAYER],
            player_support=player_support,
            session=session,
        )
        stats = AudioStats()
        audio_stats[client_id] = stats

        def _on_audio_chunk(
            timestamp_us: int, payload: bytes, fmt: object, _send_ahead: int
        ) -> None:
            stats.chunks += 1
            stats.bytes += len(payload)
            if stats.first_ts_us is None:
                stats.first_ts_us = timestamp_us
            if not payload:
                stats.empty_chunks += 1
                return
            fmt_channels = getattr(fmt, "channels", channels)
            fmt_bit_depth = getattr(fmt, "bit_depth", bit_depth)
            fmt_sample_rate = getattr(fmt, "sample_rate", sample_rate)
            frame_size = fmt_channels * (fmt_bit_depth // 8)
            if frame_size == 0 or len(payload) % frame_size != 0:
                stats.bad_sizes += 1
                return

            frames = len(payload) // frame_size
            duration_us = round(frames * 1_000_000 / fmt_sample_rate)
            if stats.last_ts_us is not None:
                if timestamp_us <= stats.last_ts_us:
                    stats.out_of_order += 1
                elif stats.last_duration_us is not None:
                    expected = stats.last_ts_us + stats.last_duration_us
                    gap = timestamp_us - expected
                    if gap > 2_000:  # allow 2ms jitter
                        stats.gaps += 1
                        stats.gap_us_total += gap
                        stats.gap_us_max = max(stats.gap_us_max, gap)
            stats.last_ts_us = timestamp_us
            stats.last_duration_us = duration_us

        client.add_audio_chunk_listener(_on_audio_chunk)
        async with semaphore:
            try:
                await asyncio.wait_for(client.connect(target_url), timeout=connect_timeout_s)
                connected_count += 1
                if connected_count % 100 == 0:
                    logger.warning("Connected %d/%d clients...", connected_count, count)
                return client
            except Exception:
                logger.exception("Failed to connect client %s", client_id)
                with contextlib.suppress(Exception):
                    if client.connected:
                        await client.disconnect()
                return None

    ramp_interval = 0.0 if ramp_rate <= 0 else 1.0 / ramp_rate
    for i in range(count):
        tasks.append(asyncio.create_task(_connect_one(i)))
        if ramp_interval:
            await asyncio.sleep(ramp_interval)

    if connect_phase_timeout_s > 0:
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=connect_phase_timeout_s,
            )
        except TimeoutError:
            for task in tasks:
                task.cancel()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            logger.warning(
                "Connect phase timed out after %.1fs (continuing with connected clients)",
                connect_phase_timeout_s,
            )
    else:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    clients: list[SendspinClient] = []
    failures = 0
    for result in results:
        if isinstance(result, SendspinClient):
            clients.append(result)
        else:
            failures += 1

    logger.warning("Connected %d/%d clients (%d failures)", len(clients), count, failures)
    return clients, audio_stats


async def _group_clients(server: SendspinServer, clients: list[SendspinClient]) -> None:
    if not clients:
        return

    leader_id = clients[0]._client_id
    leader = server.get_client(leader_id)
    if leader is None:
        raise RuntimeError("Leader client not found on server")
    group = leader.group

    for client in clients[1:]:
        server_client = server.get_client(client._client_id)
        if server_client is None:
            continue
        await group.add_client(server_client)


def _collect_sync_stats(clients: list[SendspinClient]) -> SyncStats:
    stats = SyncStats(synchronized=0, total=len(clients), error_us=[], offset_us=[])
    for client in clients:
        if not client.is_time_synchronized():
            continue
        stats.synchronized += 1
        # Access internal filter for accuracy stats (benchmark-only).
        filt = client._time_filter  # noqa: SLF001
        stats.error_us.append(filt.error)
        stats.offset_us.append(abs(int(filt.offset)))
    return stats


def _log_sync_stats(stats: SyncStats) -> None:
    prefix = "ALL GOOD"
    if stats.total == 0:
        logger.warning("%s time sync: n=0", prefix)
        return
    if stats.synchronized != stats.total:
        prefix = "CHECK"
    ratio = f"{stats.synchronized}/{stats.total}"
    err_summary = _summarize(stats.error_us or [])
    off_summary = _summarize(stats.offset_us or [])
    logger.warning(
        "%s time sync: %s | error_us(%s) offset_us(%s)",
        prefix,
        ratio,
        err_summary,
        off_summary,
    )


async def _run_stream(
    *,
    server: SendspinServer,
    clients: list[SendspinClient],
    audio_stats: dict[str, AudioStats],
    duration_s: int,
    chunk_ms: int,
    sample_rate: int,
    channels: int,
    bit_depth: int,
) -> None:
    if not clients:
        return

    leader = server.get_client(clients[0]._client_id)
    if leader is None:
        raise RuntimeError("Leader client not found on server")

    group = leader.group
    stream = group.start_stream()
    fmt = _make_audio_format(sample_rate, channels, bit_depth)
    chunk = _make_silence_chunk(fmt, chunk_ms)

    start = time.monotonic()
    next_tick = start
    chunks_sent = 0

    logger.warning(
        "ALL GOOD streaming: starting %ds with %dms chunks @ %dHz %dch %dbit",
        duration_s,
        chunk_ms,
        sample_rate,
        channels,
        bit_depth,
    )
    sampled_10s = False
    while True:
        now = time.monotonic()
        if now - start >= duration_s:
            break

        stream.prepare_audio(chunk, fmt)
        await stream.commit_audio()
        chunks_sent += 1

        if not sampled_10s and now - start >= 10.0:
            _log_sync_stats(_collect_sync_stats(clients))
            sampled_10s = True

        next_tick += chunk_ms / 1000.0
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)

    stream.stop()
    audio_played_s = chunks_sent * chunk_ms / 1000.0
    logger.warning("Streamed %.1fs of audio (%d chunks)", audio_played_s, chunks_sent)
    _log_sync_stats(_collect_sync_stats(clients))
    # Server splits into 25ms frames, so expected frames = chunks * (chunk_ms / 25)
    frame_duration_ms = 25
    expected_frames = chunks_sent * (chunk_ms // frame_duration_ms)
    _log_audio_stats(audio_stats, expected_frames)


def _log_audio_stats(audio_stats: dict[str, AudioStats], expected_chunks: int) -> None:
    if not audio_stats:
        logger.warning("CHECK streaming: audio stats n=0")
        return

    totals = AudioStats()
    complete_clients = 0
    incomplete_clients = 0
    min_chunks = expected_chunks
    max_chunks = 0
    anomalous_clients: list[tuple[str, int]] = []

    for client_id, stats in audio_stats.items():
        totals.chunks += stats.chunks
        totals.bytes += stats.bytes
        totals.empty_chunks += stats.empty_chunks
        totals.bad_sizes += stats.bad_sizes
        totals.out_of_order += stats.out_of_order
        totals.gaps += stats.gaps
        totals.gap_us_total += stats.gap_us_total
        totals.gap_us_max = max(totals.gap_us_max, stats.gap_us_max)

        min_chunks = min(min_chunks, stats.chunks)
        max_chunks = max(max_chunks, stats.chunks)
        if stats.chunks >= expected_chunks:
            complete_clients += 1
        else:
            incomplete_clients += 1

        # Track clients with unexpected chunk counts
        if stats.chunks != expected_chunks:
            anomalous_clients.append((client_id, stats.chunks))

    avg_gap = 0
    if totals.gaps:
        avg_gap = totals.gap_us_total // totals.gaps

    prefix = "ALL GOOD"
    if (
        incomplete_clients
        or totals.empty_chunks
        or totals.bad_sizes
        or totals.out_of_order
        or totals.gaps
    ):
        prefix = "CHECK"

    logger.warning(
        "%s streaming: %d/%d clients received all %d chunks (min=%d max=%d)",
        prefix,
        complete_clients,
        len(audio_stats),
        expected_chunks,
        min_chunks,
        max_chunks,
    )
    logger.warning(
        "%s streaming: audio stats chunks=%d bytes=%d empty=%d bad_sizes=%d "
        "out_of_order=%d gaps=%d gap_us_avg=%d gap_us_max=%d",
        prefix,
        totals.chunks,
        totals.bytes,
        totals.empty_chunks,
        totals.bad_sizes,
        totals.out_of_order,
        totals.gaps,
        avg_gap,
        totals.gap_us_max,
    )

    if anomalous_clients:
        # Show first 5 anomalous clients for debugging
        sample = anomalous_clients[:5]
        logger.warning(
            "CHECK: %d clients have unexpected chunk counts (showing %d): %s",
            len(anomalous_clients),
            len(sample),
            ", ".join(f"{cid}={chunks}" for cid, chunks in sample),
        )


def _verify_cross_server_sync(
    all_audio_stats: dict[str, AudioStats],
    server_count: int,
) -> None:
    """Verify all clients received the same first timestamp (cross-server sync)."""
    ts_values = [s.first_ts_us for s in all_audio_stats.values() if s.first_ts_us is not None]
    if not ts_values:
        logger.warning("CHECK sync: no first_ts_us recorded")
        return

    spread_us = max(ts_values) - min(ts_values)
    if spread_us == 0:
        logger.warning(
            "ALL GOOD sync: %d clients across %d servers received same first_ts=%d",
            len(ts_values),
            server_count,
            ts_values[0],
        )
    else:
        logger.warning(
            "CHECK sync: spread=%dus across %d clients on %d servers (min=%d max=%d)",
            spread_us,
            len(ts_values),
            server_count,
            min(ts_values),
            max(ts_values),
        )


# --- Multiprocess worker functions ---


def _worker_main(
    worker_id: int,
    urls: str | list[str],
    start_index: int,
    client_count: int,
    ramp_rate: float,
    max_in_flight: int,
    connect_timeout_s: float,
    connect_phase_timeout_s: float,
    sample_rate: int,
    channels: int,
    bit_depth: int,
    message_queue: mp.Queue,  # type: ignore[type-arg]
    stop_event: mp.Event,  # type: ignore[type-arg]
    log_level: str,
    codec: str = "pcm",
    opus_bitrate: int = 32000,
) -> None:
    """Worker process entry point."""
    asyncio.run(
        _worker_async(
            worker_id=worker_id,
            urls=urls,
            start_index=start_index,
            client_count=client_count,
            ramp_rate=ramp_rate,
            max_in_flight=max_in_flight,
            connect_timeout_s=connect_timeout_s,
            connect_phase_timeout_s=connect_phase_timeout_s,
            sample_rate=sample_rate,
            channels=channels,
            bit_depth=bit_depth,
            message_queue=message_queue,
            stop_event=stop_event,
            log_level=log_level,
            codec=codec,
            opus_bitrate=opus_bitrate,
        )
    )


async def _worker_async(
    *,
    worker_id: int,
    urls: str | list[str],
    start_index: int,
    client_count: int,
    ramp_rate: float,
    max_in_flight: int,
    connect_timeout_s: float,
    connect_phase_timeout_s: float,
    sample_rate: int,
    channels: int,
    bit_depth: int,
    message_queue: mp.Queue,  # type: ignore[type-arg]
    stop_event: mp.Event,  # type: ignore[type-arg]
    log_level: str,
    codec: str = "pcm",
    opus_bitrate: int = 32000,
) -> None:
    """Worker process async logic."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=f"%(asctime)s %(levelname)s [W{worker_id}] %(message)s",
    )
    wlogger = logging.getLogger(f"worker-{worker_id}")

    connector = TCPConnector(limit=0, limit_per_host=0, ttl_dns_cache=0)
    session = ClientSession(connector=connector)
    clients: list[SendspinClient] = []
    audio_stats: dict[str, AudioStats] = {}

    try:
        url_list = [urls] if isinstance(urls, str) else urls
        wlogger.warning(
            "Connecting %d clients (start_index=%d) to %d URLs",
            client_count,
            start_index,
            len(url_list),
        )
        clients, audio_stats = await _connect_clients(
            url=urls,
            count=client_count,
            ramp_rate=ramp_rate,
            max_in_flight=max_in_flight,
            connect_timeout_s=connect_timeout_s,
            connect_phase_timeout_s=connect_phase_timeout_s,
            session=session,
            sample_rate=sample_rate,
            channels=channels,
            bit_depth=bit_depth,
            start_index=start_index,
            codec=codec,
            opus_bitrate=opus_bitrate,
        )

        connected = len(clients)
        failed = client_count - connected
        message_queue.put(WorkerReady(worker_id=worker_id, connected=connected, failed=failed))
        wlogger.warning("Connected %d/%d clients, waiting for stop signal", connected, client_count)

        # Poll stop_event until set
        loop = asyncio.get_running_loop()
        while not stop_event.is_set():
            await asyncio.sleep(0.1)

        wlogger.warning("Stop signal received, collecting stats")
        sync_stats = _collect_sync_stats(clients)
        audio_stats_json = {cid: _audio_stats_to_dict(s) for cid, s in audio_stats.items()}
        sync_stats_json = _sync_stats_to_dict(sync_stats)
        message_queue.put(
            WorkerStats(
                worker_id=worker_id,
                audio_stats_json=audio_stats_json,
                sync_stats_json=sync_stats_json,
            )
        )
    except Exception as e:
        wlogger.exception("Worker error")
        message_queue.put(WorkerError(worker_id=worker_id, error=str(e)))
    finally:
        for client in clients:
            with contextlib.suppress(Exception):
                await client.disconnect()
        with contextlib.suppress(Exception):
            await session.close()


async def _run_stream_server_only(
    *,
    server: SendspinServer,
    duration_s: int,
    chunk_ms: int,
    sample_rate: int,
    channels: int,
    bit_depth: int,
) -> int:
    """Stream audio using server-side client references only (for multiprocess mode)."""
    connected = list(server.connected_clients)
    if not connected:
        logger.warning("No connected clients on server")
        return 0

    leader = connected[0]
    group = leader.group

    # Add all other clients to the same group
    for client in connected[1:]:
        if client.group != group:
            await group.add_client(client)

    stream = group.start_stream()
    fmt = _make_audio_format(sample_rate, channels, bit_depth)
    chunk = _make_silence_chunk(fmt, chunk_ms)

    start = time.monotonic()
    next_tick = start
    chunks_sent = 0

    logger.warning(
        "ALL GOOD streaming: starting %ds with %dms chunks @ %dHz %dch %dbit (%d clients)",
        duration_s,
        chunk_ms,
        sample_rate,
        channels,
        bit_depth,
        len(connected),
    )

    while True:
        now = time.monotonic()
        if now - start >= duration_s:
            break

        stream.prepare_audio(chunk, fmt)
        await stream.commit_audio()
        chunks_sent += 1

        next_tick += chunk_ms / 1000.0
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)

    stream.stop()
    audio_played_s = chunks_sent * chunk_ms / 1000.0
    logger.warning("Streamed %.1fs of audio (%d chunks)", audio_played_s, chunks_sent)
    return chunks_sent


async def _run_stream_multi_server(
    *,
    servers: list[SendspinServer],
    streams: list[PushStream],
    clock: LoopClock,
    duration_s: int,
    chunk_ms: int,
    sample_rate: int,
    channels: int,
    bit_depth: int,
) -> int:
    """Stream audio to multiple servers with synchronized timestamps."""
    total_clients = sum(len(list(s.connected_clients)) for s in servers)
    if total_clients == 0:
        logger.warning("No connected clients across any server")
        return 0

    fmt = _make_audio_format(sample_rate, channels, bit_depth)
    chunk = _make_silence_chunk(fmt, chunk_ms)

    # Calculate chunk duration for timestamp advancement
    bytes_per_sample = bit_depth // 8
    frame_stride = bytes_per_sample * channels
    sample_count = len(chunk) // frame_stride
    chunk_duration_us = int(sample_count * 1_000_000 / sample_rate)

    # Calculate first timestamp ONCE at stream start
    play_start_us = clock.now_us() + DEFAULT_INITIAL_DELAY_US

    start = time.monotonic()
    next_tick = start
    chunks_sent = 0

    logger.warning(
        "ALL GOOD streaming: starting %ds with %dms chunks @ %dHz %dch %dbit "
        "(%d clients across %d servers)",
        duration_s,
        chunk_ms,
        sample_rate,
        channels,
        bit_depth,
        total_clients,
        len(servers),
    )

    while True:
        now = time.monotonic()
        if now - start >= duration_s:
            break

        # Send to ALL servers with SAME timestamp
        for stream in streams:
            stream.prepare_audio(chunk, fmt)
            await stream.commit_audio(play_start_us=play_start_us)

        # Advance timestamp for next chunk
        play_start_us += chunk_duration_us
        chunks_sent += 1

        next_tick += chunk_ms / 1000.0
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)

    for stream in streams:
        stream.stop()

    audio_played_s = chunks_sent * chunk_ms / 1000.0
    logger.warning("Streamed %.1fs of audio (%d chunks)", audio_played_s, chunks_sent)
    return chunks_sent


def _aggregate_audio_stats(
    all_audio_stats: dict[str, AudioStats],
    expected_chunks: int,
) -> None:
    """Log aggregated audio stats from all workers."""
    _log_audio_stats(all_audio_stats, expected_chunks)


def _aggregate_sync_stats(all_sync_stats: list[SyncStats]) -> None:
    """Log aggregated sync stats from all workers."""
    combined = SyncStats(synchronized=0, total=0, error_us=[], offset_us=[])
    for stats in all_sync_stats:
        combined.synchronized += stats.synchronized
        combined.total += stats.total
        if stats.error_us:
            combined.error_us.extend(stats.error_us)
        if stats.offset_us:
            combined.offset_us.extend(stats.offset_us)
    _log_sync_stats(combined)


async def _main_async_multiprocess(args: argparse.Namespace) -> int:
    """Run benchmark with multiple worker processes."""
    loop = asyncio.get_running_loop()
    server = SendspinServer(loop=loop, server_id="bench", server_name="bench")
    started = False
    processes: list[mp.Process] = []

    # Use spawn context to avoid fork issues with asyncio
    ctx = mp.get_context("spawn")
    message_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    stop_event: mp.Event = ctx.Event()  # type: ignore[type-arg]

    try:
        await server.start_server(
            port=args.port,
            host=args.host,
            advertise_addresses=[],
            discover_clients=False,
        )
        started = True

        url = f"ws://{args.host}:{args.port}{server.API_PATH}"
        distribution = _distribute_clients(args.clients, args.workers)

        logger.warning(
            "Starting %d worker processes to connect %d clients to %s",
            args.workers,
            args.clients,
            url,
        )

        # Spawn worker processes
        for worker_id, (start_index, client_count) in enumerate(distribution):
            if client_count == 0:
                continue
            p = ctx.Process(
                target=_worker_main,
                args=(
                    worker_id,
                    url,
                    start_index,
                    client_count,
                    args.ramp_rate,
                    args.max_in_flight,
                    args.connect_timeout_s,
                    args.connect_phase_timeout_s,
                    args.sample_rate,
                    args.channels,
                    args.bit_depth,
                    message_queue,
                    stop_event,
                    args.log_level,
                    args.codec,
                    args.opus_bitrate,
                ),
            )
            p.start()
            processes.append(p)

        # Wait for all WorkerReady messages
        ready_count = 0
        total_connected = 0
        total_failed = 0
        expected_ready = len(processes)

        logger.warning("Waiting for %d workers to connect clients...", expected_ready)
        while ready_count < expected_ready:
            msg = await loop.run_in_executor(None, message_queue.get)
            if isinstance(msg, WorkerReady):
                ready_count += 1
                total_connected += msg.connected
                total_failed += msg.failed
                logger.warning(
                    "Worker %d ready: %d connected, %d failed (%d/%d workers ready)",
                    msg.worker_id,
                    msg.connected,
                    msg.failed,
                    ready_count,
                    expected_ready,
                )
            elif isinstance(msg, WorkerError):
                logger.error("Worker %d error: %s", msg.worker_id, msg.error)
                # Still count as ready to avoid hanging
                ready_count += 1

        logger.warning(
            "All workers ready: %d connected, %d failed",
            total_connected,
            total_failed,
        )

        # Stream audio using server-side references
        chunks_sent = await _run_stream_server_only(
            server=server,
            duration_s=args.duration_s,
            chunk_ms=args.chunk_ms,
            sample_rate=args.sample_rate,
            channels=args.channels,
            bit_depth=args.bit_depth,
        )

        # Signal workers to stop and collect stats
        logger.warning("Signaling workers to stop...")
        stop_event.set()

        # Collect WorkerStats from all workers
        all_audio_stats: dict[str, AudioStats] = {}
        all_sync_stats: list[SyncStats] = []
        stats_collected = 0

        while stats_collected < expected_ready:
            try:
                msg = await asyncio.wait_for(
                    loop.run_in_executor(None, message_queue.get),
                    timeout=10.0,
                )
                if isinstance(msg, WorkerStats):
                    stats_collected += 1
                    for cid, d in msg.audio_stats_json.items():
                        all_audio_stats[cid] = _dict_to_audio_stats(d)
                    all_sync_stats.append(_dict_to_sync_stats(msg.sync_stats_json))
                    logger.warning(
                        "Worker %d stats collected (%d/%d)",
                        msg.worker_id,
                        stats_collected,
                        expected_ready,
                    )
                elif isinstance(msg, WorkerError):
                    stats_collected += 1
                    logger.error("Worker %d error during stats: %s", msg.worker_id, msg.error)
            except TimeoutError:
                logger.warning("Timeout waiting for worker stats, continuing...")
                break

        # Log aggregated stats
        frame_duration_ms = 25
        expected_frames = chunks_sent * (args.chunk_ms // frame_duration_ms)
        _aggregate_sync_stats(all_sync_stats)
        _aggregate_audio_stats(all_audio_stats, expected_frames)

    finally:
        # Cleanup: terminate any hanging processes
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=5.0)

        if started:
            with contextlib.suppress(Exception):
                await server.close()
        else:
            with contextlib.suppress(Exception):
                if server._owns_session and not server._client_session.closed:  # noqa: SLF001
                    await server._client_session.close()  # noqa: SLF001

    return 0


# --- Distributed mode: separate process per server and per ~100 clients ---


def _server_process_main(
    server_id: int,
    port: int,
    host: str,
    command_queue: mp.Queue,  # type: ignore[type-arg]
    status_queue: mp.Queue,  # type: ignore[type-arg]
    duration_s: int,
    chunk_ms: int,
    sample_rate: int,
    channels: int,
    bit_depth: int,
    log_level: str,
) -> None:
    """Server process entry point."""
    asyncio.run(
        _server_process_async(
            server_id=server_id,
            port=port,
            host=host,
            command_queue=command_queue,
            status_queue=status_queue,
            duration_s=duration_s,
            chunk_ms=chunk_ms,
            sample_rate=sample_rate,
            channels=channels,
            bit_depth=bit_depth,
            log_level=log_level,
        )
    )


async def _server_process_async(
    *,
    server_id: int,
    port: int,
    host: str,
    command_queue: mp.Queue,  # type: ignore[type-arg]
    status_queue: mp.Queue,  # type: ignore[type-arg]
    duration_s: int,
    chunk_ms: int,
    sample_rate: int,
    channels: int,
    bit_depth: int,
    log_level: str,
) -> None:
    """Server process async logic."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=f"%(asctime)s %(levelname)s [S{server_id}] %(message)s",
    )
    slogger = logging.getLogger(f"server-{server_id}")

    loop = asyncio.get_running_loop()
    server = SendspinServer(
        loop=loop, server_id=f"bench-{server_id}", server_name=f"bench-{server_id}"
    )

    try:
        await server.start_server(
            port=port,
            host=host,
            advertise_addresses=[],
            discover_clients=False,
        )
        slogger.warning("Server listening on port %d", port)

        # Signal that we're listening and ready to accept connections
        status_queue.put(ServerListening(server_id=server_id, port=port))

        # Wait for check-clients command (coordinator sends this after all clients connected)
        cmd = await loop.run_in_executor(None, command_queue.get)
        if not isinstance(cmd, ServerCheckClients):
            status_queue.put(
                ServerError(server_id=server_id, error=f"Expected ServerCheckClients, got: {cmd}")
            )
            return

        connected = list(server.connected_clients)
        if not connected:
            status_queue.put(ServerError(server_id=server_id, error="No clients connected"))
            return

        # Group all clients
        leader = connected[0]
        group = leader.group
        for client in connected[1:]:
            if client.group != group:
                await group.add_client(client)

        stream = group.start_stream()
        slogger.warning("Server ready with %d clients", len(connected))
        status_queue.put(ServerReady(server_id=server_id, port=port, client_count=len(connected)))

        # Wait for start command
        cmd = await loop.run_in_executor(None, command_queue.get)
        if not isinstance(cmd, ServerStart):
            status_queue.put(
                ServerError(server_id=server_id, error=f"Expected ServerStart, got: {cmd}")
            )
            return

        # Stream with synchronized timestamps
        fmt = _make_audio_format(sample_rate, channels, bit_depth)
        chunk = _make_silence_chunk(fmt, chunk_ms)

        # Calculate chunk duration
        bytes_per_sample = bit_depth // 8
        frame_stride = bytes_per_sample * channels
        sample_count = len(chunk) // frame_stride
        chunk_duration_us = int(sample_count * 1_000_000 / sample_rate)

        play_start_us = cmd.initial_play_start_us
        start = time.monotonic()
        next_tick = start
        chunks_sent = 0

        slogger.warning("Starting stream with initial_play_start_us=%d", play_start_us)

        while True:
            now = time.monotonic()
            if now - start >= duration_s:
                break

            stream.prepare_audio(chunk, fmt)
            await stream.commit_audio(play_start_us=play_start_us)

            play_start_us += chunk_duration_us
            chunks_sent += 1

            next_tick += chunk_ms / 1000.0
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

        stream.stop()
        slogger.warning("Streaming complete: %d chunks sent", chunks_sent)
        status_queue.put(ServerDone(server_id=server_id, chunks_sent=chunks_sent))

    except Exception as e:
        slogger.exception("Server process error")
        status_queue.put(ServerError(server_id=server_id, error=str(e)))
    finally:
        with contextlib.suppress(Exception):
            await server.close()


async def _main_async_distributed(args: argparse.Namespace) -> int:
    """Run benchmark with separate processes for servers and client workers."""
    ctx = mp.get_context("spawn")

    # Queues for server processes
    server_command_queues: list[mp.Queue] = []  # type: ignore[type-arg]
    server_status_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]

    # Queues/events for client workers
    client_message_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    client_stop_event: mp.Event = ctx.Event()  # type: ignore[type-arg]

    server_processes: list[mp.Process] = []
    client_processes: list[mp.Process] = []

    loop = asyncio.get_running_loop()

    try:
        # Calculate distribution
        num_client_workers = (args.clients + args.clients_per_worker - 1) // args.clients_per_worker
        clients_per_server = args.clients // args.servers
        remainder = args.clients % args.servers

        logger.warning(
            "Distributed mode: %d server processes, %d client worker processes (%d clients/worker)",
            args.servers,
            num_client_workers,
            args.clients_per_worker,
        )

        # Start server processes
        for i in range(args.servers):
            cmd_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
            server_command_queues.append(cmd_queue)
            p = ctx.Process(
                target=_server_process_main,
                args=(
                    i,  # server_id
                    args.base_port + i,  # port
                    args.host,
                    cmd_queue,
                    server_status_queue,
                    args.duration_s,
                    args.chunk_ms,
                    args.sample_rate,
                    args.channels,
                    args.bit_depth,
                    args.log_level,
                ),
            )
            p.start()
            server_processes.append(p)

        # Build URLs for round-robin distribution
        urls = [
            f"ws://{args.host}:{args.base_port + i}{SendspinServer.API_PATH}"
            for i in range(args.servers)
        ]

        # Wait for all servers to be listening before starting clients
        servers_listening = 0
        logger.warning("Waiting for %d servers to start listening...", args.servers)
        while servers_listening < args.servers:
            msg = await loop.run_in_executor(None, server_status_queue.get)
            if isinstance(msg, ServerListening):
                servers_listening += 1
                logger.warning(
                    "Server %d listening on port %d (%d/%d)",
                    msg.server_id,
                    msg.port,
                    servers_listening,
                    args.servers,
                )
            elif isinstance(msg, ServerError):
                logger.error("Server %d error: %s", msg.server_id, msg.error)
                servers_listening += 1

        logger.warning("All %d servers listening, starting client workers", args.servers)

        # Start client worker processes (now that servers are listening)
        client_start_index = 0
        for worker_id in range(num_client_workers):
            worker_client_count = min(args.clients_per_worker, args.clients - client_start_index)
            if worker_client_count <= 0:
                break

            # Pass all URLs - clients within worker are distributed round-robin
            p = ctx.Process(
                target=_worker_main,
                args=(
                    worker_id,
                    urls,  # All URLs for round-robin distribution
                    client_start_index,
                    worker_client_count,
                    args.ramp_rate,
                    args.max_in_flight,
                    args.connect_timeout_s,
                    args.connect_phase_timeout_s,
                    args.sample_rate,
                    args.channels,
                    args.bit_depth,
                    client_message_queue,
                    client_stop_event,
                    args.log_level,
                    args.codec,
                    args.opus_bitrate,
                ),
            )
            p.start()
            client_processes.append(p)
            client_start_index += worker_client_count

        logger.warning(
            "Started %d client worker processes",
            len(client_processes),
        )

        # Wait for all client workers to connect first
        workers_ready = 0
        total_connected = 0
        total_failed = 0
        while workers_ready < len(client_processes):
            msg = await loop.run_in_executor(None, client_message_queue.get)
            if isinstance(msg, WorkerReady):
                workers_ready += 1
                total_connected += msg.connected
                total_failed += msg.failed
                logger.warning(
                    "Worker %d ready: %d connected (%d/%d workers ready)",
                    msg.worker_id,
                    msg.connected,
                    workers_ready,
                    len(client_processes),
                )
            elif isinstance(msg, WorkerError):
                logger.error("Worker %d error: %s", msg.worker_id, msg.error)
                workers_ready += 1

        logger.warning(
            "All %d client workers ready: %d connected, %d failed",
            len(client_processes),
            total_connected,
            total_failed,
        )

        # Tell servers to count their clients and report ready
        logger.warning("Signaling servers to check clients...")
        check_cmd = ServerCheckClients()
        for cmd_queue in server_command_queues:
            cmd_queue.put(check_cmd)

        # Now wait for all servers to report ready
        servers_ready = 0
        total_server_clients = 0
        while servers_ready < args.servers:
            msg = await loop.run_in_executor(None, server_status_queue.get)
            if isinstance(msg, ServerReady):
                servers_ready += 1
                total_server_clients += msg.client_count
                logger.warning(
                    "Server %d ready with %d clients (%d/%d servers ready)",
                    msg.server_id,
                    msg.client_count,
                    servers_ready,
                    args.servers,
                )
            elif isinstance(msg, ServerError):
                logger.error("Server %d error: %s", msg.server_id, msg.error)
                servers_ready += 1  # Count as ready to avoid hanging

        logger.warning(
            "All processes ready: %d servers with %d clients (%d failed)",
            args.servers,
            total_server_clients,
            total_failed,
        )

        # Calculate initial timestamp using system monotonic time
        # All processes share the same monotonic clock on the same machine
        initial_play_start_us = int(time.monotonic() * 1_000_000) + DEFAULT_INITIAL_DELAY_US

        # Send start command to all servers
        logger.warning("Sending start command with initial_play_start_us=%d", initial_play_start_us)
        start_cmd = ServerStart(initial_play_start_us=initial_play_start_us)
        for cmd_queue in server_command_queues:
            cmd_queue.put(start_cmd)

        # Wait for all servers to complete streaming
        servers_done = 0
        total_chunks_sent = 0
        while servers_done < args.servers:
            try:
                msg = await asyncio.wait_for(
                    loop.run_in_executor(None, server_status_queue.get),
                    timeout=args.duration_s + 30,
                )
                if isinstance(msg, ServerDone):
                    servers_done += 1
                    total_chunks_sent += msg.chunks_sent
                    logger.warning(
                        "Server %d done: %d chunks (%d/%d servers done)",
                        msg.server_id,
                        msg.chunks_sent,
                        servers_done,
                        args.servers,
                    )
                elif isinstance(msg, ServerError):
                    servers_done += 1
                    logger.error("Server %d error: %s", msg.server_id, msg.error)
            except TimeoutError:
                logger.warning("Timeout waiting for server completion")
                break

        # Signal client workers to stop
        logger.warning("Signaling client workers to stop...")
        client_stop_event.set()

        # Collect stats from client workers
        all_audio_stats: dict[str, AudioStats] = {}
        all_sync_stats: list[SyncStats] = []
        stats_collected = 0

        while stats_collected < len(client_processes):
            try:
                msg = await asyncio.wait_for(
                    loop.run_in_executor(None, client_message_queue.get),
                    timeout=30.0,
                )
                if isinstance(msg, WorkerStats):
                    stats_collected += 1
                    for cid, d in msg.audio_stats_json.items():
                        all_audio_stats[cid] = _dict_to_audio_stats(d)
                    all_sync_stats.append(_dict_to_sync_stats(msg.sync_stats_json))
                    logger.warning(
                        "Worker %d stats collected (%d/%d)",
                        msg.worker_id,
                        stats_collected,
                        len(client_processes),
                    )
                elif isinstance(msg, WorkerError):
                    stats_collected += 1
                    logger.error("Worker %d error: %s", msg.worker_id, msg.error)
            except TimeoutError:
                logger.warning("Timeout waiting for worker stats")
                break

        # Log aggregated stats
        avg_chunks = total_chunks_sent // args.servers if args.servers > 0 else 0
        frame_duration_ms = 25
        expected_frames = avg_chunks * (args.chunk_ms // frame_duration_ms)

        _aggregate_sync_stats(all_sync_stats)
        _aggregate_audio_stats(all_audio_stats, expected_frames)
        _verify_cross_server_sync(all_audio_stats, args.servers)

    finally:
        # Cleanup: terminate any hanging processes
        for p in server_processes + client_processes:
            if p.is_alive():
                p.terminate()
        for p in server_processes + client_processes:
            p.join(timeout=5.0)

    return 0


async def _main_async_multi_server(args: argparse.Namespace) -> int:
    """Run benchmark with multiple server instances sharing a clock (single process)."""
    loop = asyncio.get_running_loop()
    shared_clock = LoopClock(loop)

    servers: list[SendspinServer] = []
    streams: list[PushStream] = []
    clients: list[SendspinClient] = []
    audio_stats: dict[str, AudioStats] = {}
    session: ClientSession | None = None
    started_count = 0

    try:
        # Create and start multiple servers with shared clock
        for i in range(args.servers):
            port = args.base_port + i
            server = SendspinServer(
                loop=loop,
                server_id=f"bench-{i}",
                server_name=f"bench-{i}",
                clock=shared_clock,
            )
            await server.start_server(
                port=port,
                host=args.host,
                advertise_addresses=[],
                discover_clients=False,
            )
            servers.append(server)
            started_count += 1

        # Build URLs for round-robin client distribution
        urls = [
            f"ws://{args.host}:{args.base_port + i}{SendspinServer.API_PATH}"
            for i in range(args.servers)
        ]

        connector = TCPConnector(limit=0, limit_per_host=0, ttl_dns_cache=0)
        session = ClientSession(connector=connector)

        # Distribute clients round-robin across servers
        logger.warning(
            "Connecting %d clients across %d servers (ports %d-%d)",
            args.clients,
            args.servers,
            args.base_port,
            args.base_port + args.servers - 1,
        )

        # Connect clients to each server
        clients_per_server = args.clients // args.servers
        remainder = args.clients % args.servers

        for i, url in enumerate(urls):
            count = clients_per_server + (1 if i < remainder else 0)
            if count == 0:
                continue
            start_index = sum(clients_per_server + (1 if j < remainder else 0) for j in range(i))
            server_clients, server_audio_stats = await _connect_clients(
                url=url,
                count=count,
                ramp_rate=args.ramp_rate,
                max_in_flight=args.max_in_flight,
                connect_timeout_s=args.connect_timeout_s,
                connect_phase_timeout_s=args.connect_phase_timeout_s,
                session=session,
                sample_rate=args.sample_rate,
                channels=args.channels,
                bit_depth=args.bit_depth,
                start_index=start_index,
                codec=args.codec,
                opus_bitrate=args.opus_bitrate,
            )
            clients.extend(server_clients)
            audio_stats.update(server_audio_stats)

        # Group clients on each server and collect streams
        for server in servers:
            connected = list(server.connected_clients)
            if not connected:
                continue
            leader = connected[0]
            group = leader.group
            for client in connected[1:]:
                if client.group != group:
                    await group.add_client(client)
            stream = group.start_stream()
            streams.append(stream)

        if not streams:
            logger.warning("No streams created (no clients connected)")
            return 1

        # Stream with synchronized timestamps
        chunks_sent = await _run_stream_multi_server(
            servers=servers,
            streams=streams,
            clock=shared_clock,
            duration_s=args.duration_s,
            chunk_ms=args.chunk_ms,
            sample_rate=args.sample_rate,
            channels=args.channels,
            bit_depth=args.bit_depth,
        )

        # Collect and log stats
        _log_sync_stats(_collect_sync_stats(clients))
        frame_duration_ms = 25
        expected_frames = chunks_sent * (args.chunk_ms // frame_duration_ms)
        _log_audio_stats(audio_stats, expected_frames)
        _verify_cross_server_sync(audio_stats, args.servers)

    finally:
        for client in clients:
            with contextlib.suppress(Exception):
                await client.disconnect()
        if session is not None:
            with contextlib.suppress(Exception):
                await session.close()
        for i, server in enumerate(servers):
            if i < started_count:
                with contextlib.suppress(Exception):
                    await server.close()
            else:
                with contextlib.suppress(Exception):
                    if server._owns_session and not server._client_session.closed:  # noqa: SLF001
                        await server._client_session.close()  # noqa: SLF001

    return 0


async def _main_async(argv: Iterable[str]) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Dispatch to multiprocess mode if --workers > 0 (legacy: single server, multiple client workers)
    if args.workers > 0:
        return await _main_async_multiprocess(args)

    # Dispatch to distributed mode if --servers > 1 (separate process per server and per ~100 clients)
    if args.servers > 1:
        return await _main_async_distributed(args)

    loop = asyncio.get_running_loop()
    server = SendspinServer(loop=loop, server_id="bench", server_name="bench")
    clients: list[SendspinClient] = []
    audio_stats: dict[str, AudioStats] = {}
    session: ClientSession | None = None
    started = False

    try:
        await server.start_server(
            port=args.port,
            host=args.host,
            advertise_addresses=[],
            discover_clients=False,
        )
        started = True

        url = f"ws://{args.host}:{args.port}{server.API_PATH}"
        connector = TCPConnector(limit=0, limit_per_host=0, ttl_dns_cache=0)
        session = ClientSession(connector=connector)

        logger.warning("Connecting %d clients to %s", args.clients, url)
        clients, audio_stats = await _connect_clients(
            url=url,
            count=args.clients,
            ramp_rate=args.ramp_rate,
            max_in_flight=args.max_in_flight,
            connect_timeout_s=args.connect_timeout_s,
            connect_phase_timeout_s=args.connect_phase_timeout_s,
            session=session,
            sample_rate=args.sample_rate,
            channels=args.channels,
            bit_depth=args.bit_depth,
            codec=args.codec,
            opus_bitrate=args.opus_bitrate,
        )
        await _group_clients(server, clients)
        await _run_stream(
            server=server,
            clients=clients,
            audio_stats=audio_stats,
            duration_s=args.duration_s,
            chunk_ms=args.chunk_ms,
            sample_rate=args.sample_rate,
            channels=args.channels,
            bit_depth=args.bit_depth,
        )
    finally:
        for client in clients:
            with contextlib.suppress(Exception):
                await client.disconnect()
        if session is not None:
            with contextlib.suppress(Exception):
                await session.close()
        if started:
            with contextlib.suppress(Exception):
                await server.close()
        else:
            with contextlib.suppress(Exception):
                if server._owns_session and not server._client_session.closed:  # noqa: SLF001
                    await server._client_session.close()  # noqa: SLF001

    return 0


def main() -> None:
    try:
        raise SystemExit(asyncio.run(_main_async(sys.argv[1:])))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
