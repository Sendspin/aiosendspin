# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
This guidance is aimed at Claude Code but may also be suitable for other AI tooling, such as GitHub Copilot and OpenAI Codex.

## Commands

```bash
./scripts/setup.sh                           # One-time: create .venv, install deps, pre-commit hooks
./scripts/run-in-env.sh pre-commit run -a    # Run linting and type checks: ruff, mypy (use before commits)
./scripts/run-in-env.sh pytest               # Run tests only
./scripts/run-in-env.sh pytest tests/server/test_push_stream_behavior.py -v  # Run specific test file
./scripts/run-in-env.sh pytest -k "test_name"  # Run tests matching pattern
```

## Specification

The [Sendspin spec](https://github.com/Sendspin/spec) defines every message, field, and required behavior this library implements. Check protocol changes against it:

- `connection.md`: connection setup, discovery, and encryption
- `pairing.md`: pairing
- `messaging.md`: message format, clock synchronization, and core messages
- `roles/<family>/v1.md`: one file per role family

`README.md` in the spec repo is generated from these files.

When the spec renames or removes a field, the server keeps accepting the old form and calls `flag_noncompliance` (see `server/compliance.py`). The client is only rejected when the server runs with `allow_noncompliant_clients=False`, and in that mode validation happens before any side effect such as marking the client connected or emitting events. The client library supports only the current spec and keeps no old forms.

## Architecture

**aiosendspin** is the async Python implementation of the [Sendspin Protocol](https://github.com/Sendspin/spec) for synchronized audio streaming across networked devices.

The package has five main subpackages: `server` (the protocol server), `client` (a Python SDK for building clients), `models` (shared protocol messages and types), `audio` (PCM formats, codecs, and source bridges), and `noise` (Noise encryption, pairing, and trust stores).

Keep the library generic: no features that only one consumer needs. Policy decisions, such as auto-starting a source when it joins, belong to the application. The library exposes commands and events for them. Treat removed or renamed public symbols, changed signatures, and new abstract methods on stores as breaking changes.

Server-side `SendspinClient` (`server/client.py`) lifetime is decoupled from `SendspinConnection` (WebSocket transport). Clients persist across reconnects and disconnections; roles are retained across reconnections (warm reconnect) and only recreated when the negotiated role set changes (cold reconnect).

### Key Components

- **`server/server.py`** (`SendspinServer`): Entry point. Manages mDNS advertising (Zeroconf), persistent client registry, and external player registration. Emits `ClientAddedEvent`/`ClientRemovedEvent`.

- **`server/client.py`** (`SendspinClient`): Persistent client state. Manages role attachment/detachment, `DisconnectBehaviour` (UNGROUP vs STOP), cleanup delays, and the connected/disconnected state machine.

- **`server/group.py`** (`SendspinGroup`): Orchestrates synchronized playback. Manages `PushStream`, tracks group state/metadata/volume, coordinates `GroupRole` instances. Emits group-level events.

- **`server/push_stream.py`** (`PushStream`): Push-based audio engine with prepare/commit pattern. Handles sequential encoding per `TransformKey`, per-role buffer tracking, backpressure, late-joiner catch-up caching, and multi-channel audio.

- **`server/connection.py`** (`SendspinConnection`): Transient WebSocket transport with priority-based message queues (priority → control → per-role min-heaps sorted by timestamp). Uses epoch-based invalidation to discard stale binary messages after stream boundaries.

- **`clock.py`** (`Clock`): Clock protocol. `RawMonotonicClock` for production (`CLOCK_MONOTONIC_RAW` on Linux, `time.monotonic` elsewhere), `LoopClock` for tests with mock loops, `ManualClock` for deterministic tests.

- **`server/channels.py`**: Multi-channel routing. `MAIN_CHANNEL` is the default; additional channels support independent playback timelines.

- **`server/audio_transformers.py`** (`TransformerPool`): Encoding pool with deduplication by `TransformKey(channel_id, transformer_type, sample_rate, bit_depth, channels, frame_duration_us, options)`.

### Role System

Two-level architecture: **`Role`** (per-connection) and **`GroupRole`** (per-group per-family).

**Role** (ABC in `server/roles/base.py`):
- Identified by `role_id` (versioned, e.g. `"player@v1"`) and `role_family` (e.g. `"player"`)
- Declares capabilities via `get_stream_requirements()`, `get_audio_requirements()`, `get_binary_handling()`
- Lifecycle hooks: `on_connect()`, `on_disconnect()`, `on_stream_start()`, `on_audio_chunk()`, `on_stream_clear()`, `on_stream_end()`
- Implementations should call `_subscribe_to_group_role()` in `on_connect()` and `_unsubscribe_from_group_role()` in `on_disconnect()` (re-subscription on group changes is handled automatically via `on_group_changed()`)

**GroupRole** (ABC in `server/roles/base.py`):
- One instance per role family per group, coordinates across all member roles
- Hooks: `on_member_join()`, `on_member_leave()`, `on_client_added()`, `on_client_removed()`
- Emits events via `emit_group_event()`

**Built-in role families** (each under `server/roles/{family}/`): `player`, `controller`, `metadata`, `artwork`, `color`, `visualizer`, `source`. Each has a `v1.py` (Role impl), and all but `source` have a `group.py` (GroupRole impl). `visualizer_draft_r1/` keeps the deprecated `visualizer@_draft_r1` wire working.

**Registration**: Roles auto-register via `ROLE_FACTORIES` / `GROUP_ROLE_FACTORIES` in `server/roles/registry.py`. Negotiation (`server/roles/negotiation.py`) picks the first mutually supported role per family from the client's `client/hello`.

### Audio Pipeline

```
User PCM → prepare_audio(pcm, format) → commit_audio()
  → resample + assign timestamps
  → encode per unique TransformKey
  → route to roles via on_audio_chunk()
  → enqueue on connection → WebSocket binary
```

`prepare_historical_audio()` queues historical chunks before live audio for client catch-up on join.

### Models & Serialization

`models/` holds only types sent over the wire. Mashumaro `DataClassORJSONMixin` dataclasses with discriminator-based polymorphic dispatch on `"type"` field. Binary messages use a 9-byte header (1B message type + 8B timestamp_us); player audio chunks add a 4B `send_ahead` (13 bytes, `models/player.py`). Key model files:

- `models/core.py`: Protocol messages (`ClientHelloPayload`, `StreamStartPayload`, `ServerTimePayload`, etc.)
- `models/types.py`: Enums (`AudioCodec`, `BinaryMessageType`, `ClientStateType`, `PlaybackStateType`, etc.)
- `models/{player,controller,metadata,artwork,visualizer}.py`: Per-role model extensions

### Client Library

`client/client.py` (`SendspinClient`): Python SDK for building Sendspin clients. Manages WebSocket connection to server, sends `client/hello` and `client/state`, receives audio chunks. `client/time_sync.py` implements `SendspinTimeFilter` for clock synchronization. `client/listener.py` provides a discovery listener (Zeroconf).

### Event System

Event streams subscribable via `add_event_listener()`:
1. **Server events** (`SendspinServer`): `ClientAddedEvent`, `ClientRemovedEvent`
2. **Group events** (`SendspinGroup`): `GroupStateChangedEvent`, `GroupMemberAddedEvent`/`Removed`, `GroupDeletedEvent`, plus `GroupRoleEvent` subclasses emitted by group roles (e.g. `PlayerGroupVolumeChangedEvent`, `ControllerPlayEvent`, `MetadataUpdatedEvent`)
3. **Client events** (`SendspinClient`): `ClientGroupChangedEvent`, plus `ClientRoleEvent` subclasses (e.g. `VolumeChangedEvent`)

## Code Style

- Python ≥3.12, fully typed (strict mypy config in `pyproject.toml`)
- Ruff with `select = ["ALL"]`, 100 char line length
- Rate-limit warnings on hot paths such as per-chunk audio handling, and add no per-chunk debug logs
- Conventional commits: `type(scope): subject` (feat, fix, refactor, test, chore)

## Testing

- Tests mirror module structure: `tests/server/`, `tests/models/`, `tests/integration/`
- Common fixtures in `tests/conftest.py`: `pcm_44100_stereo_16bit`, `pcm_48000_stereo_16bit`, `mock_loop`
- `ManualClock` from `clock.py` for deterministic timing tests
- No tests that depend on the ffmpeg build installed in CI
