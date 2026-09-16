"""Tests for MetadataGroupRole."""

from __future__ import annotations

from unittest.mock import MagicMock

from aiosendspin.models.core import ServerStateMessage
from aiosendspin.models.types import RepeatMode
from aiosendspin.server.roles.metadata import Metadata, MetadataClearedEvent, MetadataUpdatedEvent
from aiosendspin.server.roles.metadata.group import MetadataGroupRole


def _make_group_stub() -> MagicMock:
    """Create a mock group for testing."""
    group = MagicMock()
    group._server = MagicMock()  # noqa: SLF001
    group._server.clock.now_us.return_value = 1_000_000  # noqa: SLF001
    group.has_active_stream = False
    return group


def test_metadata_group_role_family() -> None:
    """MetadataGroupRole has role_family of 'metadata'."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)
    assert mgr.role_family == "metadata"


def test_metadata_group_role_initial_metadata_is_none() -> None:
    """Initial metadata is None."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)
    assert mgr.metadata is None


def test_metadata_group_role_set_metadata_stores_value() -> None:
    """set_metadata() stores the metadata."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)

    metadata = Metadata(title="Test Song", artist="Test Artist")
    mgr.set_metadata(metadata)

    assert mgr.metadata is not None
    assert mgr.metadata.title == "Test Song"
    assert mgr.metadata.artist == "Test Artist"
    group._signal_event.assert_called_once()  # noqa: SLF001
    event = group._signal_event.call_args.args[0]  # noqa: SLF001
    assert isinstance(event, MetadataUpdatedEvent)
    assert event.metadata.title == "Test Song"
    assert event.previous_metadata is None


def test_metadata_group_role_set_metadata_sends_to_members() -> None:
    """set_metadata() sends update to all subscribed members."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)

    member = MagicMock()
    mgr._members = [member]  # noqa: SLF001

    metadata = Metadata(title="Test Song")
    mgr.set_metadata(metadata)

    member.send_message.assert_called_once()
    msg = member.send_message.call_args.args[0]
    assert isinstance(msg, ServerStateMessage)
    assert msg.payload.metadata is not None
    assert msg.payload.metadata.title == "Test Song"


def test_metadata_group_role_clear_metadata() -> None:
    """clear() sets metadata to None and sends clear update."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)

    member = MagicMock()
    mgr._members = [member]  # noqa: SLF001

    mgr.set_metadata(Metadata(title="Test"))
    member.reset_mock()

    mgr.clear()

    assert mgr.metadata is None
    member.send_message.assert_called_once()
    msg = member.send_message.call_args.args[0]
    assert isinstance(msg, ServerStateMessage)
    assert msg.payload.to_dict() == {"metadata": None}
    group._signal_event.assert_called()  # noqa: SLF001
    event = group._signal_event.call_args.args[0]  # noqa: SLF001
    assert isinstance(event, MetadataClearedEvent)


def test_metadata_group_role_clear_when_already_cleared_is_noop() -> None:
    """Clearing already-cleared metadata sends nothing and emits no event."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)

    member = MagicMock()
    mgr._members = [member]  # noqa: SLF001

    mgr.clear()

    member.send_message.assert_not_called()
    group._signal_event.assert_not_called()  # noqa: SLF001


def test_metadata_group_role_update_title() -> None:
    """update() updates only the title field."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)

    mgr.update(title="New Title")

    assert mgr.metadata is not None
    assert mgr.metadata.title == "New Title"


def test_metadata_group_role_update_artist() -> None:
    """update() updates only the artist field."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)

    mgr.update(artist="New Artist")

    assert mgr.metadata is not None
    assert mgr.metadata.artist == "New Artist"


def test_metadata_group_role_update_progress() -> None:
    """update() updates progress fields."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)

    mgr.update(track_progress=30000, track_duration=180000, playback_speed=1000)

    assert mgr.metadata is not None
    assert mgr.metadata.track_progress == 30000
    assert mgr.metadata.track_duration == 180000
    assert mgr.metadata.playback_speed == 1000


def test_metadata_group_role_update_batch() -> None:
    """update() can set multiple fields at once."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)

    mgr.update(title="Song", artist="Artist", year=2024)

    assert mgr.metadata is not None
    assert mgr.metadata.title == "Song"
    assert mgr.metadata.artist == "Artist"
    assert mgr.metadata.year == 2024


def test_metadata_group_role_update_can_clear_field_with_none() -> None:
    """update() should allow clearing a field via explicit None."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)
    mgr.set_metadata(Metadata(title="Song", artist="Artist"))

    mgr.update(title=None)

    assert mgr.metadata is not None
    assert mgr.metadata.title is None
    assert mgr.metadata.artist == "Artist"


def test_metadata_group_role_on_member_join_sends_current_state() -> None:
    """on_member_join() sends current metadata to new member."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)
    mgr.set_metadata(Metadata(title="Test Song"))

    new_member = MagicMock()
    mgr.on_member_join(new_member)

    new_member.send_message.assert_called_once()
    msg = new_member.send_message.call_args.args[0]
    assert isinstance(msg, ServerStateMessage)
    assert msg.payload.metadata is not None
    assert msg.payload.metadata.title == "Test Song"


def test_metadata_group_role_on_member_join_no_metadata() -> None:
    """on_member_join() sends a metadata null when no metadata is set."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)

    new_member = MagicMock()
    mgr.on_member_join(new_member)

    new_member.send_message.assert_called_once()
    msg = new_member.send_message.call_args.args[0]
    assert isinstance(msg, ServerStateMessage)
    assert msg.payload.to_dict() == {"metadata": None}


def test_metadata_group_role_skips_unchanged() -> None:
    """set_metadata() skips sending if metadata is equivalent."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)

    member = MagicMock()
    mgr._members = [member]  # noqa: SLF001

    metadata = Metadata(title="Test")
    mgr.set_metadata(metadata)
    member.reset_mock()

    # Set same metadata again
    same_metadata = Metadata(title="Test")
    mgr.set_metadata(same_metadata)

    # Should not have sent again
    member.send_message.assert_not_called()
    group._signal_event.assert_called_once()  # noqa: SLF001


def test_metadata_group_role_freeze_progress_snapshots_elapsed_position() -> None:
    """freeze_progress() should snapshot live progress and stop extrapolation."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)
    group.has_active_stream = True

    mgr.set_metadata(
        Metadata(
            title="Test",
            track_progress=30_000,
            track_duration=180_000,
            playback_speed=1000,
        )
    )

    group._server.clock.now_us.return_value = 11_000_000  # noqa: SLF001
    mgr.freeze_progress()

    assert mgr.metadata is not None
    assert mgr.metadata.track_progress == 40_000
    assert mgr.metadata.playback_speed == 0


def test_metadata_group_role_member_join_does_not_rewind_after_freeze() -> None:
    """Frozen progress should be sent unchanged after the stream becomes inactive."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)
    group.has_active_stream = True

    mgr.set_metadata(
        Metadata(
            title="Test",
            track_progress=30_000,
            track_duration=180_000,
            playback_speed=1000,
        )
    )

    group._server.clock.now_us.return_value = 11_000_000  # noqa: SLF001
    mgr.freeze_progress()
    group.has_active_stream = False

    new_member = MagicMock()
    mgr.on_member_join(new_member)

    msg = new_member.send_message.call_args.args[0]
    assert isinstance(msg, ServerStateMessage)
    assert msg.payload.metadata is not None
    assert msg.payload.metadata.progress is not None
    assert msg.payload.metadata.progress.track_progress == 40_000
    assert msg.payload.metadata.progress.playback_speed == 0


def _sent_metadata(member: MagicMock) -> dict[str, object]:
    msg = member.send_message.call_args.args[0]
    assert isinstance(msg, ServerStateMessage)
    metadata = msg.payload.to_dict()["metadata"]
    assert isinstance(metadata, dict)
    return metadata


def test_update_sends_full_state_with_progress_after_title_change() -> None:
    """A title-only change still sends every set field, including progress, with no nulls."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)

    member = MagicMock()
    mgr._members = [member]  # noqa: SLF001

    mgr.set_metadata(
        Metadata(
            title="Song",
            artist="Artist",
            album=None,
            track_progress=5_000,
            track_duration=180_000,
            playback_speed=1000,
        )
    )
    member.reset_mock()

    mgr.update(title="New Title")

    assert _sent_metadata(member) == {
        "timestamp": 1_000_000,
        "title": "New Title",
        "artist": "Artist",
        "progress": {"track_progress": 5_000, "track_duration": 180_000, "playback_speed": 1000},
    }


def test_update_omits_progress_when_position_cleared() -> None:
    """Clearing the position omits progress, which clears it on the client."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)

    member = MagicMock()
    mgr._members = [member]  # noqa: SLF001

    mgr.set_metadata(
        Metadata(title="Song", track_progress=12_345, track_duration=180_000, playback_speed=1000)
    )
    member.reset_mock()

    mgr.set_metadata(Metadata(title="Loading next track..."))

    assert _sent_metadata(member) == {"timestamp": 1_000_000, "title": "Loading next track..."}


def _playing_group_role() -> tuple[MagicMock, MetadataGroupRole, MagicMock]:
    """Return a group role playing at 30s, observed 10s later, with one member."""
    group = _make_group_stub()
    group.has_active_stream = True
    mgr = MetadataGroupRole(group)
    member = MagicMock()
    mgr._members = [member]  # noqa: SLF001
    mgr.set_metadata(
        Metadata(title="Song", track_progress=30_000, track_duration=180_000, playback_speed=1000)
    )
    member.reset_mock()
    group._server.clock.now_us.return_value = 11_000_000  # noqa: SLF001
    return group, mgr, member


def test_update_during_active_stream_sends_current_position() -> None:
    """A non-position update during playback carries the extrapolated position, stamped now."""
    _, mgr, member = _playing_group_role()

    mgr.update(title="New Title")

    assert _sent_metadata(member) == {
        "timestamp": 11_000_000,
        "title": "New Title",
        "progress": {"track_progress": 40_000, "track_duration": 180_000, "playback_speed": 1000},
    }


def test_update_with_explicit_position_is_stamped_now() -> None:
    """A supplied position is taken as the position at the time of the update."""
    _, mgr, member = _playing_group_role()

    mgr.update(track_progress=5_000)

    sent = _sent_metadata(member)
    assert sent["timestamp"] == 11_000_000
    assert sent["progress"] == {
        "track_progress": 5_000,
        "track_duration": 180_000,
        "playback_speed": 1000,
    }


def test_update_pause_during_active_stream_freezes_current_position() -> None:
    """Pausing mid-stream sends the position reached at the old speed, with speed 0."""
    _, mgr, member = _playing_group_role()

    mgr.update(playback_speed=0)

    sent = _sent_metadata(member)
    assert sent["timestamp"] == 11_000_000
    assert sent["progress"] == {
        "track_progress": 40_000,
        "track_duration": 180_000,
        "playback_speed": 0,
    }


def test_update_without_position_is_stamped_now() -> None:
    """An update to metadata without a position carries the time of the update."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)
    member = MagicMock()
    mgr._members = [member]  # noqa: SLF001
    mgr.set_metadata(Metadata(title="Song"))
    group._server.clock.now_us.return_value = 11_000_000  # noqa: SLF001

    mgr.update(title="New Title")

    assert _sent_metadata(member) == {"timestamp": 11_000_000, "title": "New Title"}


def test_update_without_active_stream_keeps_stored_position() -> None:
    """Without an active stream the stored position is sent with its own timestamp."""
    group, mgr, member = _playing_group_role()
    group.has_active_stream = False

    mgr.update(title="New Title")

    sent = _sent_metadata(member)
    assert sent["timestamp"] == 1_000_000
    assert sent["progress"] == {
        "track_progress": 30_000,
        "track_duration": 180_000,
        "playback_speed": 1000,
    }


# DEPRECATED(spec-pr-175): remove in aiosendspin <version>
def test_repeat_and_shuffle_are_accepted_but_never_sent() -> None:
    """Metadata still accepts repeat/shuffle but ignores them on the wire and in equality."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)

    member = MagicMock()
    mgr._members = [member]  # noqa: SLF001

    mgr.set_metadata(Metadata(title="Song", repeat=RepeatMode.ALL, shuffle=True))

    assert _sent_metadata(member) == {"timestamp": 1_000_000, "title": "Song"}

    member.reset_mock()
    mgr.set_metadata(Metadata(title="Song", repeat=RepeatMode.ONE, shuffle=False))

    member.send_message.assert_not_called()
