"""Tests for MetadataGroupRole."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aiosendspin.clock import ManualClock
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


def _member(*, legacy: bool) -> MagicMock:
    member = MagicMock()
    member.clears_state_with_null.return_value = legacy
    return member


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
    """clear() sets metadata to None and sends a timestamp-only metadata object."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)

    member = _member(legacy=False)
    mgr._members = [member]  # noqa: SLF001

    mgr.set_metadata(Metadata(title="Test"))
    member.reset_mock()

    mgr.clear()

    assert mgr.metadata is None
    member.send_message.assert_called_once()
    msg = member.send_message.call_args.args[0]
    assert isinstance(msg, ServerStateMessage)
    assert msg.payload.to_dict() == {"metadata": {"timestamp": 1_000_000}}
    group._signal_event.assert_called()  # noqa: SLF001
    event = group._signal_event.call_args.args[0]  # noqa: SLF001
    assert isinstance(event, MetadataClearedEvent)


def test_metadata_group_role_set_metadata_none_sends_timestamp_only() -> None:
    """set_metadata(None) sends a timestamp-only metadata object with the present time."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)

    member = _member(legacy=False)
    mgr._members = [member]  # noqa: SLF001

    mgr.set_metadata(Metadata(title="Test"))
    group._server.clock.now_us.return_value = 2_000_000  # noqa: SLF001
    member.reset_mock()

    mgr.set_metadata(None)

    msg = member.send_message.call_args.args[0]
    assert msg.payload.to_dict() == {"metadata": {"timestamp": 2_000_000}}


# DEPRECATED(spec-pr-275): remove in aiosendspin <version>
def test_metadata_group_role_clear_sends_null_to_legacy_member() -> None:
    """clear() sends a metadata null to a legacy-generation member."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)

    legacy = _member(legacy=True)
    current = _member(legacy=False)
    mgr._members = [legacy, current]  # noqa: SLF001

    mgr.set_metadata(Metadata(title="Test"))
    legacy.reset_mock()
    current.reset_mock()

    mgr.clear()

    assert legacy.send_message.call_args.args[0].to_dict() == {
        "type": "server/state",
        "payload": {"metadata": None},
    }
    assert current.send_message.call_args.args[0].payload.to_dict() == {
        "metadata": {"timestamp": 1_000_000}
    }


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
    """on_member_join() sends a timestamp-only metadata object when no metadata is set."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)

    new_member = _member(legacy=False)
    mgr.on_member_join(new_member)

    new_member.send_message.assert_called_once()
    msg = new_member.send_message.call_args.args[0]
    assert isinstance(msg, ServerStateMessage)
    assert msg.payload.to_dict() == {"metadata": {"timestamp": 1_000_000}}


# DEPRECATED(spec-pr-275): remove in aiosendspin <version>
def test_metadata_group_role_on_member_join_no_metadata_legacy() -> None:
    """on_member_join() sends a metadata null to a legacy-generation member without metadata."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)

    new_member = _member(legacy=True)
    mgr.on_member_join(new_member)

    new_member.send_message.assert_called_once()
    assert new_member.send_message.call_args.args[0].to_dict() == {
        "type": "server/state",
        "payload": {"metadata": None},
    }


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


def test_update_within_progress_tolerance_sends_nothing() -> None:
    """A progress refresh close to the extrapolated position is not sent."""
    group, mgr, member = _playing_group_role()
    group._signal_event.reset_mock()  # noqa: SLF001

    mgr.update(track_progress=40_200)

    member.send_message.assert_not_called()
    group._signal_event.assert_not_called()  # noqa: SLF001


def test_seek_within_progress_tolerance_is_sent() -> None:
    """A seek is sent even when it lands close to the extrapolated position."""
    group, mgr, member = _playing_group_role()
    other_member = MagicMock()
    mgr._members.append(other_member)  # noqa: SLF001
    group._signal_event.reset_mock()  # noqa: SLF001

    mgr.seek(40_200)

    for recipient in (member, other_member):
        assert _sent_metadata(recipient)["timestamp"] == 11_000_000
        assert _sent_metadata(recipient)["progress"] == {
            "track_progress": 40_200,
            "track_duration": 180_000,
            "playback_speed": 1000,
        }
    event = group._signal_event.call_args.args[0]  # noqa: SLF001
    assert isinstance(event, MetadataUpdatedEvent)
    assert event.metadata.track_progress == 40_200


def test_seek_to_negative_position_raises_without_sending() -> None:
    """A negative seek position is rejected and nothing is sent."""
    group, mgr, member = _playing_group_role()
    group._signal_event.reset_mock()  # noqa: SLF001

    with pytest.raises(ValueError, match="non-negative"):
        mgr.seek(-1)

    member.send_message.assert_not_called()
    group._signal_event.assert_not_called()  # noqa: SLF001
    assert mgr.metadata is not None
    assert mgr.metadata.track_progress == 30_000


def test_seek_without_playback_speed_raises() -> None:
    """A seek needs metadata that carries a playback speed."""
    mgr = MetadataGroupRole(_make_group_stub())
    with pytest.raises(ValueError, match="playback_speed"):
        mgr.seek(1_000)

    mgr.set_metadata(Metadata(title="Song"))
    with pytest.raises(ValueError, match="playback_speed"):
        mgr.seek(1_000)


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


def _make_scheduling_group() -> tuple[MagicMock, ManualClock]:
    clock = ManualClock(now_us_value=1_000_000)
    group = MagicMock()
    group._server.clock = clock  # noqa: SLF001
    group.has_active_stream = True
    return group, clock


def _all_sent_metadata(member: MagicMock) -> list[dict[str, object] | None]:
    return [
        call.args[0].payload.to_dict()["metadata"] for call in member.send_message.call_args_list
    ]


def _track(title: str, progress: int, timestamp_us: int | None = None) -> Metadata:
    return Metadata(
        title=title,
        track_progress=progress,
        track_duration=180_000,
        playback_speed=1000,
        timestamp_us=timestamp_us,
    )


def test_future_metadata_is_sent_and_takes_effect_later() -> None:
    """Metadata with a future timestamp is sent as is and becomes current once due."""
    group, clock = _make_scheduling_group()
    mgr = MetadataGroupRole(group)
    member = _member(legacy=False)
    mgr._members = [member]  # noqa: SLF001
    mgr.set_metadata(_track("Now", 30_000))

    mgr.set_metadata(_track("Next", 0, timestamp_us=1_500_000))

    assert _all_sent_metadata(member)[-1] == {
        "timestamp": 1_500_000,
        "title": "Next",
        "progress": {"track_progress": 0, "track_duration": 180_000, "playback_speed": 1000},
    }
    event = group._signal_event.call_args.args[0]  # noqa: SLF001
    assert isinstance(event, MetadataUpdatedEvent)
    assert event.timestamp_us == 1_500_000
    assert event.previous_metadata is not None
    assert event.previous_metadata.title == "Now"
    assert mgr.metadata is not None
    assert mgr.metadata.title == "Now"

    clock.advance_us(600_000)
    assert mgr.metadata is not None
    assert mgr.metadata.title == "Next"
    assert mgr.track_progress == 100


def test_late_join_gets_current_then_scheduled_metadata() -> None:
    """A joining member gets the current metadata as of now, then the scheduled one."""
    group, clock = _make_scheduling_group()
    mgr = MetadataGroupRole(group)
    mgr.set_metadata(_track("Now", 30_000))
    mgr.set_metadata(_track("Next", 0, timestamp_us=1_500_000))
    clock.advance_us(200_000)

    member = _member(legacy=False)
    mgr.on_member_join(member)

    sent = _all_sent_metadata(member)
    assert sent[0] == {
        "timestamp": 1_200_000,
        "title": "Now",
        "progress": {"track_progress": 30_200, "track_duration": 180_000, "playback_speed": 1000},
    }
    assert sent[1] is not None
    assert sent[1]["timestamp"] == 1_500_000
    assert sent[1]["title"] == "Next"


def test_later_scheduled_metadata_replaces_earlier_by_arrival() -> None:
    """The last scheduled metadata wins even when its timestamp is earlier."""
    group, clock = _make_scheduling_group()
    mgr = MetadataGroupRole(group)
    mgr.set_metadata(_track("A", 0, timestamp_us=1_800_000))
    mgr.set_metadata(_track("B", 0, timestamp_us=1_400_000))

    clock.advance_us(1_000_000)

    assert mgr.metadata is not None
    assert mgr.metadata.title == "B"


def test_present_metadata_cancels_scheduled_even_when_unchanged() -> None:
    """Unchanged metadata set now is still sent, since it cancels the scheduled track."""
    group, clock = _make_scheduling_group()
    mgr = MetadataGroupRole(group)
    member = _member(legacy=False)
    mgr._members = [member]  # noqa: SLF001
    mgr.set_metadata(_track("Now", 30_000))
    mgr.set_metadata(_track("Next", 0, timestamp_us=1_500_000))

    mgr.update(title="Now")

    assert len(member.send_message.call_args_list) == 3
    clock.advance_us(600_000)
    assert mgr.metadata is not None
    assert mgr.metadata.title == "Now"


def test_cancel_scheduled_metadata_resends_current_now() -> None:
    """cancel_scheduled() re-sends the current metadata as of now."""
    group, clock = _make_scheduling_group()
    mgr = MetadataGroupRole(group)
    member = _member(legacy=False)
    mgr._members = [member]  # noqa: SLF001
    mgr.set_metadata(_track("Now", 30_000))
    mgr.set_metadata(_track("Next", 0, timestamp_us=1_500_000))
    clock.advance_us(100_000)

    mgr.cancel_scheduled()
    mgr.cancel_scheduled()

    assert _all_sent_metadata(member)[2:] == [
        {
            "timestamp": 1_100_000,
            "title": "Now",
            "progress": {
                "track_progress": 30_100,
                "track_duration": 180_000,
                "playback_speed": 1000,
            },
        }
    ]
    clock.advance_us(600_000)
    assert mgr.metadata is not None
    assert mgr.metadata.title == "Now"
    assert mgr.track_progress == 30_700


def test_clear_discards_scheduled_metadata() -> None:
    """clear() sends null at once and the scheduled metadata never takes effect."""
    group, clock = _make_scheduling_group()
    mgr = MetadataGroupRole(group)
    member = _member(legacy=False)
    mgr._members = [member]  # noqa: SLF001
    mgr.set_metadata(_track("Next", 0, timestamp_us=1_500_000))

    mgr.clear()
    clock.advance_us(600_000)

    assert _all_sent_metadata(member)[-1] == {"timestamp": 1_000_000}
    assert mgr.metadata is None


def test_metadata_beyond_lead_limit_is_sent_20s_ahead() -> None:
    """Metadata more than 20 s ahead is sent only 20 s ahead, and not to joiners before."""
    group, clock = _make_scheduling_group()
    mgr = MetadataGroupRole(group)
    member = _member(legacy=False)
    mgr._members = [member]  # noqa: SLF001

    mgr.set_metadata(_track("Next", 0, timestamp_us=31_000_000))

    member.send_message.assert_not_called()
    (delay_s, send), _kwargs = group._server.loop.call_later.call_args  # noqa: SLF001
    assert delay_s == 10.0
    joiner = _member(legacy=False)
    mgr.on_member_join(joiner)
    assert _all_sent_metadata(joiner) == [{"timestamp": 1_000_000}]

    clock.advance_us(10_000_000)
    send()
    sent = _all_sent_metadata(member)
    assert len(sent) == 1
    assert sent[0] is not None
    assert sent[0]["timestamp"] == 31_000_000


def test_sent_metadata_replaced_by_deferred_one_is_cancelled() -> None:
    """Replacing sent metadata with metadata sent only later restates the current metadata now."""
    group, _clock = _make_scheduling_group()
    mgr = MetadataGroupRole(group)
    member = _member(legacy=False)
    mgr._members = [member]  # noqa: SLF001
    mgr.set_metadata(_track("Now", 30_000))
    mgr.set_metadata(_track("Next", 0, timestamp_us=1_500_000))

    mgr.set_metadata(_track("Much later", 0, timestamp_us=31_000_000))

    sent = _all_sent_metadata(member)
    assert len(sent) == 3
    assert sent[2] is not None
    assert sent[2]["timestamp"] == 1_000_000
    assert sent[2]["title"] == "Now"
