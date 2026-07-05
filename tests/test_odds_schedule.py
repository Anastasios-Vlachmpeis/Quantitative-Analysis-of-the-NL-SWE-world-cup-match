# Tests for CLV capture poll schedule.

from __future__ import annotations

from datetime import datetime, timezone

from nlvswe.config import ClvCaptureWindow, load_config
from nlvswe.data.odds_schedule import (
    capture_window_start,
    planned_poll_times,
    poll_interval_seconds,
    seconds_until_next_action,
    sleep_seconds_after_poll,
)

KICKOFF = datetime(2026, 6, 20, 17, 0, tzinfo=timezone.utc)
WINDOWS = [
    ClvCaptureWindow(before_kickoff_hours=48, interval_minutes=360),
    ClvCaptureWindow(before_kickoff_hours=6, interval_minutes=60),
    ClvCaptureWindow(before_kickoff_hours=1, interval_minutes=15),
]


def test_config_loads_clv_capture_windows():
    cfg = load_config()
    assert len(cfg.betting.clv_capture.windows) == 3
    assert cfg.betting.clv_capture.min_requests_remaining == 10


def test_poll_interval_tightens_near_kickoff():
    assert poll_interval_seconds(50 * 3600, WINDOWS) is None
    assert poll_interval_seconds(30 * 3600, WINDOWS) == 6 * 3600
    assert poll_interval_seconds(3 * 3600, WINDOWS) == 3600
    assert poll_interval_seconds(30 * 60, WINDOWS) == 15 * 60


def test_capture_window_start():
    start = capture_window_start(KICKOFF, WINDOWS)
    assert start == datetime(2026, 6, 18, 17, 0, tzinfo=timezone.utc)


def test_seconds_until_next_action_wait_then_poll():
    too_early = datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc)
    action, wait = seconds_until_next_action(too_early, KICKOFF, WINDOWS)
    assert action == "wait"
    assert wait > 0

    in_window = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)
    action, wait = seconds_until_next_action(in_window, KICKOFF, WINDOWS)
    assert action == "poll"
    assert wait == 0.0


def test_planned_poll_times_produces_multiple_snapshots():
    start = datetime(2026, 6, 18, 17, 0, tzinfo=timezone.utc)
    times = planned_poll_times(start, KICKOFF, WINDOWS)
    assert len(times) >= 2
    assert all(t < KICKOFF for t in times)
    assert times == sorted(times)
