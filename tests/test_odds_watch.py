# Tests for the CLV odds watcher

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from nlvswe.config import ClvCaptureConfig, ClvCaptureWindow, load_config
from nlvswe.data.acquire import OddsLiveSnapshot
from nlvswe.data.odds_watch import run_odds_watch

FIXTURES = Path(__file__).parent / "fixtures"
FAKE_API_KEY = "TEST_SECRET_DO_NOT_PERSIST_xyz789"
KICKOFF = datetime(2026, 6, 20, 17, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _clear_config_cache():
    from nlvswe.config import get_config

    get_config.cache_clear()
    yield
    get_config.cache_clear()


@pytest.fixture
def watch_cfg(tmp_path, monkeypatch):
    import nlvswe.data.acquire as acquire

    monkeypatch.setattr(acquire, "PROJECT_ROOT", tmp_path)
    cfg = load_config()
    cfg.betting.clv_capture = ClvCaptureConfig(
        min_requests_remaining=10,
        windows=[ClvCaptureWindow(before_kickoff_hours=1, interval_minutes=2)],
    )
    return tmp_path, cfg


def _fake_snapshot(tmp_path: Path, when: datetime, remaining: int = 100) -> OddsLiveSnapshot:
    stamp = when.strftime("%Y%m%dT%H%M%SZ")
    snap_dir = tmp_path / "data" / "raw" / "odds_live"
    snap_dir.mkdir(parents=True, exist_ok=True)
    dest = snap_dir / f"{stamp}.json"
    dest.write_text("[]", encoding="utf-8")
    manifest = {
        "source": "odds_live",
        "captured_at": when.replace(microsecond=0).isoformat(),
    }
    dest.with_suffix(".json.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return OddsLiveSnapshot(path=dest, requests_remaining=remaining)


def test_odds_watch_simulated_schedule_writes_multiple_snapshots(watch_cfg):
    tmp_path, cfg = watch_cfg
    clock = [datetime(2026, 6, 20, 16, 50, tzinfo=timezone.utc)]

    def now_fn():
        return clock[0]

    def sleep_fn(seconds: float) -> None:
        clock[0] = clock[0] + timedelta(seconds=seconds)

    def fetch_fn(_cfg, *, captured_at=None):
        return _fake_snapshot(tmp_path, captured_at or clock[0], remaining=100)

    paths = run_odds_watch(
        KICKOFF,
        cfg,
        now_fn=now_fn,
        sleep_fn=sleep_fn,
        fetch_fn=fetch_fn,
    )
    assert len(paths) >= 2
    assert len({p.name for p in paths}) == len(paths)


def test_odds_watch_budget_guard_stops_before_next_fetch(watch_cfg):
    tmp_path, cfg = watch_cfg
    clock = [datetime(2026, 6, 20, 16, 50, tzinfo=timezone.utc)]
    fetch_calls = 0

    def now_fn():
        return clock[0]

    def sleep_fn(seconds: float) -> None:
        clock[0] = clock[0] + timedelta(seconds=seconds)

    def fetch_fn(_cfg, *, captured_at=None):
        nonlocal fetch_calls
        fetch_calls += 1
        return _fake_snapshot(tmp_path, captured_at or clock[0], remaining=5)

    paths = run_odds_watch(
        KICKOFF,
        cfg,
        now_fn=now_fn,
        sleep_fn=sleep_fn,
        fetch_fn=fetch_fn,
    )
    assert fetch_calls == 1
    assert len(paths) == 1
