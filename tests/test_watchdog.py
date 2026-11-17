from __future__ import annotations

import asyncio
import stat

import pytest

from kuni import watchdog as w
from kuni.watchdog import Probe, Watchdog


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def ok(**kw) -> Probe:
    base = dict(
        queue_oldest_age=None, worker_busy=False, sleeping=False, paused=False,
        disk_free_ratio=0.5, youtube_proxy_ok=None,
    )
    return Probe(**{**base, **kw})


@pytest.fixture
def env():
    clock, sent = Clock(), []

    async def notify(text):
        sent.append(text)

    return clock, sent, Watchdog(notify, clock)


async def test_healthy_is_silent(env):
    _, sent, wd = env
    await wd.check(ok())
    assert sent == []


async def test_queue_stuck_fires(env):
    _, sent, wd = env
    await wd.check(ok(queue_oldest_age=w.QUEUE_STUCK_SECONDS + 60))
    assert len(sent) == 1 and "Очередь" in sent[0]


async def test_queue_fresh_does_not_fire(env):
    _, sent, wd = env
    await wd.check(ok(queue_oldest_age=w.QUEUE_STUCK_SECONDS - 1))
    assert sent == []


@pytest.mark.parametrize("kw", [{"paused": True}, {"sleeping": True}])
async def test_paused_or_sleeping_suppresses_queue_stuck(env, kw):
    _, sent, wd = env
    await wd.check(ok(queue_oldest_age=99999, **kw))
    assert sent == []


async def test_disk_low_fires(env):
    _, sent, wd = env
    await wd.check(ok(disk_free_ratio=0.05))
    assert len(sent) == 1 and "диск" in sent[0] and "5%" in sent[0]


async def test_proxy_needs_three_consecutive_and_resets(env):
    _, sent, wd = env
    await wd.check(ok(youtube_proxy_ok=False))
    await wd.check(ok(youtube_proxy_ok=False))
    await wd.check(ok(youtube_proxy_ok=True))  # reset
    await wd.check(ok(youtube_proxy_ok=False))
    await wd.check(ok(youtube_proxy_ok=None))  # reset
    await wd.check(ok(youtube_proxy_ok=False))
    await wd.check(ok(youtube_proxy_ok=False))
    assert sent == []
    await wd.check(ok(youtube_proxy_ok=False))
    assert len(sent) == 1 and "Прокси" in sent[0]


async def test_llm_stall_alerts(env):
    clock, sent, wd = env
    clock.t += w.NO_SUCCESS_SECONDS + 1
    for _ in range(w.STALL_RATE_LIMITS):
        wd.record_rate_limit()
    await wd.check(ok())
    assert len(sent) == 1 and "Нейросеть" in sent[0]


async def test_slow_but_progressing_429_does_not_alert(env):
    clock, sent, wd = env
    for _ in range(20):
        clock.t += 30
        wd.record_rate_limit()
        wd.record_rate_limit()
        wd.record_llm_success()
        await wd.check(ok())
    assert sent == []


async def test_old_rate_limits_expire(env):
    clock, sent, wd = env
    for _ in range(w.STALL_RATE_LIMITS):
        wd.record_rate_limit()
    clock.t += w.NO_SUCCESS_SECONDS + 1  # no success, but the 429s are outside the window
    await wd.check(ok())
    assert sent == []


async def test_llm_recovery_requires_success_not_just_expired_errors(env):
    clock, sent, wd = env
    clock.t += w.NO_SUCCESS_SECONDS + 1
    for _ in range(w.STALL_RATE_LIMITS):
        wd.record_rate_limit()
    await wd.check(ok())
    clock.t += w.STALL_WINDOW + 1
    await wd.check(ok())
    assert len(sent) == 1
    wd.record_llm_success()
    await wd.check(ok())
    assert sent[-1] == w._RECOVERED["llm_stalled"]


async def test_pause_does_not_pretend_queue_recovered(env):
    _, sent, wd = env
    await wd.check(ok(queue_oldest_age=99999))
    await wd.check(ok(queue_oldest_age=99999, paused=True))
    assert len(sent) == 1
    await wd.check(ok())
    assert sent[-1] == w._RECOVERED["queue_stuck"]


async def test_cooldown_and_realert(env):
    clock, sent, wd = env
    bad = ok(disk_free_ratio=0.01)
    await wd.check(bad)
    clock.t += w.ALERT_COOLDOWN - 1
    await wd.check(bad)
    assert len(sent) == 1
    clock.t += 2
    await wd.check(bad)
    assert len(sent) == 2


async def test_kinds_are_independent(env):
    clock, sent, wd = env
    await wd.check(ok(disk_free_ratio=0.01))
    clock.t += 60
    await wd.check(ok(disk_free_ratio=0.01, queue_oldest_age=99999))
    assert len(sent) == 2 and "Очередь" in sent[1]


async def test_recovery_only_after_alert(env):
    _, sent, wd = env
    await wd.check(ok(disk_free_ratio=0.5))
    assert sent == []
    await wd.check(ok(disk_free_ratio=0.01))
    await wd.check(ok(disk_free_ratio=0.5))
    assert len(sent) == 2 and sent[1] == w._RECOVERED["disk_low"]
    await wd.check(ok(disk_free_ratio=0.5))
    assert len(sent) == 2


async def test_notify_failure_is_retried():
    clock, sent, fail = Clock(), [], [True]

    async def notify(text):
        if fail[0]:
            raise RuntimeError("telegram down")
        sent.append(text)

    wd = Watchdog(notify, clock)
    bad = ok(disk_free_ratio=0.01)
    await wd.check(bad)  # must not raise
    assert sent == []
    fail[0] = False
    clock.t += 60  # well inside the cooldown, yet it retries
    await wd.check(bad)
    assert len(sent) == 1


async def test_alerts_contain_no_secrets(env):
    _, sent, wd = env
    await wd.check(ok(disk_free_ratio=0.01, queue_oldest_age=99999))
    assert all("http" not in s and "@" not in s for s in sent)


async def test_run_survives_probe_errors_and_stops_on_cancel(env):
    _, sent, wd = env
    calls = []

    async def probe_fn():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("probe broke")
        return ok(disk_free_ratio=0.01)

    async def sleep(_):
        if len(calls) >= 3:
            raise asyncio.CancelledError
        await asyncio.sleep(0)

    with pytest.raises(asyncio.CancelledError):
        await wd.run(probe_fn, interval=0, sleep=sleep)
    assert len(calls) == 3 and len(sent) == 1


def test_crash_marker(tmp_path):
    marker = tmp_path / "new" / "dir" / "running"  # directory is created
    assert w.was_unclean_shutdown(marker) is False
    assert w.mark_running(marker) is False  # first run is clean
    assert w.was_unclean_shutdown(marker) is True
    assert stat.S_IMODE(marker.stat().st_mode) == 0o600
    assert w.mark_running(marker) is True  # previous run never shut down cleanly
    w.mark_clean_shutdown(marker)
    assert not marker.exists()
    w.mark_clean_shutdown(marker)  # idempotent
    assert w.mark_running(marker) is False
    assert [p.name for p in marker.parent.iterdir()] == ["running"]  # no temp leftovers


def test_unclean_restart_message_is_russian():
    assert w.unclean_restart_message().startswith("Куни перезапустилась после аварийной остановки")
