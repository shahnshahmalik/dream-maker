"""Single-instance lock tests."""

import os
from pathlib import Path

from utils.instance_lock import InstanceLock, acquire_instance_lock, is_instance_running, release_instance_lock


def test_second_acquire_fails_while_running(tmp_path):
    lock_path = tmp_path / "dream-maker.lock"
    first = acquire_instance_lock(lock_path)
    assert first is not None
    assert is_instance_running(lock_path)

    second = InstanceLock(lock_path)
    assert second.acquire() is False
    second.release()

    release_instance_lock()
    assert not is_instance_running(lock_path)


def test_stale_lock_can_be_replaced(tmp_path, monkeypatch):
    lock_path = tmp_path / "dream-maker.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("999999\n", encoding="utf-8")

    lock = acquire_instance_lock(lock_path)
    assert lock is not None
    assert lock_path.read_text(encoding="utf-8").strip() == str(os.getpid())
    release_instance_lock()
