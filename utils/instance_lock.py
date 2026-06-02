"""Ensure only one dream-maker process runs at a time."""

from __future__ import annotations

import atexit
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("dream_maker.lock")

_LOCK: "InstanceLock | None" = None


class InstanceLock:
    """Advisory file lock — released automatically on process exit."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "a+", encoding="utf-8")

    def acquire(self) -> bool:
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            existing = self._read_pid()
            if existing and _pid_alive(existing):
                log.error(
                    "Another dream-maker instance is running (pid %s, lock %s)",
                    existing,
                    self.path,
                )
                return False
            log.warning("Stale lock detected (pid %s dead) — taking over", existing)
            return self._steal_lock()

        self._write_pid()
        return True

    def _steal_lock(self) -> bool:
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            return False
        self._write_pid()
        return True

    def _write_pid(self) -> None:
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(f"{os.getpid()}\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def _read_pid(self) -> int | None:
        try:
            self._handle.seek(0)
            raw = self._handle.read().strip()
            return int(raw) if raw else None
        except (ValueError, OSError):
            return None

    def release(self) -> None:
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            self._handle.seek(0)
            self._handle.truncate()
            self._handle.flush()
            os.fsync(self._handle.fileno())
        except OSError:
            pass
        try:
            self._handle.close()
        except OSError:
            pass


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    else:
        return True


def acquire_instance_lock(path: Path) -> InstanceLock | None:
    global _LOCK
    lock = InstanceLock(path)
    if not lock.acquire():
        lock.release()
        return None
    _LOCK = lock
    atexit.register(release_instance_lock)
    log.info("Instance lock acquired: %s (pid %s)", path, os.getpid())
    return lock


def release_instance_lock() -> None:
    global _LOCK
    if _LOCK is None:
        return
    _LOCK.release()
    _LOCK = None


def is_instance_running(path: Path) -> bool:
    """Return True if another process holds the instance lock."""
    if not path.exists():
        return False
    try:
        handle = open(path, "a+", encoding="utf-8")
    except OSError:
        return False
    try:
        if sys.platform == "win32":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                return True
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            return False

        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False
    finally:
        try:
            handle.close()
        except OSError:
            pass
