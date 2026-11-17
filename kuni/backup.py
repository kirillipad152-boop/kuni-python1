"""Private, daily recovery archives; independent of the bot and its configuration loader.

Run: python -m kuni.backup --source /path/to/kuni-py --destination /private/backups/kuni
Stdout is one JSON object containing the archive path, byte size and SHA-256.
Archives contain credentials, including Telegram sessions; they are not encrypted.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import sys
import tarfile
import tempfile
import time


ARCHIVE_NAME = re.compile(r"kuni-backup-[0-9]{8}T[0-9]{6}Z\.tar\.gz\Z")
_ATOMIC_TEMP = re.compile(r"\..+\.(?:md|json)\.[a-z0-9_]{8}\Z")
_SQLITE_SUFFIXES = {".sqlite", ".sqlite3", ".db", ".session"}
_EXCLUDED_DIRS = {"logs", "cache", ".cache", "venv", ".venv", "__pycache__"}


def _snapshot_database(source: Path, target: Path) -> None:
    deadline = time.monotonic() + 120

    def progress(status: int, remaining: int, total: int) -> None:
        if time.monotonic() > deadline:
            raise TimeoutError("SQLite backup timed out")

    # mode=ro (not immutable=1) sees committed WAL pages without creating a missing DB.
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as reader:
        with closing(sqlite3.connect(target)) as writer:
            reader.backup(writer, pages=256, progress=progress, sleep=0.05)
            writer.execute("PRAGMA journal_mode=DELETE")
            if writer.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise ValueError("SQLite snapshot failed quick_check")
    target.chmod(0o600)


def _copy(source: Path, target: Path) -> None:
    # Ignore atomic_write_text scratch files and SQLite sidecars before they can disappear.
    if (_ATOMIC_TEMP.fullmatch(source.name)
            or any(source.name.endswith(suffix + sidecar)
                   for suffix in _SQLITE_SUFFIXES for sidecar in ("-wal", "-shm", "-journal"))):
        return
    if source.is_symlink():
        raise ValueError("Backup sources must not contain symlinks")
    if source.is_dir():
        target.mkdir(mode=0o700)
        for child in sorted(source.iterdir()):
            if child.name not in _EXCLUDED_DIRS:
                _copy(child, target / child.name)
    elif source.is_file():
        if source.suffix in _SQLITE_SUFFIXES:
            _snapshot_database(source, target)
        else:
            shutil.copyfile(source, target)
            target.chmod(0o600)
    else:
        raise ValueError("Missing or non-regular backup source")


def _archives(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir()
                  if ARCHIVE_NAME.fullmatch(path.name) and not path.is_symlink() and path.is_file())


def _verify_archive(path: Path) -> None:
    # Read through the gzip trailer too: tar iteration alone need not check its CRC.
    with gzip.open(path, "rb") as stream:
        while stream.read(1024 * 1024):
            pass
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = {member.name for member in members}
        if not {"data/memory/graph.sqlite", "config.toml", "prompts"} <= names:
            raise ValueError("Incomplete recovery archive")
        if len(names) != len(members):
            raise ValueError("Duplicate archive members")
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or not (member.isfile() or member.isdir()):
                raise ValueError("Unsafe archive member")
            if member.isfile():
                with archive.extractfile(member) as stream:
                    while stream.read(1024 * 1024):
                        pass


def _private_member(member: tarfile.TarInfo) -> tarfile.TarInfo:
    member.mode = 0o700 if member.isdir() else 0o600
    member.uid = member.gid = 0
    member.uname = member.gname = ""
    return member


def create_backup(source: str | Path, destination: str | Path, force: bool = False) -> Path:
    """Create (or verify and reuse) today's UTC backup, retaining 14 archives; `force` always makes a fresh one.

    shortcut: each DB/file is a separate snapshot; stop the bot for an app-wide transaction.
    In-RAM context that the bot has not persisted cannot be recovered from this archive.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError("Source and backup directories must be separate trees")
    graph = source / "data/memory/graph.sqlite"
    if graph.is_symlink() or not graph.is_file() or graph.stat().st_size == 0:
        raise ValueError("Existing graph.sqlite is required")
    if not (source / "config.toml").is_file() or not (source / "prompts").is_dir():
        raise ValueError("Existing config.toml and prompts are required")

    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination.chmod(0o700)
    fd = os.open(destination / ".backup.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a+b") as lock:
        os.fchmod(lock.fileno(), 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        now = datetime.now(timezone.utc)
        today = f"kuni-backup-{now:%Y%m%d}T"
        existing = [] if force else [path for path in _archives(destination) if path.name.startswith(today)]
        if existing:
            archive_path = existing[-1]
            _verify_archive(archive_path)
            archive_path.chmod(0o600)
        else:
            with tempfile.TemporaryDirectory(prefix=".kuni-backup-", dir=destination) as temporary:
                stage = Path(temporary)
                snapshot = stage / "snapshot"
                snapshot.mkdir(mode=0o700)
                for name in ("data", "config.toml", "prompts", "working_memory.md", "tg_session"):
                    path = source / name
                    if name in ("working_memory.md", "tg_session") and not path.exists() and not path.is_symlink():
                        continue
                    _copy(path, snapshot / name)
                temporary_archive = stage / "archive.tar.gz"
                with temporary_archive.open("xb") as stream:
                    os.fchmod(stream.fileno(), 0o600)
                    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
                        for path in sorted(snapshot.iterdir()):
                            archive.add(path, arcname=path.name, filter=_private_member)
                    stream.flush()
                    os.fsync(stream.fileno())
                _verify_archive(temporary_archive)
                archive_path = destination / f"kuni-backup-{now:%Y%m%dT%H%M%SZ}.tar.gz"
                # Publish atomically without ever overwriting an existing archive.
                os.link(temporary_archive, archive_path)
                directory_fd = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        # A clock correction must not make us prune the archive we are returning.
        previous = [path for path in _archives(destination) if path != archive_path]
        for old in previous[:-13]:
            old.unlink()
        return archive_path


def archive_info(path: Path) -> dict[str, str | int]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "size": path.stat().st_size, "sha256": digest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        path = create_backup(args.source, args.destination)
        print(json.dumps(archive_info(path)))
    except Exception as exc:
        # Never echo configuration, session data or subprocess output into scheduler logs.
        print(f"kuni backup failed ({type(exc).__name__})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
