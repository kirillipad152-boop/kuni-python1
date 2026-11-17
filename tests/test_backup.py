from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tarfile
import threading

import pytest

from kuni import backup


PULL_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/kuni-backup-pull"


@pytest.fixture
def backup_source(tmp_path):
    root = tmp_path / "bot"
    (root / "data/memory").mkdir(parents=True)
    (root / "data/chats/123").mkdir(parents=True)
    (root / "data/chats/123/working_memory.md").write_text("per-chat memory")
    (root / "data/working_memory").mkdir()
    (root / "data/working_memory/-123.md").write_text("per-chat atomic memory")
    (root / "data/contexts").mkdir()
    (root / "data/contexts/-123.json").write_text('{"chat_id": -123, "session_id": "123", "messages": []}')
    (root / "data/contexts/.-123.json.abcdefgh").write_text("incomplete temp file")
    (root / "data/gallery").mkdir()
    (root / "data/gallery/audio.opus").write_bytes(b"audio data")
    (root / "data/working_memory.md").write_text("shared memory")
    (root / "data/reminders.json").write_text('{"reminders": [{"id":"r123456", "chat_id":42, "note":"private"}]}')
    (root / "working_memory.md").write_text("legacy memory")
    (root / "config.toml").write_text('telegramApiHash = "private-config"\n')
    (root / "prompts").mkdir()
    (root / "prompts/system.md").write_text("private prompt")
    (root / "tg_session").mkdir()
    for directory in ("logs", "cache", ".venv", "data/cache"):
        (root / directory).mkdir(exist_ok=True)
        (root / directory / "excluded").write_text("not backed up")
    connections = []
    for relative in ("data/memory/graph.sqlite", "tg_session/kuni.session"):
        connection = sqlite3.connect(root / relative, isolation_level=None)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA wal_autocheckpoint=0")
        connection.execute("CREATE TABLE entries(value TEXT)")
        connection.execute("INSERT INTO entries VALUES ('committed-in-wal')")
        assert (root / (relative + "-wal")).stat().st_size > 0
        connections.append(connection)
    yield root
    for connection in connections:
        connection.close()


def test_restore_live_wal_databases_and_all_memory(backup_source, tmp_path):
    started, stop = threading.Event(), threading.Event()
    errors = []

    def write():
        try:
            with sqlite3.connect(backup_source / "data/memory/graph.sqlite", isolation_level=None) as connection:
                while not stop.is_set():
                    connection.execute("INSERT INTO entries VALUES ('live-writer')")
                    started.set()
                    stop.wait(0.002)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=write)
    thread.start()
    try:
        assert started.wait(5)
        path = backup.create_backup(backup_source, tmp_path / "backups")
    finally:
        stop.set()
        thread.join(timeout=5)
    assert not thread.is_alive() and not errors
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert not list(path.parent.glob(".kuni-backup-*"))

    restore = tmp_path / "restore"
    with tarfile.open(path) as archive:
        names = archive.getnames()
        assert not any(name.endswith(("-wal", "-shm", "-journal")) for name in names)
        assert "data/contexts/.-123.json.abcdefgh" not in names
        assert not any(name in names for name in ("logs", "cache", ".venv", "data/cache"))
        assert all(member.mode == (0o700 if member.isdir() else 0o600) for member in archive.getmembers())
        archive.extractall(restore, filter="data")
    for relative in ("data/memory/graph.sqlite", "tg_session/kuni.session"):
        with sqlite3.connect(restore / relative) as connection:
            assert connection.execute("PRAGMA quick_check").fetchall() == [("ok",)]
            assert ("committed-in-wal",) in connection.execute("SELECT value FROM entries").fetchall()
            assert connection.execute("PRAGMA journal_mode").fetchone() == ("delete",)
    assert (restore / "data/chats/123/working_memory.md").read_text() == "per-chat memory"
    assert (restore / "data/working_memory/-123.md").read_text() == "per-chat atomic memory"
    assert json.loads((restore / "data/contexts/-123.json").read_text())["chat_id"] == -123
    assert (restore / "data/gallery/audio.opus").read_bytes() == b"audio data"
    assert (restore / "data/working_memory.md").read_text() == "shared memory"
    assert (restore / "data/reminders.json").read_bytes() == (backup_source / "data/reminders.json").read_bytes()
    assert (restore / "working_memory.md").read_text() == "legacy memory"
    assert (restore / "config.toml").read_bytes() == (backup_source / "config.toml").read_bytes()
    assert (restore / "prompts/system.md").read_text() == "private prompt"


def test_missing_graph_does_not_create_an_empty_database(backup_source, tmp_path):
    graph = backup_source / "data/memory/graph.sqlite"
    graph.rename(graph.with_suffix(".absent"))
    with pytest.raises(ValueError, match="graph.sqlite"):
        backup.create_backup(backup_source, tmp_path / "backups")
    assert not graph.exists()
    assert not (tmp_path / "backups").exists()


def test_disappearing_scratch_files_and_sidecars_are_not_read(tmp_path):
    for name in ("graph.sqlite-wal", "kuni.session-journal", ".123.md.abcdefgh", ".123.json.abcdefgh"):
        backup._copy(tmp_path / name, tmp_path / "unused")
    assert not (tmp_path / "unused").exists()


def test_corrupt_database_fails_without_publishing(backup_source, tmp_path):
    (backup_source / "data/memory/broken.sqlite").write_bytes(b"not a database")
    destination = tmp_path / "backups"
    with pytest.raises(sqlite3.DatabaseError):
        backup.create_backup(backup_source, destination)
    assert list(destination.iterdir()) == [destination / ".backup.lock"]


def test_staging_failure_cleans_up_without_pruning(backup_source, tmp_path, monkeypatch):
    destination = tmp_path / "backups"
    destination.mkdir()
    old = []
    for day in range(1, 16):
        path = destination / f"kuni-backup-202001{day:02}T000000Z.tar.gz"
        path.write_bytes(b"keep previous backup")
        old.append(path)

    def fail(*args, **kwargs):
        raise OSError("simulated full disk")

    monkeypatch.setattr(backup.shutil, "copyfile", fail)
    with pytest.raises(OSError, match="full disk"):
        backup.create_backup(backup_source, destination)
    assert all(path.read_bytes() == b"keep previous backup" for path in old)
    assert not list(destination.glob(".kuni-backup-*"))
    assert len(backup._archives(destination)) == 15


@pytest.mark.parametrize("year", [2020, 2099])
def test_server_retention_and_daily_reuse(backup_source, tmp_path, monkeypatch, year):
    destination = tmp_path / "backups"
    destination.mkdir()
    old = []
    for day in range(1, 16):
        path = destination / f"kuni-backup-{year}01{day:02}T000000Z.tar.gz"
        path.write_bytes(b"old")
        old.append(path)
    unrelated = destination / "other-backup.tar.gz"
    unrelated.write_bytes(b"unrelated")
    symlink = destination / "kuni-backup-20190101T000000Z.tar.gz"
    symlink.symlink_to(unrelated)
    path = backup.create_backup(backup_source, destination)
    assert backup._archives(destination) == sorted(old[2:] + [path])
    assert unrelated.read_bytes() == b"unrelated" and symlink.is_symlink()

    def fail(*args):
        raise AssertionError("Today's valid backup must be reused")

    monkeypatch.setattr(backup, "_snapshot_database", fail)
    assert backup.create_backup(backup_source, destination) == path
    info = backup.archive_info(path)
    assert info["size"] == path.stat().st_size
    assert info["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_bytes(b"corrupt gzip")
    with pytest.raises(OSError):
        backup.create_backup(backup_source, destination)
    assert path.read_bytes() == b"corrupt gzip"


def test_source_destination_and_symlink_validation(backup_source, tmp_path):
    with pytest.raises(ValueError, match="separate trees"):
        backup.create_backup(backup_source, backup_source / "backups")
    with pytest.raises(ValueError, match="separate trees"):
        backup.create_backup(backup_source, tmp_path)
    (backup_source / "data/escape").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="symlinks"):
        backup.create_backup(backup_source, tmp_path / "backups")
    assert not backup._archives(tmp_path / "backups")


def test_server_lock_prevents_concurrent_run(backup_source, tmp_path):
    destination = tmp_path / "backups"
    destination.mkdir()
    with (destination / ".backup.lock").open("wb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            backup.create_backup(backup_source, destination)


def test_module_cli_outputs_metadata_without_secrets(backup_source, tmp_path, capsys):
    # -S ensures this CLI does not depend on the bot's third-party packages or startup hooks.
    result = subprocess.run([sys.executable, "-S", "-m", "kuni.backup", "--source", str(backup_source),
                             "--destination", str(tmp_path / "backups")],
                            env={**os.environ, "PYTHONPATH": str(PULL_SCRIPT.parents[1])},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert not result.stderr and "private-config" not in result.stdout
    assert Path(json.loads(result.stdout)["path"]).is_file()
    assert backup.main(["--source", str(tmp_path / "missing"), "--destination", str(tmp_path / "backups")]) == 1
    captured = capsys.readouterr()
    assert not captured.out and "failed" in captured.err


@pytest.fixture
def pull_setup(tmp_path, monkeypatch):
    # Fake only the transport; exercise the actual script's argument parsing and filesystem writes.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    transport = bin_dir / "transport"
    transport.write_text(f"""#!{sys.executable}
import json, os, pathlib, shutil, sys
with open(os.environ['TRANSPORT_LOG'], 'a') as stream:
    stream.write(json.dumps([pathlib.Path(sys.argv[0]).name, *sys.argv[1:]]) + '\\n')
if pathlib.Path(sys.argv[0]).name == 'ssh':
    if os.environ.get('SSH_FAIL'):
        sys.exit(255)
    print(os.environ['REMOTE_INFO'])
else:
    shutil.copyfile(os.environ['REMOTE_PAYLOAD'], sys.argv[-1])
    if os.environ.get('SCP_FAIL'):
        sys.exit(1)
""")
    transport.chmod(0o700)
    for name in ("ssh", "scp"):
        (bin_dir / name).symlink_to(transport)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    log = tmp_path / "transport.jsonl"
    monkeypatch.setenv("TRANSPORT_LOG", str(log))
    payload = tmp_path / "payload"
    payload.write_bytes(b"verified archive transport payload")
    name = "kuni-backup-20261116T050000Z.tar.gz"
    info = {"path": "/srv/backups/kuni/" + name, "size": payload.stat().st_size,
            "sha256": hashlib.sha256(payload.read_bytes()).hexdigest()}
    monkeypatch.setenv("REMOTE_INFO", json.dumps(info))
    monkeypatch.setenv("REMOTE_PAYLOAD", str(payload))
    key = tmp_path / "key with spaces"
    key.write_text("fake private key")
    destination = tmp_path / "local backups"
    args = [sys.executable, str(PULL_SCRIPT), "--host", "bot@server", "--key", str(key),
            "--remote-root", "/srv/bot with spaces", "--remote-dir", "/srv/backups/kuni",
            "--local-dir", str(destination)]
    return args, destination, payload, info, key, log


def test_pull_verifies_atomic_download_and_reuses_it(pull_setup):
    args, destination, payload, info, key, log = pull_setup
    for _ in range(2):
        result = subprocess.run(args, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
    target = destination / Path(info["path"]).name
    assert target.read_bytes() == payload.read_bytes()
    assert target.stat().st_mode & 0o777 == 0o600
    assert destination.stat().st_mode & 0o777 == 0o700
    assert not list(destination.glob(".kuni-pull-*"))
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    assert [command[0] for command in commands] == ["ssh", "scp", "ssh"]
    for command in commands:
        assert command[command.index("-i") + 1] == str(key)
        assert {"IdentitiesOnly=yes", "StrictHostKeyChecking=yes", "BatchMode=yes"} <= set(command)
    assert "cd -- '/srv/bot with spaces'" in commands[0][-1]
    assert commands[1][-1].endswith(".partial")


@pytest.mark.parametrize("failure", ["SSH_FAIL", "SCP_FAIL", "checksum", "size", "remote_path", "boolean_size"])
def test_pull_failure_retains_old_backups_and_cleans_partial(pull_setup, monkeypatch, failure):
    args, destination, payload, info, key, log = pull_setup
    destination.mkdir()
    old = destination / "kuni-backup-20200101T000000Z.tar.gz"
    old.write_bytes(b"previous successful backup")
    if failure in ("SSH_FAIL", "SCP_FAIL"):
        monkeypatch.setenv(failure, "1")
    elif failure == "checksum":
        info["sha256"] = "0" * 64
    elif failure == "size":
        info["size"] += 1
    elif failure == "boolean_size":
        info["size"] = True
    else:
        info["path"] = "/srv/backups/kuni/../other/" + Path(info["path"]).name
    monkeypatch.setenv("REMOTE_INFO", json.dumps(info))
    result = subprocess.run(args, capture_output=True, text=True, timeout=10)
    assert result.returncode == 1 and not result.stdout
    assert old.read_bytes() == b"previous successful backup"
    assert list(destination.glob("*.tar.gz")) == [old]
    assert not list(destination.glob(".kuni-pull-*"))


@pytest.mark.parametrize("year", [2020, 2099])
def test_pull_local_retention_never_mirrors_remote_deletion(pull_setup, year):
    args, destination, payload, info, key, log = pull_setup
    destination.mkdir()
    old = []
    for day in range(1, 32):
        path = destination / f"kuni-backup-{year}01{day:02}T000000Z.tar.gz"
        path.write_bytes(b"old local backup")
        old.append(path)
    unrelated = destination / "manual.tar.gz"
    unrelated.write_bytes(b"keep")
    result = subprocess.run(args, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert not old[0].exists() and not old[1].exists()
    assert all(path.exists() for path in old[2:])
    assert unrelated.read_bytes() == b"keep"
    assert len(list(destination.glob("kuni-backup-*.tar.gz"))) == 30


def test_pull_rejects_unsafe_server_directory_and_tilde(pull_setup):
    args, destination, payload, info, key, log = pull_setup
    for unsafe in ("~/backups", "/srv/backups/$(id)", "/srv/backups/../other"):
        bad_args = args.copy()
        bad_args[bad_args.index("--remote-dir") + 1] = unsafe
        result = subprocess.run(bad_args, capture_output=True, text=True, timeout=10)
        assert result.returncode == 1 and not result.stdout
    assert not log.exists()


def test_pull_rejects_filesystem_root_destination(pull_setup):
    args, destination, payload, info, key, log = pull_setup
    args[args.index("--local-dir") + 1] = "/"
    result = subprocess.run(args, capture_output=True, text=True, timeout=10)
    assert result.returncode == 1 and not result.stdout
    assert not log.exists()


def test_pull_lock_prevents_concurrent_run(pull_setup):
    args, destination, payload, info, key, log = pull_setup
    destination.mkdir()
    with (destination / ".pull.lock").open("wb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(args, capture_output=True, text=True, timeout=10)
        assert result.returncode == 1 and not result.stdout
    assert not log.exists()


def test_pull_environment_configuration(pull_setup, monkeypatch):
    args, destination, payload, info, key, log = pull_setup
    for flag in ("host", "key", "remote-root", "remote-dir", "local-dir"):
        monkeypatch.setenv("KUNI_BACKUP_" + flag.replace("-", "_").upper(), args[args.index("--" + flag) + 1])
    result = subprocess.run(args[:2], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
