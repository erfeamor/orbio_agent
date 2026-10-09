from pathlib import Path

import pytest

from agent.domain.models import ConversationRecord
from agent.storage.base import VersionConflictError
from agent.storage.file_repo import JsonFileRepository


@pytest.fixture
def repo(tmp_path: Path) -> JsonFileRepository:
    return JsonFileRepository(tmp_path / "conversations")


def test_save_then_get_returns_the_record(repo: JsonFileRepository) -> None:
    record = ConversationRecord()
    record.add_message("agent", "Welcome!")

    saved = repo.save(record)
    loaded = repo.get(record.id)

    assert saved.version == 1
    assert loaded == saved


def test_get_unknown_id_returns_none(repo: JsonFileRepository) -> None:
    assert repo.get(ConversationRecord().id) is None


def test_saving_a_stale_copy_raises_conflict(repo: JsonFileRepository) -> None:
    first = repo.save(ConversationRecord())
    repo.save(first)  # someone else saves version 1 -> 2

    with pytest.raises(VersionConflictError):
        repo.save(first)  # still at version 1


def test_sequential_saves_increment_the_version(repo: JsonFileRepository) -> None:
    record = repo.save(ConversationRecord())
    record = repo.save(record)
    assert record.version == 2
    assert repo.get(record.id) == record


@pytest.mark.parametrize("bad_id", ["../etc/passwd", "abc", "A" * 32, ""])
def test_invalid_ids_are_rejected(repo: JsonFileRepository, bad_id: str) -> None:
    with pytest.raises(ValueError, match="invalid conversation id"):
        repo.get(bad_id)


def test_no_temp_files_are_left_behind(repo: JsonFileRepository, tmp_path: Path) -> None:
    repo.save(ConversationRecord())
    files = list((tmp_path / "conversations").iterdir())
    assert len(files) == 1
    assert files[0].suffix == ".json"


def test_failed_write_keeps_the_previous_version(
    repo: JsonFileRepository, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = repo.save(ConversationRecord())

    def broken_replace(src: str, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("agent.storage.file_repo.os.replace", broken_replace)
    with pytest.raises(OSError, match="disk full"):
        repo.save(saved)

    assert repo.get(saved.id) == saved
    assert [p.suffix for p in (tmp_path / "conversations").iterdir()] == [".json"]
