"""File-specific behaviour; shared repository behaviour is in test_repositories.py."""

from pathlib import Path

import pytest

from agent.domain.models import ConversationRecord
from agent.storage.file_repo import JsonFileRepository


@pytest.fixture
def repo(tmp_path: Path) -> JsonFileRepository:
    return JsonFileRepository(tmp_path / "conversations")


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
