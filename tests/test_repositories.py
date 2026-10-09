"""Behaviour every ConversationRepository must have, run against each implementation."""

from collections.abc import Iterator
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from agent.domain.models import ConversationRecord
from agent.storage.base import ConversationRepository, VersionConflictError
from agent.storage.file_repo import JsonFileRepository
from agent.storage.s3_repo import S3ConversationRepository

BUCKET = "test-conversations"


@pytest.fixture(params=["file", "s3"])
def repo(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[ConversationRepository]:
    if request.param == "file":
        yield JsonFileRepository(tmp_path / "conversations")
        return
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield S3ConversationRepository(client=client, bucket=BUCKET, prefix="conversations/")


def test_save_then_get_returns_the_record(repo: ConversationRepository) -> None:
    record = ConversationRecord()
    record.add_message("agent", "Welcome!")

    saved = repo.save(record)

    assert saved.version == 1
    assert repo.get(record.id) == saved


def test_get_unknown_id_returns_none(repo: ConversationRepository) -> None:
    assert repo.get(ConversationRecord().id) is None


def test_sequential_saves_increment_the_version(repo: ConversationRepository) -> None:
    record = repo.save(ConversationRecord())
    record = repo.save(record)
    assert record.version == 2
    assert repo.get(record.id) == record


def test_saving_a_stale_copy_raises_conflict(repo: ConversationRepository) -> None:
    first = repo.save(ConversationRecord())
    repo.save(first)  # someone else saves version 1 -> 2

    with pytest.raises(VersionConflictError, match="at version 2"):
        repo.save(first)  # still at version 1

    assert repo.get(first.id).version == 2  # type: ignore[union-attr]


def test_creating_an_existing_conversation_again_raises_conflict(
    repo: ConversationRepository,
) -> None:
    record = ConversationRecord()
    repo.save(record)

    with pytest.raises(VersionConflictError):
        repo.save(record)  # version 0 again: a second "create"


@pytest.mark.parametrize("bad_id", ["../etc/passwd", "abc", "A" * 32, ""])
def test_invalid_ids_are_rejected(repo: ConversationRepository, bad_id: str) -> None:
    with pytest.raises(ValueError, match="invalid conversation id"):
        repo.get(bad_id)
