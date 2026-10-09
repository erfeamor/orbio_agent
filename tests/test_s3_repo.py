"""S3-specific behaviour; shared repository behaviour is in test_repositories.py."""

import json
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import boto3
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError
from moto import mock_aws

from agent.domain.models import ConversationRecord
from agent.storage.base import StorageUnavailableError, VersionConflictError
from agent.storage.s3_repo import S3ConversationRepository

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

BUCKET = "test-conversations"
PREFIX = "conversations/"


@pytest.fixture
def s3() -> Iterator["S3Client"]:
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


@pytest.fixture
def repo(s3: "S3Client") -> S3ConversationRepository:
    return S3ConversationRepository(client=s3, bucket=BUCKET, prefix=PREFIX)


def key(record: ConversationRecord) -> str:
    return f"{PREFIX}{record.id}.json"


def test_object_is_json_with_version_metadata_and_encryption(
    repo: S3ConversationRepository, s3: "S3Client"
) -> None:
    saved = repo.save(ConversationRecord())

    head = s3.head_object(Bucket=BUCKET, Key=key(saved))
    body = s3.get_object(Bucket=BUCKET, Key=key(saved))["Body"].read()

    assert head["ContentType"] == "application/json"
    assert head["Metadata"] == {"version": "1"}
    assert head["ServerSideEncryption"] == "AES256"
    assert json.loads(body)["id"] == saved.id


def test_write_between_check_and_save_is_a_conflict(
    repo: S3ConversationRepository, s3: "S3Client", monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = repo.save(ConversationRecord())
    real_head = s3.head_object

    def head_then_concurrent_write(**kwargs: Any) -> Any:
        # Version check passes, then another writer replaces the object (new ETag).
        result = real_head(**kwargs)
        s3.put_object(Bucket=BUCKET, Key=key(saved), Body=b"{}", Metadata={"version": "1"})
        return result

    monkeypatch.setattr(s3, "head_object", head_then_concurrent_write)

    with pytest.raises(VersionConflictError, match="no longer at the expected version"):
        repo.save(saved)


@pytest.mark.parametrize("metadata", [{}, {"version": "not-a-number"}], ids=["missing", "bad"])
def test_object_without_a_readable_version_is_a_conflict(
    repo: S3ConversationRepository, s3: "S3Client", metadata: dict[str, str]
) -> None:
    saved = repo.save(ConversationRecord())
    s3.put_object(Bucket=BUCKET, Key=key(saved), Body=saved.model_dump_json(), Metadata=metadata)

    with pytest.raises(VersionConflictError):
        repo.save(saved)


def test_updating_a_deleted_conversation_is_a_conflict(
    repo: S3ConversationRepository, s3: "S3Client"
) -> None:
    saved = repo.save(ConversationRecord())
    s3.delete_object(Bucket=BUCKET, Key=key(saved))

    with pytest.raises(VersionConflictError):
        repo.save(saved)


def test_prefix_is_part_of_the_key(s3: "S3Client") -> None:
    repo = S3ConversationRepository(client=s3, bucket=BUCKET, prefix="tenant-a/")
    saved = repo.save(ConversationRecord())
    s3.head_object(Bucket=BUCKET, Key=f"tenant-a/{saved.id}.json")  # raises if missing


def access_denied(operation: str) -> ClientError:
    return ClientError({"Error": {"Code": "AccessDenied", "Message": "no"}}, operation)


@pytest.mark.parametrize(
    "error",
    [access_denied("GetObject"), EndpointConnectionError(endpoint_url="https://s3")],
    ids=["client-error", "network-error"],
)
def test_read_failures_become_storage_unavailable(
    repo: S3ConversationRepository,
    s3: "S3Client",
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    def fail(**kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(s3, "get_object", fail)
    with pytest.raises(StorageUnavailableError, match="cannot read"):
        repo.get(ConversationRecord().id)


@pytest.mark.parametrize(
    "error",
    [access_denied("PutObject"), EndpointConnectionError(endpoint_url="https://s3")],
    ids=["client-error", "network-error"],
)
def test_write_failures_become_storage_unavailable(
    repo: S3ConversationRepository,
    s3: "S3Client",
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    def fail(**kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(s3, "put_object", fail)
    with pytest.raises(StorageUnavailableError, match="cannot write"):
        repo.save(ConversationRecord())


def test_failed_version_check_becomes_storage_unavailable(
    repo: S3ConversationRepository, s3: "S3Client", monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = repo.save(ConversationRecord())

    def fail(**kwargs: Any) -> Any:
        raise access_denied("HeadObject")

    monkeypatch.setattr(s3, "head_object", fail)
    with pytest.raises(StorageUnavailableError, match="cannot write"):
        repo.save(saved)
