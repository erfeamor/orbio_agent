"""Conversation storage in S3, one JSON object per conversation (AWS deployment).

Concurrency without locks: each object carries its record version in its metadata, and
writes are conditional. Creating uses ``IfNoneMatch="*"`` (fails if the object exists);
updating checks the stored version, then writes with ``IfMatch=<ETag>`` (fails if anyone
wrote in between). Both failures become ``VersionConflictError``, as in the file repository.
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

from botocore.exceptions import BotoCoreError, ClientError

from agent.domain.models import ConversationRecord
from agent.storage.base import (
    ConversationRepository,
    StorageUnavailableError,
    VersionConflictError,
    validate_conversation_id,
)

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client
    from mypy_boto3_s3.type_defs import HeadObjectOutputTypeDef

_VERSION_METADATA = "version"
_CONTENT_TYPE = "application/json"
_ENCRYPTION: Final = "AES256"
_NOT_FOUND_CODES = frozenset({"404", "NoSuchKey", "NotFound"})
_CONDITION_FAILED_CODES = frozenset({"PreconditionFailed", "ConditionalRequestConflict"})


class S3ConversationRepository(ConversationRepository):
    def __init__(self, *, client: "S3Client", bucket: str, prefix: str) -> None:
        self._s3 = client
        self._bucket = bucket
        self._prefix = prefix

    def get(self, conversation_id: str) -> ConversationRecord | None:
        key = self._key(conversation_id)
        try:
            obj = self._s3.get_object(Bucket=self._bucket, Key=key)
            body = obj["Body"].read()
        except ClientError as exc:
            if _error_code(exc) in _NOT_FOUND_CODES:
                return None
            raise StorageUnavailableError(f"cannot read {key}: {_error_code(exc)}") from exc
        except BotoCoreError as exc:
            raise StorageUnavailableError(f"cannot read {key}: {exc}") from exc
        return ConversationRecord.model_validate_json(body)

    def save(self, record: ConversationRecord) -> ConversationRecord:
        key = self._key(record.id)
        saved = record.model_copy(
            update={"version": record.version + 1, "updated_at": datetime.now(UTC)}
        )
        body = saved.model_dump_json().encode("utf-8")
        metadata = {_VERSION_METADATA: str(saved.version)}
        try:
            if record.version == 0:
                # New conversation: fails if the object already exists.
                self._s3.put_object(
                    Bucket=self._bucket,
                    Key=key,
                    Body=body,
                    ContentType=_CONTENT_TYPE,
                    Metadata=metadata,
                    ServerSideEncryption=_ENCRYPTION,
                    IfNoneMatch="*",
                )
            else:
                head = self._head(key)
                stored_version = _stored_version(head)
                if head is None or stored_version != record.version:
                    raise VersionConflictError.for_record(record, stored_version)
                # Fails if someone wrote between the head request and this write.
                self._s3.put_object(
                    Bucket=self._bucket,
                    Key=key,
                    Body=body,
                    ContentType=_CONTENT_TYPE,
                    Metadata=metadata,
                    ServerSideEncryption=_ENCRYPTION,
                    IfMatch=head["ETag"],
                )
        except ClientError as exc:
            if _error_code(exc) in _CONDITION_FAILED_CODES:
                raise VersionConflictError.for_record(record, None) from exc
            raise StorageUnavailableError(f"cannot write {key}: {_error_code(exc)}") from exc
        except BotoCoreError as exc:
            raise StorageUnavailableError(f"cannot write {key}: {exc}") from exc
        return saved

    def _key(self, conversation_id: str) -> str:
        return f"{self._prefix}{validate_conversation_id(conversation_id)}.json"

    def _head(self, key: str) -> "HeadObjectOutputTypeDef | None":
        try:
            return self._s3.head_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            if _error_code(exc) in _NOT_FOUND_CODES:
                return None
            raise


def _error_code(exc: ClientError) -> str:
    return str(exc.response.get("Error", {}).get("Code", ""))


def _stored_version(head: "HeadObjectOutputTypeDef | None") -> int | None:
    """The version in the object's metadata; None if missing or unreadable (written elsewhere)."""
    if head is None:
        return None
    try:
        return int(head.get("Metadata", {})[_VERSION_METADATA])
    except (KeyError, ValueError):
        return None
