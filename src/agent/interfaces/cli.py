"""Text interface: chat with the advisor in the terminal."""

import argparse
import logging
import sys

from agent.config import Settings
from agent.domain.catalog import load_catalog
from agent.domain.models import Phase
from agent.engine.engine import ConversationNotFoundError, Engine
from agent.llm.base import LLMConfigurationError
from agent.llm.claude import ClaudeLLM
from agent.storage.file_repo import JsonFileRepository

EXIT_WORDS = {"exit", "quit", "/exit", "/quit"}


def build_engine(settings: Settings) -> Engine:
    # An empty ANTHROPIC_API_KEY= line in .env means "not set".
    api_key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else ""
    llm = ClaudeLLM(
        settings.anthropic_model,
        effort=settings.llm_effort,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        api_key=api_key or None,
    )
    return Engine(
        llm, JsonFileRepository(settings.conversations_dir), load_catalog(settings.catalog_path)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat with the Arcanum Academy advisor.")
    parser.add_argument("--resume", metavar="ID", help="continue a saved conversation")
    parser.add_argument("-v", "--verbose", action="store_true", help="show LLM call logs")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    settings = Settings()
    engine = build_engine(settings)

    if args.resume:
        conversation_id = args.resume
        print(f"(resuming {conversation_id})")
    else:
        result = engine.start()
        conversation_id = result.record.id
        print(f"advisor> {result.reply}")

    print("(type 'exit' to leave)")
    while True:
        try:
            text = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.strip().lower() in EXIT_WORDS:
            break
        try:
            result = engine.handle_turn(conversation_id, text)
        except ConversationNotFoundError:
            print(f"error: no conversation with id {conversation_id}", file=sys.stderr)
            return 1
        except LLMConfigurationError as exc:
            print(f"error: {exc}\nCheck ANTHROPIC_API_KEY in .env.", file=sys.stderr)
            return 1
        print(f"advisor> {result.reply}")
        if args.verbose:
            record = result.record
            profile = record.profile.model_dump(mode="json", exclude_none=True)
            print(f"  [phase={record.phase.value} profile={profile} errors={record.field_errors}]")
        if result.record.phase is Phase.DONE and result.record.summary:
            print("(interview complete; you can ask follow-up questions or type 'exit')")

    path = settings.conversations_dir / f"{conversation_id}.json"
    print(f"Conversation saved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
