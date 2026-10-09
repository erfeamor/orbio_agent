"""Text interface: chat with the advisor in the terminal."""

import argparse
import sys

from agent.bootstrap import build_engine
from agent.config import Settings
from agent.domain.models import Phase
from agent.engine.engine import ConversationNotFoundError
from agent.llm.base import LLMConfigurationError
from agent.logging_config import configure_logging

EXIT_WORDS = {"exit", "quit", "/exit", "/quit"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat with the Arcanum Academy advisor.")
    parser.add_argument("--resume", metavar="ID", help="continue a saved conversation")
    parser.add_argument("-v", "--verbose", action="store_true", help="show LLM call logs")
    args = parser.parse_args(argv)

    settings = Settings()
    configure_logging("INFO" if args.verbose else "WARNING", settings.log_format)
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
