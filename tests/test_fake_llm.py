import pytest

from agent.engine.schemas import ReplyOutput, SummaryOutput
from agent.llm.base import ChatMessage
from agent.llm.fake import FakeLLM

MESSAGES = [ChatMessage("user", "hi")]


def test_returns_queued_responses_in_order_and_records_calls() -> None:
    llm = FakeLLM([ReplyOutput(reply="one"), ReplyOutput(reply="two")])

    first = llm.generate(system="s1", messages=MESSAGES, output_type=ReplyOutput)
    second = llm.generate(system="s2", messages=MESSAGES, output_type=ReplyOutput)

    assert (first.reply, second.reply) == ("one", "two")
    assert [c.system for c in llm.calls] == ["s1", "s2"]
    assert llm.pending == 0


def test_raises_queued_exceptions() -> None:
    llm = FakeLLM([RuntimeError("boom")])
    with pytest.raises(RuntimeError, match="boom"):
        llm.generate(system="s", messages=MESSAGES, output_type=ReplyOutput)


def test_fails_loudly_when_the_script_is_wrong() -> None:
    llm = FakeLLM([SummaryOutput(key_points=[])])
    with pytest.raises(AssertionError, match="expected to return ReplyOutput"):
        llm.generate(system="s", messages=MESSAGES, output_type=ReplyOutput)
    with pytest.raises(AssertionError, match="no response queued"):
        llm.generate(system="s", messages=MESSAGES, output_type=ReplyOutput)
