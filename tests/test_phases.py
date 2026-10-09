"""The phase-transition table, tested without an LLM."""

import pytest

from agent.domain.models import Intent, Phase
from agent.engine.phases import Action, next_action

A, P, In = Action, Phase, Intent


@pytest.mark.parametrize(
    ("phase", "intent", "changed", "ready", "expected"),
    [
        # stop
        (P.CONSENT, In.STOP, False, False, A.DECLINE),
        (P.COLLECTING, In.STOP, False, False, A.STOP),
        (P.CONFIRMING, In.STOP, True, True, A.STOP),
        (P.DONE, In.STOP, False, True, A.REPLY),
        # consent
        (P.CONSENT, In.REJECT, False, False, A.DECLINE),
        (P.CONSENT, In.REFUSE, False, False, A.DECLINE),
        (P.CONSENT, In.OFF_TOPIC, False, False, A.REPLY),
        (P.CONSENT, In.CONFIRM, False, False, A.START_COLLECTING),
        (P.CONSENT, In.ANSWER, True, False, A.START_COLLECTING),
        (P.CONSENT, In.ANSWER, True, True, A.CONFIRM),  # everything in one message
        # collecting
        (P.COLLECTING, In.ANSWER, True, False, A.REPLY),
        (P.COLLECTING, In.ANSWER, True, True, A.CONFIRM),
        (P.COLLECTING, In.ANSWER, False, True, A.REPLY),  # nothing new: don't re-confirm
        (P.COLLECTING, In.CONFIRM, False, True, A.CONFIRM),
        (P.COLLECTING, In.REFUSE, True, True, A.CONFIRM),  # skipped the last optional field
        # confirming
        (P.CONFIRMING, In.CORRECTION, True, True, A.CONFIRM),
        (P.CONFIRMING, In.CONFIRM, False, True, A.RECOMMEND),
        (P.CONFIRMING, In.REJECT, False, True, A.BACK_TO_COLLECTING),
        (P.CONFIRMING, In.CORRECTION, False, False, A.BACK_TO_COLLECTING),  # invalid fix
        (P.CONFIRMING, In.OFF_TOPIC, False, True, A.REPLY),
        # done
        (P.DONE, In.CORRECTION, True, True, A.CONFIRM),
        (P.DONE, In.OFF_TOPIC, False, True, A.REPLY),
        (P.DONE, In.CORRECTION, True, False, A.REPLY),  # rejected value
    ],
)
def test_next_action(
    phase: Phase, intent: Intent, changed: bool, ready: bool, expected: Action
) -> None:
    assert next_action(phase, intent, changed=changed, ready=ready) is expected


@pytest.mark.parametrize("phase", list(Phase))
@pytest.mark.parametrize("intent", list(Intent))
def test_every_phase_and_intent_has_an_action(phase: Phase, intent: Intent) -> None:
    for changed in (False, True):
        for ready in (False, True):
            assert isinstance(next_action(phase, intent, changed=changed, ready=ready), Action)
