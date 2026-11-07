"""Laya wrapper + escalation policy, with a fake agent (no model download)."""
import pytest

from nervous_system import vocabulary
from nervous_system.laya import EscalationPolicy
from nervous_system.laya.model import QUESTION_ID, LayaDecisionModel, build_question


class FakeAgent:
    def __init__(self, probs, state_tokens=60, truncated=False):
        self.probs, self.state_tokens, self.truncated = probs, state_tokens, truncated
        self.calls = []

    def _result(self):
        return {"answers": {QUESTION_ID: {"type": "choice", "probabilities": dict(self.probs)}},
                "usage": {"state_tokens": self.state_tokens, "truncated": self.truncated}}

    def predict(self, state, questions):
        self.calls.append((state, questions))
        return self._result()

    def predict_batch(self, states, questions):
        self.calls.append((states, questions))
        return [self._result() for _ in states]


OPTS = ["engage", "disengage", "use_ability:totem_swap", "escalate"]
PROBS = {"engage": 0.8, "disengage": 0.1, "use_ability:totem_swap": 0.05, "escalate": 0.05}


def test_question_uses_vocabulary_descriptions():
    q = build_question(OPTS)[QUESTION_ID]
    assert q["type"] == "choice"
    assert list(q["criteria"]) == OPTS
    assert q["criteria"]["use_ability:totem_swap"] == "use the totem swap ability"


def test_decide_returns_distribution_over_options():
    m = LayaDecisionModel(agent=FakeAgent(PROBS))
    assert m.decide("SELF hp 12/20", OPTS) == PROBS
    assert m.last.state_tokens == 60 and m.last.latency_ms >= 0


def test_decide_rejects_oversized_state():
    m = LayaDecisionModel(agent=FakeAgent(PROBS, state_tokens=351))
    with pytest.raises(ValueError, match="serializer"):
        m.decide("x", OPTS)
    m = LayaDecisionModel(agent=FakeAgent(PROBS, truncated=True))
    with pytest.raises(ValueError):
        m.decide("x", OPTS)


def test_decide_batch_groups_by_option_set():
    agent = FakeAgent({**PROBS, "continue": 0.5, "return:base": 0.5})
    m = LayaDecisionModel(agent=agent)
    other = ["continue", "return:base", "escalate"]
    out = m.decide_batch([("a", OPTS), ("b", other), ("c", OPTS)])
    assert len(agent.calls) == 2
    assert list(out[0]) == OPTS and list(out[1]) == other and list(out[2]) == OPTS


def test_vocabulary_rules():
    with pytest.raises(ValueError, match="escalate"):
        vocabulary.validate(["engage", "disengage"])
    with pytest.raises(KeyError):
        vocabulary.validate(["fly", "escalate"])
    with pytest.raises(ValueError):
        vocabulary.describe("gather")          # needs an argument
    with pytest.raises(ValueError):
        vocabulary.describe("engage:steve")    # takes none
    assert vocabulary.describe("return:base") == "go to base"


def test_minecraft_option_sets_are_in_vocabulary():
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/games/minecraft.yaml").read_text())
    for opts in cfg["option_sets"].values():
        vocabulary.validate(opts)


@pytest.mark.parametrize("dist,act,top", [
    ({"engage": 0.9, "disengage": 0.05, "escalate": 0.05}, True, "engage"),
    ({"engage": 0.6, "disengage": 0.3, "escalate": 0.1}, False, "engage"),     # below threshold
    ({"engage": 0.8, "disengage": 0.7}, False, "engage"),                       # margin too small
    ({"escalate": 0.95, "engage": 0.05}, False, "escalate"),                    # chose escalate
])
def test_escalation_policy(dist, act, top):
    a, t, _ = EscalationPolicy().should_act(dist)
    assert (a, t) == (act, top)
