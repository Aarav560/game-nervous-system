"""Claude brain (Architecture §4). Requires `pip install anthropic` and ANTHROPIC_API_KEY.

Prompts are short and ask for JSON only. Everything the model returns is validated against
the vocabulary / offered options before use; anything invalid falls back to a safe answer
(`escalate` is never returned to Laya — the brain must commit).
"""
from __future__ import annotations

import json
import re
from typing import Any

from .. import vocabulary
from ..protocol import Escalation, Event, Goal
from . import Brain, fill_goal

SYSTEM = (
    "You are the strategic brain of a game-playing bot. A fast local model (Laya) makes tactical "
    "choices from a fixed option list; reflex controllers execute them. You set goals and answer "
    "Laya when it is unsure. You never press keys. Reply with JSON only, no prose.\n"
    "Goal JSON: {\"intent\": <verb or verb:arg>, \"target\": str|null, \"constraints\": [str], "
    "\"priorities\": [str], \"valid_while\": [str], \"ttl_s\": seconds, \"on_fail\": <verb>}.\n"
    "Vocabulary verbs: " + ", ".join(f"{v} ({d})" for v, d in vocabulary.VERBS.items())
)


def _json(text: str) -> Any:
    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    if not m:
        raise ValueError(f"no JSON in brain reply: {text[:200]!r}")
    return json.loads(m.group(1))


class ClaudeBrain(Brain):
    name = "claude"

    def __init__(self, model: str = "claude-sonnet-5-5", label_model: str | None = None,
                 client: Any = None, max_tokens: int = 400, goal_ttl_s: float = 120.0):
        super().__init__()
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.label_model = label_model or model
        self.max_tokens = max_tokens
        self.goal_ttl_s = goal_ttl_s

    def _ask(self, prompt: str, model: str | None = None) -> str:
        r = self.client.messages.create(model=model or self.model, max_tokens=self.max_tokens,
                                        system=SYSTEM, messages=[{"role": "user", "content": prompt}])
        text = "".join(getattr(b, "text", "") for b in r.content)
        u = getattr(r, "usage", None)
        self.usage.calls += 1
        if u is not None:
            self.usage.tokens_in += int(getattr(u, "input_tokens", 0))
            self.usage.tokens_out += int(getattr(u, "output_tokens", 0))
        else:
            self.usage.tokens_in += len(SYSTEM + prompt) // 4
            self.usage.tokens_out += len(text) // 4
        return text

    def _valid_goal(self, d: dict) -> bool:
        try:
            vocabulary.describe(d["intent"])
            vocabulary.describe(d.get("on_fail", "disengage"))
            return True
        except (KeyError, ValueError, TypeError):
            return False

    def plan(self, mission: str, world_summary: str, header: dict) -> list[Goal]:
        text = self._ask(f"Mission: {mission}\nWorld:\n{world_summary}\n"
                         "Return a JSON list of 1-4 goals, current goal first.")
        try:
            goals = [g for g in _json(text) if isinstance(g, dict) and self._valid_goal(g)]
        except ValueError:
            goals = []
        return [fill_goal(g, header, self.goal_ttl_s) for g in goals]

    def label(self, state_text: str, options: list[str]) -> str:
        allowed = [o for o in options if o != "escalate"]
        text = self._ask("Laya state:\n" + state_text + "\nOptions: " + ", ".join(allowed)
                         + '\nReturn {"choice": <one option exactly>}.', self.label_model)
        try:
            choice = _json(text).get("choice")
        except (ValueError, AttributeError):
            choice = None
        return choice if choice in allowed else allowed[0]

    def on_escalation(self, esc: Escalation, state_text: str, options: list[str]) -> str:
        return self.label(f"(escalated: {esc.reason})\n{state_text}", options)

    def on_event(self, ev: Event, world_summary: str, goal: Goal | None, header: dict) -> Goal | None:
        cur = goal.intent if goal else "none"
        text = self._ask(f"Event: {ev.description} (salience {ev.salience})\nCurrent goal: {cur}\n"
                         f"World:\n{world_summary}\nReturn {{\"keep\": true}} to keep the goal, "
                         "or a new goal JSON.")
        try:
            d = _json(text)
        except ValueError:
            return None
        if not isinstance(d, dict) or d.get("keep") or not self._valid_goal(d):
            return None
        return fill_goal(d, header, self.goal_ttl_s)
