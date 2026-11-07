"""Per-bot runtime loop (Architecture §5, §10, §11).

Every tick:     kill switch first; read state; forward model, drives, salience; deliver due brain
                replies; goal expiry / completion / valid_while; brain wake conditions.
Every cadence:  adapter.describe -> model.decide -> EscalationPolicy -> act or escalate; log the
                Decision (every one, Architecture §12.1); execute via the adapter's reflexes.
Then:           adapter.tick() advances the game (stepped) or services I/O (real time).

Brain replies land `brain_latency_s` of game time after the call, so the design's staleness rules
are exercised even with an instant brain: replies whose goal or context has changed are logged
and discarded, never applied.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from ..brain import Brain
from ..laya import DecisionModel, EscalationPolicy
from ..modules import KillSwitch, WorkingMemory
from ..modules.drives import BasicDrives
from ..protocol import Decision, Escalation, Event, Goal
from .bus import Bus
from .log import ExperienceLog


@dataclass
class RuntimeConfig:
    cadence_s: float = 0.1               # Laya decision period (game time)
    brain_latency_s: float = 2.0         # game seconds before a brain reply lands
    heartbeat_s: float = 10.0            # brain safety-net wake period
    event_driven: bool = True            # False: brain also reviews the world every heartbeat
    brain_event_threshold: float = 0.6   # salience needed to wake the brain
    surprise_threshold: float = 0.5
    override_s: float = 1.0              # how long a brain answer stands in its context
    label_fraction: float = 0.0          # DAgger: brain spot-checks this fraction of decisions
    outcome_period_s: float = 1.0
    watchdog_s: float | None = None      # wall-clock watchdog on the loop itself (real time)
    seed: int = 0


@dataclass
class _Pending:
    due: float
    kind: str                            # "plan" | "goal" | "answer"
    goal_id: str | None
    context: str | None
    payload: Any


@dataclass
class EpisodeStats:
    decisions: int = 0
    acted: int = 0
    escalations: int = 0
    cache_hits: int = 0
    laya_ms: float = 0.0
    stale_replies: int = 0
    events_to_brain: int = 0
    corrections: int = 0
    goals: int = 0
    reward: dict = field(default_factory=dict)
    escalation_reasons: dict = field(default_factory=dict)


class BotRuntime:
    def __init__(self, adapter, brain: Brain, model: DecisionModel, log: ExperienceLog,
                 header: dict, mission: str, config: RuntimeConfig | None = None,
                 policy: EscalationPolicy | None = None, kill: KillSwitch | None = None,
                 forward=None, salience=None, drives=None, memory: WorkingMemory | None = None,
                 bus: Bus | None = None, model_name: str = ""):
        self.a = adapter
        self.brain = brain
        self.model = model
        self.log = log
        self.hdr = dict(header)            # bot_id, game, episode_id
        self.mission = mission
        self.cfg = config or RuntimeConfig()
        self.policy = policy or EscalationPolicy()
        self.kill = kill or KillSwitch()
        self.forward, self.salience = forward, salience
        self.drives = drives or BasicDrives()
        self.wm = memory or WorkingMemory()
        self.bus = bus or Bus()
        self.model_name = model_name or getattr(model, "name", type(model).__name__)
        self.rng = random.Random(self.cfg.seed)

        self.goals: list[Goal] = []
        self.pending: list[_Pending] = []
        self.override: tuple[str, str, float] | None = None   # (context, choice, until)
        self.waiting_ctx: str | None = None                   # context with an open escalation
        self.prev_state = self.prev_facts = None
        self.last_decision_t = -1e9
        self.last_brain_t = 0.0
        self.last_outcome_t = 0.0
        self.reward_acc: dict[str, float] = {}
        self.stats = EpisodeStats()
        self.t = 0.0
        self.done = False
        self.batcher = None                                   # set by HiveRuntime
        self.summary_extra = None                             # hive: extra status for the brain

        self.kill.on_trip(self._on_kill)
        if self.cfg.watchdog_s:
            self.kill.watch(f"runtime:{self.hdr['bot_id']}", self.cfg.watchdog_s)

    # ------------------------------------------------------------------ helpers
    def h(self, t: float | None = None) -> dict:
        return {**self.hdr, "t": self.t if t is None else t}

    def _event(self, kind: str, salience: float, desc: str, **data) -> Event:
        ev = Event(**self.h(), kind=kind, salience=salience, description=desc, data=data)
        self.log.write(ev)
        self.bus.publish(f"event.{self.hdr['bot_id']}", ev)
        return ev

    def _on_kill(self) -> None:
        self.done = True
        try:
            self.a.release_all()
        finally:
            self.a.disconnect()
            self._event("kill", 1.0, f"kill switch: {self.kill.reason}")
            self.log.flush()

    def _summary(self, state, goal) -> str:
        text = self.a.summarise(state, goal)
        return text + "\n" + self.summary_extra() if self.summary_extra else text

    @property
    def goal(self) -> Goal | None:
        return self.goals[0] if self.goals else None

    def set_goals(self, goals: list[Goal], replace: bool = True) -> None:
        if replace:
            self.goals = []
        for g in goals:
            self.log.write(g)
            self.bus.publish(f"goal.{self.hdr['bot_id']}", g)
            self.stats.goals += 1
        self.goals = list(goals) + self.goals
        self.override = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        state = self.a.read_state()
        self.t = self.a.facts(state).t
        self.last_brain_t = self.t
        self.set_goals(self.brain.plan(self.mission, self._summary(state, None), self.h()))

    def step(self) -> bool:
        if self.done or self.kill.check():
            return False
        self.kill.beat(f"runtime:{self.hdr['bot_id']}")
        state = self.a.read_state()
        f = self.a.facts(state)
        self.t = f.t

        # tier-2 modules
        pred = self.forward.predict(f) if self.forward else {}
        surprise = self.forward.surprise(pred, f) if self.forward else 0.0
        drives = self.drives.compute(f)
        events = self.salience.score(self.prev_facts, f) if self.salience else []
        for ev in events:
            self.log.write(ev)
            self.bus.publish(f"event.{self.hdr['bot_id']}", ev)
        if surprise >= self.cfg.surprise_threshold:
            events.append(self._event("surprise", min(1.0, surprise), f"prediction error {surprise:.2f}"))

        self._deliver(state)
        self._goal_upkeep(state)
        self._brain_wake(state, events)

        if self.t - self.last_decision_t >= self.cfg.cadence_s - 1e-9:
            self.last_decision_t = self.t
            self._decide(state, pred, drives)

        if self.prev_state is not None:
            for k, v in self.a.reward(self.prev_state, state).items():
                self.reward_acc[k] = self.reward_acc.get(k, 0.0) + v
                self.stats.reward[k] = self.stats.reward.get(k, 0.0) + v
        if self.t - self.last_outcome_t >= self.cfg.outcome_period_s and self.reward_acc:
            self._event("outcome", 0.0, "reward window", **{k: round(v, 3) for k, v in self.reward_acc.items()})
            self.reward_acc, self.last_outcome_t = {}, self.t

        self.prev_state, self.prev_facts = state, f
        if self.done:
            return False
        if self.batcher is not None:            # hive ticks adapters after the batch commits
            return True
        return self.after_decisions()

    def after_decisions(self) -> bool:
        alive = self.a.tick()
        if not alive:
            self.done = True
        return alive

    def run(self, max_steps: int | None = None) -> EpisodeStats:
        self.start()
        n = 0
        while self.step():
            n += 1
            if max_steps and n >= max_steps:
                break
        self.finish()
        return self.stats

    def finish(self) -> None:
        self.kill.unwatch(f"runtime:{self.hdr['bot_id']}")
        if self.reward_acc:
            self._event("outcome", 0.0, "reward window", **{k: round(v, 3) for k, v in self.reward_acc.items()})
        self.log.flush()

    # ------------------------------------------------------------------ goals & brain
    def _goal_upkeep(self, state) -> None:
        g = self.goal
        if g is None:
            if not any(p.kind == "plan" for p in self.pending):
                self._ask_plan(state, "no active goal")
            return
        if self.t >= g.expires_at:
            self.goals.pop(0)
            self._event("override", 0.5, f"goal {g.intent} expired", goal_id=g.goal_id)
            self._ask_plan(state, f"goal {g.intent} expired")
            return
        status = self.a.goal_status(state, g)
        if status != "active":
            self.goals.pop(0)
            sal = 0.7 if status == "failed" else 0.4
            ev = self._event("event", sal, f"goal {g.intent} {status}", goal_id=g.goal_id, status=status)
            if not self.goals:
                self._ask_plan(state, f"goal {g.intent} {status}")
            else:                                   # wake condition 1 (§4): goal completed / failed
                self.stats.events_to_brain += 1
                self.last_brain_t = self.t
                ng = self.brain.on_event(ev, self._summary(state, self.goal), self.goal, self.h())
                if ng is not None:
                    self.pending.append(_Pending(self.t + self.cfg.brain_latency_s, "goal",
                                                 self.goal.goal_id, None, ng))
            return
        broken = [c for c in g.valid_while if not self.a.check(c, state)]
        if broken:
            self.goals.pop(0)
            self._event("override", 0.7, f"goal {g.intent} suspended: {broken[0]}", goal_id=g.goal_id)
            self._ask_plan(state, f"goal {g.intent} suspended ({broken[0]})")

    def _ask_plan(self, state, why: str) -> None:
        if any(p.kind == "plan" for p in self.pending):
            return
        self.last_brain_t = self.t
        self.stats.events_to_brain += 1
        goals = self.brain.plan(self.mission, self._summary(state, None) + f"\nwhy: {why}", self.h())
        self.pending.append(_Pending(self.t + self.cfg.brain_latency_s, "plan", None, None, goals))

    def _brain_wake(self, state, events: list[Event]) -> None:
        salient = [e for e in events if e.salience >= self.cfg.brain_event_threshold]
        due_heartbeat = self.t - self.last_brain_t >= self.cfg.heartbeat_s
        if not salient and not due_heartbeat:
            return
        if salient:
            ev = max(salient, key=lambda e: e.salience)
            self.stats.events_to_brain += 1
        else:
            ev = Event(**self.h(), kind="event", salience=0.0, description="heartbeat")
        if not salient and self.cfg.event_driven and self.goal is not None:
            # event-driven brain: heartbeat is only a safety net when the bot has a goal
            self.last_brain_t = self.t
            return
        self.last_brain_t = self.t
        g = self.brain.on_event(ev, self._summary(state, self.goal), self.goal, self.h())
        if g is not None:
            self.pending.append(_Pending(self.t + self.cfg.brain_latency_s, "goal",
                                         self.goal.goal_id if self.goal else None, None, g))

    def _deliver(self, state) -> None:
        due = [p for p in self.pending if p.due <= self.t + 1e-9]
        if not due:
            return
        self.pending = [p for p in self.pending if p.due > self.t + 1e-9]
        ctx = self.a.context(state, self.goal)
        cur = self.goal.goal_id if self.goal else None
        for p in due:
            if p.kind == "plan":
                if p.payload:
                    self.set_goals(p.payload)
            elif p.kind == "goal":
                if p.goal_id != cur:
                    self.stats.stale_replies += 1
                    self._event("override", 0.1, "stale brain goal discarded", goal=p.payload.intent)
                else:
                    self.set_goals([p.payload], replace=False)
            elif p.kind == "answer":
                if self.waiting_ctx == p.context:
                    self.waiting_ctx = None
                if p.context != ctx or p.goal_id != cur:
                    self.stats.stale_replies += 1
                    self._event("override", 0.1, "stale brain answer discarded", choice=p.payload,
                                context=p.context)
                else:
                    self.override = (ctx, p.payload, self.t + self.cfg.override_s)

    # ------------------------------------------------------------------ decisions
    def _decide(self, state, pred: dict, drives: dict) -> None:
        g = self.goal
        text = self.a.describe(state, g, {"pred": pred, "drives": drives, "wm": self.wm.recent(3)})
        opts = self.a.options(state, g)
        if self.batcher is not None:            # hive: decided in one batch with the other bots
            self.batcher.add(self, (state, g, text, opts))
            return
        dist = self.model.decide(text, opts)
        latency = getattr(self.model, "last_latency_ms", None)
        if latency is None:
            latency = getattr(getattr(self.model, "last", None), "latency_ms", 0.0)
        cached = bool(getattr(self.model, "last_cached", False))
        self.commit_decision((state, g, text, opts), dist, latency, cached)

    def commit_decision(self, req, dist: dict, latency: float, cached: bool) -> None:
        state, g, text, opts = req
        ctx = self.a.context(state, g)
        act, top, margin = self.policy.should_act(dist)
        self.stats.decisions += 1
        self.stats.cache_hits += int(cached)
        self.stats.laya_ms += latency
        if act:
            choice = top
            self.stats.acted += 1
        else:
            reason = ("chose_escalate" if top == "escalate" else
                      "low_confidence" if dist[top] < self.policy.act_threshold else "small_margin")
            ov = self.override
            if ov and ov[0] == ctx and ov[2] >= self.t and ov[1] in opts:
                choice = ov[1]                         # brain's answer for this context still stands
            else:
                choice = self.a.safe_default(state, g)
                if choice not in opts:
                    choice = next(o for o in opts if o != "escalate")
                if self.waiting_ctx != ctx:
                    self._escalate(state, g, reason, text, opts, dist, choice, ctx)
        self.log.write(Decision(**self.h(), goal_id=g.goal_id if g else "none", state_text=text,
                                options=opts, choice=choice, distribution=dist, acted=act,
                                margin=round(margin, 4), latency_ms=round(latency, 2),
                                model=self.model_name + ("+cache" if cached else "")))
        if self.cfg.label_fraction and act and self.rng.random() < self.cfg.label_fraction:
            label = self.brain.label(text, opts)
            self._correction(text, opts, label, top, "spot_check")
        self.wm.push(choice)
        self.a.execute(choice, g, state)

    def _escalate(self, state, g, reason, text, opts, dist, fallback, ctx) -> None:
        self.stats.escalations += 1
        self.stats.escalation_reasons[reason] = self.stats.escalation_reasons.get(reason, 0) + 1
        esc = Escalation(**self.h(), goal_id=g.goal_id if g else "none", reason=reason,
                         summary=self._summary(state, g), fallback_action=fallback,
                         distribution=dist)
        self.log.write(esc)
        self.bus.publish(f"escalation.{self.hdr['bot_id']}", esc)
        self.waiting_ctx = ctx
        self.last_brain_t = self.t
        answer = self.brain.on_escalation(esc, text, opts)
        top = max(dist, key=dist.get)
        self._correction(text, opts, answer, top, "escalation")
        self.pending.append(_Pending(self.t + self.cfg.brain_latency_s, "answer",
                                     g.goal_id if g else None, ctx, answer))

    def _correction(self, text: str, opts: list[str], label: str, laya_top: str, source: str) -> None:
        self.stats.corrections += 1
        self._event("correction", 0.0, f"brain label {label} (laya {laya_top})", state_text=text,
                    options=opts, label=label, laya_choice=laya_top, agree=label == laya_top,
                    source=source)
