"""Kill switch, play rules, experience log, tier-2 modules, brains, calibration."""
import json
import os
import time
from types import SimpleNamespace

import pytest

from nervous_system.adapters import PlayRules
from nervous_system.brain import fill_goal
from nervous_system.brain.claude import ClaudeBrain
from nervous_system.laya.metrics import brier, ece
from nervous_system.laya.models import CachedDecisionModel, TemperatureCalibrator
from nervous_system.modules import KillSwitch
from nervous_system.modules.drives import BasicDrives
from nervous_system.modules.forward import PhysicsForwardModel
from nervous_system.modules.memory import EpisodicMemory, SkillLibrary
from nervous_system.modules.salience import RuleSalience
from nervous_system.protocol import Decision, Event
from nervous_system.runtime.bus import Bus
from nervous_system.runtime.log import ExperienceLog
from nervous_system.state import Entity, Facts

HDR = dict(bot_id="b", game="g", episode_id="e")


# ------------------------------------------------------------------ kill switch
def test_kill_switch_runs_every_handler_once_even_if_one_fails():
    k, calls = KillSwitch(), []
    k.on_trip(lambda: calls.append("release"))
    k.on_trip(lambda: 1 / 0)
    k.on_trip(lambda: calls.append("disconnect"))
    k.trip("test")
    k.trip("again")
    assert calls == ["release", "disconnect"] and k.reason == "test" and len(k.errors) == 1


def test_kill_switch_sentinel_and_late_handler(tmp_path):
    k = KillSwitch(sentinel=str(tmp_path / "KILL"))
    assert not k.check()
    (tmp_path / "KILL").write_text("")
    assert k.check() and "sentinel" in k.reason
    late = []
    k.on_trip(lambda: late.append(1))
    assert late == [1]


def test_kill_switch_watchdog_trips_on_silent_component():
    k, stopped = KillSwitch(), []
    k.on_trip(lambda: stopped.append(1))
    k.watch("laya", 0.05)
    k.start_watchdog(0.01)
    time.sleep(0.3)
    assert k.tripped and "watchdog" in k.reason and stopped == [1]


# ------------------------------------------------------------------ play rules
def test_play_rules():
    r = PlayRules(allowed=["singleplayer", "own_server", "permitted_server"],
                  permitted_servers=[{"address": "a:1", "owner": "o", "agreed_on": "2026-11-01"},
                                     {"address": "b:1"}])
    r.check("singleplayer")
    r.check("own_server", "127.0.0.1:25565")
    r.check("permitted_server", "a:1")
    for ctx, addr in [("permitted_server", "c:1"), ("permitted_server", "b:1"), ("own_server", "x.example:1"),
                      ("own_server", None)]:
        with pytest.raises(PermissionError):
            r.check(ctx, addr)
    with pytest.raises(PermissionError):
        PlayRules(allowed=["singleplayer"]).check("own_server", "127.0.0.1:1")


# ------------------------------------------------------------------ log
def test_experience_log_jsonl_and_index(tmp_path):
    log = ExperienceLog(tmp_path)
    log.start_episode("e", "g")
    log.write(Decision(**HDR, t=1.0, goal_id="g1", state_text="s", options=["a", "escalate"], choice="a",
                       distribution={"a": 0.9, "escalate": 0.1}, acted=True))
    log.write(Event(**HDR, t=2.0, kind="surprise", salience=0.8))
    log.flush()
    lines = [json.loads(l) for l in open(tmp_path / "e.jsonl")]
    assert [l["type"] for l in lines] == ["decision", "event"]
    assert log.db.execute("SELECT choice, p_top, line FROM decisions").fetchone() == ("a", 0.9, 0)
    assert log.db.execute("SELECT kind, line FROM events").fetchone() == ("surprise", 1)


# ------------------------------------------------------------------ modules
def _facts(t, hp=20.0, ents=()):
    return Facts(t=t, x=0, y=0, hp=hp, hp_max=20, entities=list(ents))


def test_forward_model_predicts_and_detects_surprise():
    fm = PhysicsForwardModel(horizon_ms=100)
    z = Entity("z", "zombie", True, 2.0, 0.0, vx=-1.0, facing_self=True, attack_ready=True)
    p = fm.predict(_facts(0.0, ents=[z]))
    assert p["opp_aim_lock"] > 0.5 and p["crit_incoming"] > 0.3
    z2 = Entity("z", "zombie", True, 1.9, 0.0, vx=-1.0)
    assert fm.surprise(p, _facts(0.1, ents=[z2])) < 0.2
    fm.predict(_facts(0.1, ents=[z2]))
    teleported = Entity("z", "zombie", True, 9.0, 9.0)
    assert fm.surprise({}, _facts(0.2, hp=10, ents=[teleported])) >= 0.5


def test_salience_flags_new_hostile_damage_and_death():
    s = RuleSalience(dict(HDR))
    f0 = _facts(0.0)
    z = Entity("z", "player", True, 3.0, 0.0)
    evs = s.score(f0, _facts(0.1, ents=[z]))
    assert any("new hostile" in e.description and e.salience >= 0.6 for e in evs)
    evs = s.score(_facts(0.1, ents=[z]), _facts(0.5, hp=14, ents=[z]))
    assert any("damage spike" in e.description for e in evs)
    dead = _facts(0.6, hp=0)
    dead.alive = False
    assert any(e.description == "died" and e.salience == 1.0 for e in s.score(_facts(0.5), dead))


def test_drives():
    d = BasicDrives()
    calm = d.compute(_facts(0.0))
    scared = d.compute(_facts(0.0, hp=4, ents=[Entity(str(i), "zombie", True, 2, 0) for i in range(3)]))
    assert calm["fear"] < 0.2 and scared["fear"] > 0.9


def test_memory(tmp_path):
    em = EpisodicMemory(tmp_path / "ep.sqlite")
    em.add("e1", "gridcraft", "combat", "lost to 3 zombies", "death", -20)
    em.add("e2", "skirmish", "combat", "won", "win", 5)
    assert [r["episode_id"] for r in em.recall("combat")] == ["e2", "e1"]
    assert [r["episode_id"] for r in em.recall("combat", game="gridcraft")] == ["e1"]
    sk = SkillLibrary(tmp_path / "skills.json")
    sk.record("gridcraft", "gather:wood", True)
    sk.record("gridcraft", "gather:wood", False)
    sk.save()
    assert SkillLibrary(tmp_path / "skills.json").get("gridcraft", "gather:wood").success_rate == 0.5


def test_bus_prefix_subscription():
    b, got = Bus(), []
    b.subscribe("hive.*", lambda t, m: got.append((t, m)))
    b.publish("hive.danger", 1)
    b.publish("other", 2)
    assert got == [("hive.danger", 1)]


# ------------------------------------------------------------------ calibration
def test_temperature_calibration_reduces_overconfidence():
    import math, random
    rng = random.Random(0)
    logits, labels, dists = [], [], []
    for _ in range(400):
        y = rng.randrange(3)
        top = y if rng.random() < 0.6 else rng.randrange(3)        # 60% accurate...
        z = [0.0, 0.0, 0.0]
        z[top] = 5.0                                               # ...but ~98% confident
        logits.append(z); labels.append(y)
    cal = TemperatureCalibrator()
    cal.fit(logits, labels)
    assert cal.temperature > 1.5
    names = ["a", "b", "c"]
    raw = [{n: math.exp(v) / sum(math.exp(u) for u in z) for n, v in zip(names, z)} for z in logits]
    lab = [names[y] for y in labels]
    assert ece([cal.apply(d) for d in raw], lab) < ece(raw, lab)
    assert brier([cal.apply(d) for d in raw], lab) < brier(raw, lab)


def test_cache():
    class M:
        calls = 0
        def decide(self, s, o):
            M.calls += 1
            return {o[0]: 1.0}
        def decide_batch(self, items):
            return [self.decide(s, o) for s, o in items]
    c = CachedDecisionModel(M())
    c.decide("x", ["a", "b"]); c.decide("x", ["a", "b"]); c.decide("y", ["a", "b"])
    assert M.calls == 2 and c.hits == 1 and c.last_cached is False


# ------------------------------------------------------------------ brains
class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.messages = self

    def create(self, **kw):
        return SimpleNamespace(content=[SimpleNamespace(text=self.replies.pop(0))],
                               usage=SimpleNamespace(input_tokens=100, output_tokens=10))


def test_claude_brain_validates_output():
    hdr = {**HDR, "t": 5.0}
    b = ClaudeBrain(client=FakeClient([
        '[{"intent": "gather:wood", "ttl_s": 30}, {"intent": "fly"}]',
        '{"choice": "engage"}', '{"choice": "teleport"}', 'not json']))
    goals = b.plan("m", "w", hdr)
    assert [g.intent for g in goals] == ["gather:wood"] and goals[0].expires_at == 35.0
    assert b.label("s", ["engage", "disengage", "escalate"]) == "engage"
    assert b.label("s", ["engage", "disengage", "escalate"]) == "engage"      # invalid -> first
    assert b.label("s", ["disengage", "escalate"]) == "disengage"             # unparseable
    assert b.usage.calls == 4 and b.usage.tokens_in == 400


def test_fill_goal_defaults():
    g = fill_goal({"intent": "wait"}, {**HDR, "t": 1.0}, default_ttl=9)
    assert g.expires_at == 10.0 and g.on_fail == "disengage"
