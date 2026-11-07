"""Protocol v0: pydantic models and protocol/*.schema.json must agree (CLAUDE.md ground rules)."""
import json
from pathlib import Path

import jsonschema
import pytest
from pydantic import ValidationError

from nervous_system.protocol import (
    MESSAGE_TYPES, SCHEMA_VERSION, Decision, Escalation, Event, Goal,
)

SCHEMAS = Path(__file__).resolve().parents[1] / "protocol"
HDR = dict(bot_id="b1", game="minecraft", episode_id="ep1", t=12.5)

EXAMPLES = {
    "goal": Goal(**HDR, goal_id="g1", intent="gather:wood", expires_at=42.0, on_fail="return:base",
                 constraints=["stay_near base"], priorities=["survive", "finish_task"]),
    "decision": Decision(**HDR, goal_id="g1", state_text="GOAL ...", options=["continue", "escalate"],
                         choice="continue", distribution={"continue": 0.9, "escalate": 0.1},
                         acted=True, margin=0.8, latency_ms=33.0, model="laya-typed-decisions"),
    "escalation": Escalation(**HDR, goal_id="g1", reason="low_confidence", summary="...",
                             fallback_action="disengage", distribution={"engage": 0.5}),
    "event": Event(**HDR, kind="surprise", salience=0.7, data={"err": 1.2}),
}


def schema(name):
    return json.loads((SCHEMAS / f"{name}.schema.json").read_text())


@pytest.mark.parametrize("name", MESSAGE_TYPES)
def test_example_validates_against_schema(name):
    jsonschema.validate(json.loads(EXAMPLES[name].to_json()), schema(name))


@pytest.mark.parametrize("name", MESSAGE_TYPES)
def test_fields_match_schema(name):
    s, model = schema(name), MESSAGE_TYPES[name]
    assert set(model.model_fields) == set(s["properties"])
    required = {k for k, f in model.model_fields.items() if f.is_required()}
    # schema_version and wall_t have defaults in Python but are always serialised
    assert required | {"schema_version"} == set(s["required"])
    assert s["properties"]["schema_version"]["const"] == SCHEMA_VERSION
    assert s["additionalProperties"] is False
    assert s["$id"] == f"{name}.v{SCHEMA_VERSION}.schema.json"


@pytest.mark.parametrize("name", MESSAGE_TYPES)
def test_round_trip(name):
    msg = EXAMPLES[name]
    assert MESSAGE_TYPES[name].model_validate_json(msg.to_json()) == msg


def test_rejects_unknown_fields_and_bad_values():
    with pytest.raises(ValidationError):
        Event(**HDR, kind="event", salience=0.5, extra_field=1)
    with pytest.raises(ValidationError):
        Event(**HDR, kind="event", salience=1.5)
    with pytest.raises(ValidationError):
        Decision(**HDR, goal_id="g", state_text="", options=["only"], choice="only",
                 distribution={"only": 1.0}, acted=True)
    with pytest.raises(ValidationError):
        Goal(**HDR, goal_id="g", intent="wait", expires_at=1, on_fail="wait", schema_version="1")
