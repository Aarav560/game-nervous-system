"""Shared action vocabulary (Architecture §9).

Generic verbs that mean the same thing in every game. Laya sees each option together with
its description; adapters map verbs to game specifics. Options may be parameterised as
`verb:arg` (e.g. `gather:wood`, `return:base`, `use_ability:totem_swap`).
"""
from __future__ import annotations

VERBS: dict[str, str] = {
    "engage": "fight the current target",
    "disengage": "break contact and create distance",
    "heal": "restore health",
    "reposition": "move to a better spot without disengaging",
    "use_ability": "use the {} ability",
    "gather": "collect {}",
    "explore": "seek unexplored area",
    "return": "go to {}",
    "regroup": "move toward allies",
    "build": "construct {}",
    "wait": "hold position and observe",
    "escalate": "unsure; ask the planner",
    # Not in the §9 table but used by every non-combat option set in configs/games/*.yaml.
    "continue": "keep working on the current goal",
}

PARAMETERISED = {v for v, d in VERBS.items() if "{}" in d}


def split(option: str) -> tuple[str, str | None]:
    verb, _, arg = option.partition(":")
    return verb, (arg or None)


def describe(option: str) -> str:
    """Natural-language description of an option for Laya's choice criteria."""
    verb, arg = split(option)
    if verb not in VERBS:
        raise KeyError(f"unknown vocabulary verb {verb!r} in option {option!r}")
    if verb in PARAMETERISED:
        if arg is None:
            raise ValueError(f"option {option!r} needs an argument, e.g. {verb}:<name>")
        return VERBS[verb].format(arg.replace("_", " "))
    if arg is not None:
        raise ValueError(f"verb {verb!r} takes no argument (got {option!r})")
    return VERBS[verb]


def validate(options: list[str]) -> None:
    """Raise if any option is outside the vocabulary or `escalate` is missing (§5)."""
    for o in options:
        describe(o)
    if "escalate" not in options:
        raise ValueError("'escalate' must always be offered (Architecture §5)")
    if len(set(options)) != len(options):
        raise ValueError(f"duplicate options: {options}")
