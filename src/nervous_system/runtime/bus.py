"""In-process message bus (Architecture §10). Topic pub/sub; ZeroMQ/NATS can replace it later."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

Handler = Callable[[str, Any], None]


class Bus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Handler]] = defaultdict(list)
        self.published = 0

    def subscribe(self, topic: str, handler: Handler) -> None:
        """`topic` may end in `*` to match a prefix (e.g. `hive.*`)."""
        self._subs[topic].append(handler)

    def publish(self, topic: str, msg: Any) -> None:
        self.published += 1
        for pattern, handlers in list(self._subs.items()):
            if pattern == topic or (pattern.endswith("*") and topic.startswith(pattern[:-1])):
                for h in list(handlers):
                    h(topic, msg)
