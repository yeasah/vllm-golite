"""One event stream, multiplexed, with typed events.

Not a stream per subject: browsers cap concurrent connections per origin on HTTP/1.1,
and a page watching engine state, logs and progress would spend that budget on plumbing
before it displayed anything.

The stream is also what keeps the request rate down. Clients never poll for state --
they act over HTTP and learn what happened here -- so polling turning up anywhere is the
signal that something belongs on this stream instead.

Subscribers are bounded. A client that stops reading is a client whose queue grows
without limit, so a full queue drops the oldest event and records that it did: a viewer
missing a log line is a nuisance, and a manager growing until it is killed is an outage.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

#: Deep enough to absorb a burst of engine startup logging, shallow enough that a dead
#: reader costs bounded memory.
QUEUE_DEPTH = 512


@dataclass
class Event:
    type: str
    data: dict[str, Any] = field(default_factory=dict)


class EventBus:
    def __init__(self) -> None:
        self._queues: set[asyncio.Queue[Event]] = set()
        self.dropped = 0

    def publish(self, type: str, **data: Any) -> None:
        """Synchronous on purpose: it is called from the supervisor's log pump, which is
        not the place to await a slow consumer."""
        event = Event(type, data)
        for queue in self._queues:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()  # drop the oldest, keep the newest
                self.dropped += 1
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(event)

    @contextlib.asynccontextmanager
    async def subscribe(self) -> AsyncIterator[asyncio.Queue[Event]]:
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=QUEUE_DEPTH)
        self._queues.add(queue)
        try:
            yield queue
        finally:
            self._queues.discard(queue)

    @property
    def subscribers(self) -> int:
        return len(self._queues)
