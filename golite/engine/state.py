"""Engine lifecycle states, and an honest account of failure.

"Reports honestly: starting, healthy, failed-and-why" is the whole requirement. The
hard half is the *why*: vLLM's interesting failures are reported to the log rather than
raised, so an exit code alone says almost nothing. `logscan` supplies the reason and
this module is where it lands.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum


class EngineState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    #: Terminal until something calls stop(); the record keeps the reason.
    FAILED = "failed"


class FailureKind(StrEnum):
    #: Process exited during startup. Usually the log says more than the code does.
    EXIT_BEFORE_READY = "exit_before_ready"
    #: Out of GPU memory, at profile time or during capture.
    OOM = "oom"
    #: Profiling succeeded but no memory was left for the cache.
    NO_KV_CACHE = "no_kv_cache"
    #: Never became healthy within the deadline, and never exited either.
    START_TIMEOUT = "start_timeout"
    PORT_IN_USE = "port_in_use"
    MODEL_NOT_FOUND = "model_not_found"
    #: Was READY and then the process went away.
    DIED_WHILE_READY = "died_while_ready"
    #: Classified as nothing above. Carries the log tail so it is still actionable.
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Failure:
    kind: FailureKind
    #: One line, suitable for a status field.
    summary: str
    exit_code: int | None = None
    #: The evidence. Kept because an unclassified failure is still diagnosable by hand,
    #: and because a tail is what tells us which pattern we are missing.
    log_tail: tuple[str, ...] = ()


@dataclass(slots=True)
class EngineRecord:
    """One engine. It carries an id even though there is only ever one of these today --
    the cost of foreclosing multi-engine is letting the singleton assumption spread, and
    an id costs nothing now."""

    id: str
    config_name: str
    state: EngineState = EngineState.STOPPED
    port: int | None = None
    pid: int | None = None
    failure: Failure | None = None
    #: Facts harvested from the startup log. This is the tier-2 measurement: an engine
    #: start is expensive, so anything it reports is worth keeping.
    facts: dict[str, str] = field(default_factory=dict)
    started_at: float | None = None
    ready_at: float | None = None

    @property
    def uptime(self) -> float | None:
        return None if self.ready_at is None else time.monotonic() - self.ready_at

    @property
    def startup_seconds(self) -> float | None:
        """What a model change costs, which is the appliance's dominant UX cost with one
        engine. Recorded on every start rather than measured in a special run."""
        if self.started_at is None or self.ready_at is None:
            return None
        return self.ready_at - self.started_at
