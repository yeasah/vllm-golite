"""One engine, started and stopped honestly.

The lifecycle is vllm-tuner's trial loop, which is the same state machine a manager
needs anyway: start, health check with a deadline, run, read the log for what actually
happened, clean up. The log-reading step is not incidental -- it is where both the
failure reason and the startup measurements come from.

v1 supervises exactly one engine. The record still carries an id, and the supervisor
still holds a table with one row, because the cost of foreclosing multi-engine is
letting a singleton assumption spread through everything downstream.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable

import httpx

from golite.engine.config import EngineConfig
from golite.engine.logscan import LogScanner
from golite.engine.runtime import EngineHandle, EngineRuntime, SubprocessRuntime
from golite.engine.state import EngineRecord, EngineState, Failure, FailureKind

#: How much log to keep in memory. A tail is what makes an unclassified failure still
#: diagnosable; keeping all of it would grow without bound on a long-running engine.
LOG_TAIL_LINES = 400


def pick_free_port() -> int:
    """Bind port 0 and let the OS choose.

    There is an unavoidable race: the port is free when we look and the child binds it a
    moment later. On a single-engine appliance the window is not contended, and a lost
    race surfaces cleanly as PORT_IN_USE rather than as corruption.
    """
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class EngineStartError(RuntimeError):
    """Raised when start() is called on a supervisor that is not idle."""


class Supervisor:
    def __init__(
        self,
        runtime: EngineRuntime | None = None,
        *,
        health_path: str = "/health",
        start_timeout: float = 600.0,
        poll_interval: float = 0.5,
    ) -> None:
        self.runtime = runtime or SubprocessRuntime()
        self.health_path = health_path
        #: Generous by default. A cold start pays interpreter startup, imports, CUDA
        #: context creation, weight load and graph capture, and none of that can be
        #: amortized by reusing a process -- see the leak note in docs/design.md.
        self.start_timeout = start_timeout
        self.poll_interval = poll_interval

        self.record: EngineRecord | None = None
        self._handle: EngineHandle | None = None
        self._scanner = LogScanner()
        self._log: deque[str] = deque(maxlen=LOG_TAIL_LINES)
        self._pump: asyncio.Task[None] | None = None
        self._watch: asyncio.Task[None] | None = None
        self._subscribers: set[Callable[[str], None]] = set()

    # -- observation ------------------------------------------------------------

    @property
    def state(self) -> EngineState:
        return self.record.state if self.record else EngineState.STOPPED

    def log_tail(self, n: int = 50) -> tuple[str, ...]:
        return tuple(self._log)[-n:]

    def subscribe(self, sink: Callable[[str], None]) -> Callable[[], None]:
        """Watch the log as it arrives. The event stream will be built on this."""
        self._subscribers.add(sink)
        return lambda: self._subscribers.discard(sink)

    # -- lifecycle --------------------------------------------------------------

    async def start(self, config: EngineConfig, port: int | None = None) -> EngineRecord:
        if self.state in (EngineState.STARTING, EngineState.READY, EngineState.STOPPING):
            raise EngineStartError(f"engine is {self.state}; stop it first")

        # Every start is a fresh process. Never reuse one: vLLM leaks across engine
        # constructions, so a warm process is not an optimization that is available.
        self._scanner = LogScanner()
        self._log.clear()

        port = port or pick_free_port()
        record = EngineRecord(
            id=uuid.uuid4().hex[:8],
            config_name=config.name,
            state=EngineState.STARTING,
            port=port,
            started_at=time.monotonic(),
        )
        self.record = record

        self._handle = await self.runtime.spawn(config, port)
        record.pid = self._handle.pid
        self._pump = asyncio.create_task(self._pump_log(), name=f"log-{record.id}")

        try:
            await self._await_ready(record)
        except Exception:
            await self._reap()
            raise

        return record

    async def stop(self, grace: float = 10.0) -> None:
        if self.record is None or self._handle is None:
            return
        if self.record.state is not EngineState.FAILED:
            self.record.state = EngineState.STOPPING
        await self._reap(grace)
        self.record.state = EngineState.STOPPED
        self.record.pid = None

    # -- internals --------------------------------------------------------------

    async def _pump_log(self) -> None:
        """Read the engine's output for as long as it produces any."""
        assert self._handle is not None
        while (line := await self._handle.readline()) is not None:
            text = line.rstrip("\n")
            self._log.append(text)
            self._scanner.feed(line)
            for sink in list(self._subscribers):
                with contextlib.suppress(Exception):
                    sink(text)

    async def _await_ready(self, record: EngineRecord) -> None:
        """Poll for health until it answers, the process dies, or the deadline passes."""
        assert self._handle is not None
        deadline = time.monotonic() + self.start_timeout
        url = f"http://127.0.0.1:{record.port}{self.health_path}"

        async with httpx.AsyncClient(timeout=2.0) as client:
            while time.monotonic() < deadline:
                if (code := self._handle.returncode) is not None:
                    record.state = EngineState.FAILED
                    record.failure = self._classify_exit(code)
                    return
                with contextlib.suppress(httpx.HTTPError):
                    if (await client.get(url)).status_code == 200:
                        record.state = EngineState.READY
                        record.ready_at = time.monotonic()
                        # The scanner's own dict, not a copy: health can answer while
                        # lines are still in the pipe, and vLLM prints its most useful
                        # numbers late. Facts accrue for the life of the engine.
                        record.facts = self._scanner.facts
                        self._watch = asyncio.create_task(
                            self._watch_liveness(record), name=f"watch-{record.id}"
                        )
                        return
                await asyncio.sleep(self.poll_interval)

        record.state = EngineState.FAILED
        record.failure = Failure(
            kind=FailureKind.START_TIMEOUT,
            summary=f"no health response within {self.start_timeout:.0f}s",
            log_tail=self.log_tail(),
        )

    def _classify_exit(self, code: int) -> Failure:
        """What the exit code cannot tell us, the log usually can."""
        if self._scanner.failure_kind is not None:
            return Failure(
                kind=self._scanner.failure_kind,
                summary=self._scanner.failure_line or str(self._scanner.failure_kind),
                exit_code=code,
                log_tail=self.log_tail(),
            )
        return Failure(
            kind=FailureKind.EXIT_BEFORE_READY,
            summary=f"exited with code {code} before becoming healthy",
            exit_code=code,
            log_tail=self.log_tail(),
        )

    async def _watch_liveness(self, record: EngineRecord) -> None:
        """A ready engine that goes away is a failure, not a stop.

        Open question this does not yet answer: whether vLLM's in-process leak has a
        per-request component. If it does, an unattended appliance needs RSS drift
        monitoring here and a restart policy above it.
        """
        assert self._pump is not None
        await self._pump  # returns at EOF, which means the process closed its output
        if record.state is EngineState.READY:
            record.state = EngineState.FAILED
            record.failure = Failure(
                kind=FailureKind.DIED_WHILE_READY,
                summary="engine exited while serving",
                exit_code=self._handle.returncode if self._handle else None,
                log_tail=self.log_tail(),
            )

    async def _reap(self, grace: float = 10.0) -> None:
        if self._handle is not None:
            await self._handle.stop(grace)
        for task in (self._watch, self._pump):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._watch = self._pump = None
        self._handle = None


@contextlib.asynccontextmanager
async def running(
    config: EngineConfig, supervisor: Supervisor | None = None, **kw
) -> AsyncIterator[Supervisor]:
    """Start an engine, hand it over, and make sure it is gone afterwards.

    The fit tiers each pay a process launch per candidate, so a shape that cannot leak a
    process on an exception is worth having before the loop that uses it exists.
    """
    sup = supervisor or Supervisor(**kw)
    try:
        await sup.start(config)
        yield sup
    finally:
        await sup.stop()
