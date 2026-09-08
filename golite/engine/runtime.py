"""The process boundary, behind a seam thin enough to be worth having.

There is one implementation and there will be one for a long time. The interface exists
because it is where a container or remote runtime would land if multi-node ever mattered,
and because naming the boundary is what stops the manager from growing calls into an
engine's internals.

Engines are child processes rather than in-process, and that is forced rather than
preferred: vLLM leaks in-process across engine constructions, so a process is never
reused for a second start. A crash also stays a crash of the child, which is what lets
the manager be the thing that notices.
"""

from __future__ import annotations

import asyncio
import os
import signal
from typing import Protocol, runtime_checkable

from golite.engine.config import EngineConfig


@runtime_checkable
class EngineHandle(Protocol):
    """A live engine process."""

    @property
    def pid(self) -> int | None: ...

    @property
    def returncode(self) -> int | None: ...

    async def readline(self) -> str | None:
        """Next line of merged stdout/stderr, or None at EOF."""

    async def stop(self, grace: float = 10.0) -> int | None:
        """Ask it to go away, then insist. Returns the exit code."""


class EngineRuntime(Protocol):
    async def spawn(self, config: EngineConfig, port: int) -> EngineHandle: ...


class SubprocessHandle:
    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        self._proc = proc
        # Captured now: once the process is reaped, getpgid can no longer find it, and
        # the group is what we actually have to clear.
        try:
            self._pgid: int | None = os.getpgid(proc.pid)
        except ProcessLookupError:
            self._pgid = None

    @property
    def pid(self) -> int | None:
        return self._proc.pid

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode

    async def readline(self) -> str | None:
        assert self._proc.stdout is not None
        raw = await self._proc.stdout.readline()
        return raw.decode(errors="replace") if raw else None

    async def stop(self, grace: float = 10.0) -> int | None:
        if self._proc.returncode is not None:
            return self._proc.returncode
        # Signal the *group*, not the process. vLLM spawns its own children
        # (EngineCore among them); terminating only the parent orphans them still
        # holding VRAM, which is the failure that makes the next start look like a
        # code regression rather than a leftover.
        self._signal_group(signal.SIGTERM)
        try:
            code = await asyncio.wait_for(self._proc.wait(), timeout=grace)
        except TimeoutError:
            self._signal_group(signal.SIGKILL)
            code = await self._proc.wait()
        # Waiting on the parent is not the same as the group being gone. Returning
        # early would report "stopped" while a child still holds VRAM, which is
        # precisely what makes the *next* start look like a code regression.
        await self._drain_group(grace)
        return code

    async def _drain_group(self, grace: float) -> None:
        if self._pgid is None:
            return
        deadline = asyncio.get_running_loop().time() + grace
        escalated = False
        while True:
            try:
                os.killpg(self._pgid, 0)  # signal 0 only probes
            except (ProcessLookupError, PermissionError):
                return
            now = asyncio.get_running_loop().time()
            if now >= deadline:
                return
            if not escalated and now >= deadline - grace / 2:
                self._signal_group(signal.SIGKILL)
                escalated = True
            await asyncio.sleep(0.05)

    def _signal_group(self, sig: int) -> None:
        try:
            os.killpg(os.getpgid(self._proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            # Already gone, or not ours to signal. Fall back to the process itself.
            try:
                self._proc.send_signal(sig)
            except ProcessLookupError:
                pass


class SubprocessRuntime:
    """Spawns `vllm serve` locally."""

    def __init__(self, base_env: dict[str, str] | None = None) -> None:
        #: The environment engines start from. In the shipped image this is controlled
        #: exactly, so a configuration only ever *adds* to it.
        self.base_env = dict(base_env if base_env is not None else os.environ)

    async def spawn(self, config: EngineConfig, port: int) -> SubprocessHandle:
        env = {**self.base_env, **config.env}
        proc = await asyncio.create_subprocess_exec(
            *config.argv(port),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # one ordered stream; interleaving matters
            env=env,
            # Its own session, so the whole tree can be signalled as a unit.
            start_new_session=True,
        )
        return SubprocessHandle(proc)
