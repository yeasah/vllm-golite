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

from vllm_untwisted.engine.config import EngineConfig
from vllm_untwisted.engine.orphans import OWNER_VAR


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


#: Namespaces the engine stack reads as *configuration*. Anything here that untwisted did
#: not put there is ambient state leaking into a run, so it is dropped unless the
#: configuration asks for it.
#:
#: This is not hypothetical. The first real capture through this supervisor inherited
#: `VLLM_DISABLE_COMPILE_CACHE=1` from the developer's shell and paid 50 s of
#: compilation that the config it was reproducing does not pay -- the shell script being
#: replaced starts with `unset VLLM_DISABLE_COMPILE_CACHE` for exactly this reason.
#: In the shipped image the base set is controlled and this filter should find nothing;
#: on a development box it is the difference between a measurement and a coincidence.
TUNING_PREFIXES = (
    "VLLM_", "EXL3_", "PYTORCH_", "TORCH_", "TRITON_", "CUDA_", "NCCL_", "TORCHINDUCTOR_",
)


def declared_env(ambient: dict[str, str] | None = None) -> tuple[dict[str, str], list[str]]:
    """Ambient environment with engine-tuning variables removed.

    Returns the base and the names dropped, because a silently filtered environment is
    its own kind of surprise -- the supervisor reports what it took away.
    """
    ambient = dict(ambient if ambient is not None else os.environ)
    dropped = [k for k in ambient if k.startswith(TUNING_PREFIXES)]
    for k in dropped:
        del ambient[k]
    return ambient, sorted(dropped)


class SubprocessRuntime:
    """Spawns `vllm serve` locally."""

    def __init__(self, base_env: dict[str, str] | None = None,
                 owner: str | None = None) -> None:
        #: Stamped into every engine's environment so an orphan can be recognised after
        #: the manager that started it is gone. See `vllm_untwisted.engine.orphans`.
        self.owner = owner
        #: The environment engines start from. A configuration *adds* to this; nothing
        #: reaches an engine that untwisted did not decide to send.
        self.base_env, self.dropped_env = declared_env(base_env)

    async def spawn(self, config: EngineConfig, port: int) -> SubprocessHandle:
        env = {**self.base_env, **config.env}
        if self.owner:
            env[OWNER_VAR] = self.owner
        proc = await asyncio.create_subprocess_exec(
            *config.argv(port),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # one ordered stream; interleaving matters
            env=env,
            # Its own session, so the whole tree can be signalled as a unit.
            start_new_session=True,
        )
        return SubprocessHandle(proc)
