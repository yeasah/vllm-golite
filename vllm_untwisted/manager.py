"""Start a stored configuration, and keep what the start reported.

The join the project exists for. The supervisor knows how to run an engine and nothing
about where configurations come from; the store knows about configurations and nothing
about running them. Keeping them apart matters because the supervisor is the process
boundary and should not acquire a database.

What this adds is the recording. An engine start costs a whole process launch and cannot
be amortized, so anything it measured is worth keeping -- and a measurement is only worth
keeping if it says what it was taken on, which is why every run carries a fingerprint.
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm_untwisted import fingerprint as fp
from vllm_untwisted.engine import EngineState, Supervisor
from vllm_untwisted.engine.state import EngineRecord
from vllm_untwisted.store import ConfigEntry, Store

#: A start that never became healthy.
FAILED = "failed"
#: Became healthy.
READY = "ready"
#: Became healthy and then the engine went away on its own.
DIED = "died"
#: Became healthy and was stopped deliberately.
STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class Started:
    entry: ConfigEntry
    record: EngineRecord
    run_id: str

    @property
    def ok(self) -> bool:
        return self.record.state is EngineState.READY


class Manager:
    def __init__(
        self,
        store: Store,
        supervisor: Supervisor | None = None,
        fingerprint: dict[str, str] | None = None,
    ) -> None:
        self.store = store
        self.supervisor = supervisor or Supervisor()
        #: Collected once. It describes the box, which does not change between starts,
        #: and shelling out to nvidia-smi on every launch would be noise.
        self.fingerprint = fp.collect() if fingerprint is None else fingerprint
        self._run_id: str | None = None
        self._entry: ConfigEntry | None = None

    async def start(self, ref: str) -> Started:
        entry = self.store.get(ref)
        if entry is None:
            raise KeyError(f"no configuration {ref!r}")

        record = await self.supervisor.start(entry.config)
        failure = record.failure
        run_id = self.store.record_run(
            entry.id,
            outcome=READY if record.state is EngineState.READY else FAILED,
            # A snapshot: the scanner's dict keeps filling, and this is what was true
            # when the start resolved.
            facts=dict(record.facts),
            fingerprint=self.fingerprint,
            failure_kind=None if failure is None else str(failure.kind),
            failure_summary=None if failure is None else failure.summary,
            startup_seconds=record.startup_seconds,
            compile_state=self.supervisor.compile_state,
        )
        self._run_id, self._entry = run_id, entry
        return Started(entry=entry, record=record, run_id=run_id)

    async def stop(self) -> None:
        """Stop the engine and settle the run's outcome.

        A start that succeeded and an engine that later died are one run: recording the
        death as a second run would double-count every start and make "has this ever
        worked" wrong.
        """
        state_before = self.supervisor.state
        failure = self.supervisor.record.failure if self.supervisor.record else None
        await self.supervisor.stop()

        if self._run_id is not None:
            if state_before is EngineState.READY:
                self.store.close_run(self._run_id, outcome=STOPPED)
            elif state_before is EngineState.FAILED and failure is not None:
                # Only meaningful if the start had succeeded; a failed start already
                # recorded its reason.
                self.store.close_run(
                    self._run_id, outcome=DIED, failure_kind=str(failure.kind),
                    failure_summary=failure.summary)
        self._run_id, self._entry = None, None

    @property
    def running(self) -> ConfigEntry | None:
        return self._entry if self.supervisor.state is EngineState.READY else None
