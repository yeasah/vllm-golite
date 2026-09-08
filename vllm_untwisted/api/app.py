"""The one contract everything else speaks.

The rule this file exists to keep: **the manager has no internal path that bypasses its
own API.** The CLI is a client of these endpoints, not a second implementation, and when
a frontend arrives it gets the same ones. Without that, the UI grows privileged access,
the CLI never catches up, and maintaining both becomes the tax the arrangement was
supposed to avoid.

Starting an engine is a background task rather than a blocking POST. A cold start is
tens of seconds and a fresh container's first start is over a minute; no caller should
hold a request open for that, and a UI certainly cannot. So `POST /api/engine/start`
returns immediately and the outcome arrives on the event stream -- which is also the
reason the stream exists.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, Body, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from vllm_untwisted.api.events import EventBus
from vllm_untwisted.api.models import (
    ConfigIn,
    ConfigOut,
    EngineOut,
    FindingOut,
    ImportShIn,
    ImportShOut,
    RenameIn,
    RunOut,
    StartIn,
)
from vllm_untwisted.engine import EngineConfig, EngineState, Supervisor
from vllm_untwisted.manager import Manager
from vllm_untwisted.store import ConfigEntry, Store
from vllm_untwisted.store.db import RunRecord
from dataclasses import asdict

from vllm_untwisted.store.lint import lint
from vllm_untwisted.store.shell import parse_text


def _config_out(entry: ConfigEntry) -> ConfigOut:
    return ConfigOut(
        id=entry.id, name=entry.name, status=entry.status, origin=entry.origin,
        note=entry.note, run_count=entry.run_count, created_at=entry.created_at,
        command_line=entry.config.command_line(), doc=entry.config.to_doc(),
    )


def _run_out(run: RunRecord) -> RunOut:
    return RunOut(
        id=run.id, started_at=run.started_at, outcome=run.outcome,
        became_ready=run.became_ready, failure_kind=run.failure_kind,
        startup_seconds=run.startup_seconds, compile_state=run.compile_state,
        facts=run.facts, fingerprint=run.fingerprint,
    )


def create_app(store: Store | None = None, manager: Manager | None = None) -> FastAPI:
    app = FastAPI(title="vllm-untwisted", version="0.0.1")
    app.state.store = store or Store(_default_store())
    app.state.manager = manager or Manager(app.state.store, Supervisor())
    app.state.events = EventBus()
    app.state.starting = None

    # Engine output goes straight onto the stream. Synchronous by design: this runs in
    # the supervisor's log pump, which must not wait on a slow subscriber.
    app.state.manager.supervisor.subscribe(
        lambda line: app.state.events.publish("engine.log", line=line))

    api = APIRouter(prefix="/api")

    # -- configurations ---------------------------------------------------------

    @api.get("/configs", response_model=list[ConfigOut])
    def list_configs() -> list[ConfigOut]:
        return [_config_out(e) for e in app.state.store.list()]

    @api.post("/configs", response_model=ConfigOut, status_code=201)
    def create_config(body: ConfigIn) -> ConfigOut:
        config = EngineConfig(name=body.name, model=body.model, args=body.args,
                              env=body.env, launcher=tuple(body.launcher))
        try:
            cid = app.state.store.add(config, note=body.note)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        app.state.events.publish("config.added", id=cid, name=body.name)
        return _config_out(app.state.store.get(cid))

    @api.post("/configs/import-sh", response_model=ImportShOut)
    def import_sh(body: ImportShIn) -> ImportShOut:
        result = parse_text(body.content, body.filename)
        added, skipped = [], []
        for item in result.configs:
            if app.state.store.get(item.config.name) is not None:
                skipped.append(item.config.name)
                continue
            if not body.dry_run:
                app.state.store.add(item.config, note=item.note)
            added.append(item.config.name)
        if added and not body.dry_run:
            app.state.events.publish("config.added", names=added)
        return ImportShOut(added=added, skipped=skipped, warnings=result.warnings)

    @api.get("/configs/{ref}", response_model=ConfigOut)
    def get_config(ref: str) -> ConfigOut:
        return _config_out(_require(app, ref))

    @api.patch("/configs/{ref}", response_model=ConfigOut)
    def rename_config(ref: str, body: RenameIn) -> ConfigOut:
        entry = _require(app, ref)
        try:
            app.state.store.rename(entry.id, body.name)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        app.state.events.publish("config.renamed", id=entry.id, name=body.name)
        return _config_out(app.state.store.get(entry.id))

    @api.delete("/configs/{ref}", status_code=204)
    def delete_config(ref: str) -> None:
        entry = _require(app, ref)
        app.state.store.delete(entry.id)
        app.state.events.publish("config.deleted", id=entry.id, name=entry.name)

    @api.get("/configs/{ref}/runs", response_model=list[RunOut])
    def config_runs(ref: str, limit: int = 20) -> list[RunOut]:
        entry = _require(app, ref)
        return [_run_out(r) for r in app.state.store.runs(entry.id, limit=limit)]

    @api.get("/lint", response_model=dict[str, list[FindingOut]])
    def lint_all() -> dict[str, list[FindingOut]]:
        out: dict[str, list[FindingOut]] = {}
        for entry in app.state.store.list():
            findings = lint(entry)
            if findings:
                out[entry.name] = [FindingOut(**asdict(f)) for f in findings]
        return out

    # -- engine -----------------------------------------------------------------

    @api.get("/engine", response_model=EngineOut)
    def engine_state() -> EngineOut:
        return _engine_out(app)

    @api.post("/engine/start", response_model=EngineOut, status_code=202)
    async def engine_start(body: StartIn) -> EngineOut:
        manager: Manager = app.state.manager
        if manager.supervisor.state in (EngineState.STARTING, EngineState.READY):
            raise HTTPException(409, f"engine is {manager.supervisor.state}")
        entry = _require(app, body.ref)

        async def go() -> None:
            app.state.events.publish("engine.state", state="starting", config=entry.name)
            try:
                started = await manager.start(entry.id)
            except Exception as exc:  # a start that fails to even launch
                app.state.events.publish("engine.state", state="failed", error=str(exc))
                return
            app.state.events.publish(
                "engine.state",
                state=str(started.record.state),
                config=entry.name,
                port=started.record.port,
                startup_seconds=started.record.startup_seconds,
                failure_kind=None if not started.record.failure
                else str(started.record.failure.kind),
            )

        app.state.starting = asyncio.create_task(go(), name=f"start-{entry.name}")
        return EngineOut(state="starting", config=entry.name, config_id=entry.id)

    @api.post("/engine/stop", response_model=EngineOut)
    async def engine_stop() -> EngineOut:
        task = app.state.starting
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await app.state.manager.stop()
        app.state.events.publish("engine.state", state="stopped")
        return _engine_out(app)

    # -- events -----------------------------------------------------------------

    @api.get("/events")
    async def events() -> StreamingResponse:
        """One multiplexed stream. Typed events, SSE framing, and a periodic comment so
        an idle connection is not mistaken for a dead one."""
        bus: EventBus = app.state.events

        async def stream() -> AsyncIterator[bytes]:
            async with bus.subscribe() as queue:
                yield b": connected\n\n"
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    except TimeoutError:
                        yield b": keepalive\n\n"
                        continue
                    payload = json.dumps(event.data)
                    yield f"event: {event.type}\ndata: {payload}\n\n".encode()

        return StreamingResponse(
            stream(), media_type="text/event-stream",
            # Buffering an event stream turns it into a batch delivery.
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(api)
    return app


def _require(app: FastAPI, ref: str) -> ConfigEntry:
    entry = app.state.store.get(ref)
    if entry is None:
        raise HTTPException(404, f"no configuration {ref!r}")
    return entry


def _engine_out(app: FastAPI) -> EngineOut:
    manager: Manager = app.state.manager
    record = manager.supervisor.record
    if record is None:
        return EngineOut(state=str(manager.supervisor.state))
    failure = record.failure
    return EngineOut(
        state=str(record.state), config=record.config_name, port=record.port,
        pid=record.pid, startup_seconds=record.startup_seconds,
        compile_state=manager.supervisor.compile_state,
        failure_kind=None if failure is None else str(failure.kind),
        failure_summary=None if failure is None else failure.summary,
        facts=dict(record.facts), reclaimed_shm=list(record.reclaimed_shm),
    )


def _default_store() -> Path:
    from vllm_untwisted.cli import DEFAULT_STORE
    return DEFAULT_STORE
