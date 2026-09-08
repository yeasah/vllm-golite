"""The contract. Everything else is a client of it, including the CLI."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

from vllm_untwisted.api import create_app
from vllm_untwisted.api.events import QUEUE_DEPTH, EventBus
from vllm_untwisted.engine import EngineConfig, Supervisor
from vllm_untwisted.manager import Manager
from vllm_untwisted.store import Store

FAKE = str(Path(__file__).parent / "fake_engine.py")
SCRIPT = "# long context\nvllm serve /ckpt/qwen --max-num-seqs 1\n"


@pytest.fixture
def app(tmp_path):
    store = Store(tmp_path / "s.db")
    sup = Supervisor(start_timeout=15.0, poll_interval=0.05, shm_root=str(tmp_path))
    app = create_app(store, Manager(store, sup, fingerprint={"test": "1"}))
    yield app
    store.close()


@pytest.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        yield c


def fake_config(*extra: str) -> dict:
    return {"name": "fake", "model": "a-model", "args": list(extra),
            "launcher": [sys.executable, FAKE]}


async def test_health(client):
    assert (await client.get("/api/health")).json() == {"status": "ok"}


async def test_configurations_round_trip_over_http(client):
    r = await client.post("/api/configs", json=fake_config("--max-num-seqs", "1"))
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "draft"
    assert body["command_line"].endswith("--max-num-seqs 1")

    assert (await client.get(f"/api/configs/{body['id']}")).json()["name"] == "fake"
    assert len((await client.get("/api/configs")).json()) == 1


async def test_a_duplicate_name_is_a_conflict(client):
    await client.post("/api/configs", json=fake_config())
    assert (await client.post("/api/configs", json=fake_config())).status_code == 409


async def test_a_missing_configuration_is_a_404(client):
    assert (await client.get("/api/configs/nope")).status_code == 404
    assert (await client.patch("/api/configs/nope", json={"name": "x"})).status_code == 404


async def test_import_takes_content_not_a_path(client):
    """The manager runs in a container and cannot open the caller's files."""
    r = await client.post("/api/configs/import-sh",
                          json={"filename": "/elsewhere/run-x.sh", "content": SCRIPT})
    assert r.json()["added"] == ["long-context"]
    assert (await client.get("/api/configs/long-context")).status_code == 200


async def test_a_dry_run_stores_nothing(client):
    r = await client.post("/api/configs/import-sh",
                          json={"filename": "x.sh", "content": SCRIPT, "dry_run": True})
    assert r.json()["added"] == ["long-context"]
    assert (await client.get("/api/configs")).json() == []


async def test_rename_and_delete(client):
    cid = (await client.post("/api/configs", json=fake_config())).json()["id"]
    assert (await client.patch(f"/api/configs/{cid}",
                               json={"name": "renamed"})).json()["name"] == "renamed"
    assert (await client.delete(f"/api/configs/{cid}")).status_code == 204
    assert (await client.get("/api/configs")).json() == []


async def test_lint_is_served(client):
    await client.post("/api/configs", json={
        "name": "pinned", "model": "/m", "args": ["--kv-cache-memory=123"]})
    findings = (await client.get("/api/lint")).json()
    assert findings["pinned"][0]["rule"] == "pinned-kv-cache-memory"


async def test_starting_returns_at_once_rather_than_blocking(client):
    """A cold start is tens of seconds and a fresh container's first is over a minute.
    No caller holds a request open for that."""
    await client.post("/api/configs", json=fake_config("--ready-delay", "1.0"))
    r = await client.post("/api/engine/start", json={"ref": "fake"})
    assert r.status_code == 202
    assert r.json()["state"] == "starting"
    # The engine is demonstrably not up yet.
    assert (await client.get("/api/engine")).json()["state"] in ("starting", "stopped")
    await client.post("/api/engine/stop")


async def test_the_outcome_arrives_on_the_event_stream(app):
    """Over a real socket, not the in-process transport: httpx's ASGI transport buffers
    a whole response before returning it, so an endpoint that never completes -- which
    is what a stream is -- would hang here forever."""
    from vllm_untwisted.api.serve import running_server

    async with running_server(app) as url, httpx.AsyncClient(base_url=url) as client:
        await client.post("/api/configs", json=fake_config())

        async def watch() -> dict:
            async with client.stream("GET", "/api/events", timeout=20.0) as stream:
                event = None
                async for line in stream.aiter_lines():
                    if line.startswith("event: "):
                        event = line.removeprefix("event: ").strip()
                    elif line.startswith("data: ") and event == "engine.state":
                        data = json.loads(line.removeprefix("data: "))
                        if data.get("state") == "ready":
                            return data
            raise AssertionError("stream ended without a ready state")

        task = asyncio.create_task(watch())
        await asyncio.sleep(0.3)  # subscribe before the start publishes
        await client.post("/api/engine/start", json={"ref": "fake"})
        data = await asyncio.wait_for(task, timeout=25)
        assert data["port"] and data["startup_seconds"] > 0
        await client.post("/api/engine/stop")


async def test_the_in_process_transport_cannot_stream(app):
    """Recorded as a test because it is the reason the CLI starts a real server.

    httpx's ASGITransport awaits the application to completion and asserts the response
    is complete before returning. An event stream never completes."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        with pytest.raises((TimeoutError, asyncio.TimeoutError)):
            await asyncio.wait_for(c.get("/api/events"), timeout=2.0)


async def test_starting_twice_is_a_conflict(client):
    await client.post("/api/configs", json=fake_config("--ready-delay", "2.0"))
    await client.post("/api/engine/start", json={"ref": "fake"})
    await asyncio.sleep(0.1)
    assert (await client.post("/api/engine/start",
                              json={"ref": "fake"})).status_code == 409
    await client.post("/api/engine/stop")


async def test_a_run_is_recorded_through_the_api(app, client):
    await client.post("/api/configs", json=fake_config())
    await client.post("/api/engine/start", json={"ref": "fake"})
    for _ in range(200):
        if (await client.get("/api/engine")).json()["state"] == "ready":
            break
        await asyncio.sleep(0.05)
    await client.post("/api/engine/stop")

    runs = (await client.get("/api/configs/fake/runs")).json()
    assert len(runs) == 1 and runs[0]["became_ready"]
    assert runs[0]["fingerprint"] == {"test": "1"}
    assert (await client.get("/api/configs/fake")).json()["status"] == "known-good"


async def test_a_slow_subscriber_drops_events_rather_than_growing():
    """A client that stops reading must cost bounded memory. Losing a log line is a
    nuisance; a manager growing until it is killed is an outage."""
    bus = EventBus()
    async with bus.subscribe() as queue:
        for i in range(QUEUE_DEPTH + 50):
            bus.publish("engine.log", line=str(i))
        assert queue.qsize() <= QUEUE_DEPTH
        assert bus.dropped >= 50
        # The newest survived; the oldest went.
        newest = [queue.get_nowait() for _ in range(queue.qsize())][-1]
        assert newest.data["line"] == str(QUEUE_DEPTH + 49)


async def test_unsubscribing_stops_delivery():
    bus = EventBus()
    async with bus.subscribe():
        assert bus.subscribers == 1
    assert bus.subscribers == 0
    bus.publish("engine.log", line="nobody is listening")
