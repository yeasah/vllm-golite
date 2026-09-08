"""Run the API on a loopback port, for the length of one command.

The obvious shortcut for a CLI that wants to be a client without requiring a server is
to mount the ASGI app in-process. It does not work here, and the reason is worth
recording: httpx's ASGI transport buffers a whole response before returning it, so an
endpoint that never completes -- which is exactly what an event stream is -- hangs
forever. Half the API would work and the interesting half would not.

Rather than keeping two ways to learn what happened (stream when remote, poll when
local), the client starts a real server and speaks to it over a socket. Same transport
as production, streaming included, and no second code path to keep honest. It costs a
few hundred milliseconds and a loopback port.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

import uvicorn
from fastapi import FastAPI


@contextlib.asynccontextmanager
async def running_server(app: FastAPI, host: str = "127.0.0.1") -> AsyncIterator[str]:
    """Serve `app` on an ephemeral port and yield its base URL."""
    config = uvicorn.Config(app, host=host, port=0, log_level="warning",
                            lifespan="off", access_log=False)
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve(), name="untwisted-api")
    try:
        while not server.started:
            if task.done():
                await task  # surface whatever stopped it
                raise RuntimeError("server exited before it started")
            await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        yield f"http://{host}:{port}"
    finally:
        server.should_exit = True
        with contextlib.suppress(asyncio.CancelledError, TimeoutError):
            await asyncio.wait_for(task, timeout=10.0)
