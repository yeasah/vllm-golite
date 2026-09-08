"""A client of the API, not a second way in.

The rule from `docs/design.md`: the manager has no internal path that bypasses its own
API. This used to break it -- every command reached into the store directly -- which is
how a UI ends up with privileged access a CLI can never catch up to.

So every command here is an HTTP call. With `--url` it talks to a running manager;
without one it starts a server on a loopback port for the length of the command and
talks to that. Same handlers, same wire format, no requirement that a manager already be
running just to import a shell script -- and no second code path that would have to be
kept honest as the API grows.

This also stays deliberately small. It is not a mirror of every endpoint -- mirroring a
UI's command surface is what makes two surfaces a permanent tax -- it is the handful of
things done constantly, plus `api` for everything else.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import httpx

DEFAULT_STORE = Path(
    os.environ.get("UNTWISTED_STORE")
    or Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    / "vllm-untwisted" / "configs.db"
)
DEFAULT_URL = os.environ.get("UNTWISTED_URL")
STATUS_MARK = {"known-good": "ok", "regressed": "!!", "draft": "--"}


@contextlib.asynccontextmanager
async def connect(args: argparse.Namespace) -> AsyncIterator[httpx.AsyncClient]:
    if args.url:
        async with httpx.AsyncClient(base_url=args.url, timeout=30.0) as client:
            yield client
        return
    # No server given: run one here for the length of the command. Not an in-process
    # shortcut -- httpx's ASGI transport buffers whole responses, so the event stream
    # would hang, and half an API is worse than a server that costs 200ms to start.
    from vllm_untwisted.api import create_app
    from vllm_untwisted.api.serve import running_server
    from vllm_untwisted.store import Store

    store = Store(args.store)
    try:
        async with running_server(create_app(store)) as url:
            async with httpx.AsyncClient(base_url=url, timeout=30.0) as client:
                yield client
    finally:
        store.close()


def _fail(response: httpx.Response) -> int:
    detail = response.json().get("detail", response.text) if response.content else response.text
    print(f"error: {detail}", file=sys.stderr)
    return 1


async def cmd_import_sh(client: httpx.AsyncClient, args) -> int:
    total = 0
    for name in args.files:
        path = Path(name).resolve()
        # Read here, parse there: in a container the manager cannot open this file, and
        # parsing stays in one place so nothing can disagree about what a script means.
        r = await client.post("/api/configs/import-sh",
                              json={"filename": str(path), "content": path.read_text(),
                                    "dry_run": args.dry_run})
        if r.status_code >= 400:
            return _fail(r)
        body = r.json()
        for warning in body["warnings"]:
            print(f"warning: {warning}", file=sys.stderr)
        for skipped in body["skipped"]:
            print(f"skip   {skipped} (already present)", file=sys.stderr)
        for added in body["added"]:
            print(f"{'would add' if args.dry_run else 'added   '} {added}")
        total += len(body["added"])
    print(f"{total} configuration(s){' (dry run)' if args.dry_run else ''}",
          file=sys.stderr)
    return 0


async def cmd_ls(client: httpx.AsyncClient, args) -> int:
    entries = (await client.get("/api/configs")).json()
    if not entries:
        print("no configurations", file=sys.stderr)
        return 0
    width = max(len(e["name"]) for e in entries)
    for e in entries:
        runs = f"{e['run_count']} run{'s' if e['run_count'] != 1 else ''}"
        print(f"{STATUS_MARK.get(e['status'], '??')} {e['name']:<{width}}  {e['id']}  {runs}")
    return 0


async def cmd_show(client: httpx.AsyncClient, args) -> int:
    r = await client.get(f"/api/configs/{args.ref}")
    if r.status_code >= 400:
        return _fail(r)
    e = r.json()
    print(f"{e['name']}  [{e['id']}]  {e['status']}")
    if e["note"]:
        print(f"  note     {e['note']}")
    print(f"  origin   {e['origin']}")
    print(f"  created  {e['created_at']}")
    print()
    print(e["command_line"])
    runs = (await client.get(f"/api/configs/{args.ref}/runs",
                             params={"limit": args.runs})).json()
    if runs:
        print(f"\n{len(runs)} most recent run(s):")
        for run in runs:
            detail = run["failure_kind"] or (
                f"{run['startup_seconds']:.1f}s" if run["startup_seconds"] else "")
            compile_state = f"  compile={run['compile_state']}" if run["compile_state"] else ""
            print(f"  {run['started_at']}  {run['outcome']:<7} {detail}{compile_state}")
    return 0


async def cmd_lint(client: httpx.AsyncClient, args) -> int:
    findings = (await client.get("/api/lint")).json()
    if args.ref:
        findings = {k: v for k, v in findings.items() if k == args.ref}
    if not findings:
        print("nothing to report", file=sys.stderr)
        return 0
    warned = 0
    for name, items in findings.items():
        print(name)
        for f in items:
            warned += f["severity"] == "warn"
            print(f"  {f['severity']}  {f['rule']}: {f['message']}")
    return 1 if warned else 0


async def cmd_export(client: httpx.AsyncClient, args) -> int:
    entries = (await client.get("/api/configs")).json()
    if args.ref:
        entries = [e for e in entries if args.ref in (e["name"], e["id"])]
        if not entries:
            print(f"error: no configuration {args.ref!r}", file=sys.stderr)
            return 1
    print(json.dumps([{"name": e["name"], "note": e["note"], **e["doc"]} for e in entries],
                     indent=2))
    return 0


async def cmd_import_json(client: httpx.AsyncClient, args) -> int:
    for item in json.loads(Path(args.file).read_text()):
        r = await client.post("/api/configs", json=item)
        if r.status_code == 409:
            print(f"skip   {item['name']} (already present)", file=sys.stderr)
        elif r.status_code >= 400:
            return _fail(r)
        else:
            print(f"added  {item['name']}")
    return 0


async def cmd_rename(client: httpx.AsyncClient, args) -> int:
    r = await client.patch(f"/api/configs/{args.ref}", json={"name": args.name})
    return 0 if r.status_code < 400 else _fail(r)


async def cmd_rm(client: httpx.AsyncClient, args) -> int:
    r = await client.delete(f"/api/configs/{args.ref}")
    return 0 if r.status_code < 400 else _fail(r)


async def cmd_api(client: httpx.AsyncClient, args) -> int:
    """Reach any endpoint. What keeps this CLI from having to grow a command per feature."""
    body = json.loads(args.data) if args.data else None
    r = await client.request(args.method, args.path, json=body)
    print(r.text)
    return 0 if r.status_code < 400 else 1


async def cmd_run(client: httpx.AsyncClient, args) -> int:
    """Start an engine and watch the event stream until it settles.

    A start is tens of seconds cold and over a minute on a fresh container, so the API
    returns immediately and the outcome arrives here. This is what the stream is for.
    """
    r = await client.post("/api/engine/start", json={"ref": args.ref})
    if r.status_code >= 400:
        return _fail(r)

    outcome = 1
    async with client.stream("GET", "/api/events", timeout=None) as stream:
        event = None
        async for line in stream.aiter_lines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ").strip()
            elif line.startswith("data: ") and event == "engine.state":
                data = json.loads(line.removeprefix("data: "))
                state = data.get("state")
                if state == "ready":
                    print(f"ready on port {data['port']} in "
                          f"{data['startup_seconds']:.1f}s")
                    outcome = 0
                    break
                if state in ("failed", "stopped"):
                    print(f"failed: {data.get('failure_kind') or data.get('error')}",
                          file=sys.stderr)
                    break
            elif line.startswith("data: ") and event == "engine.log" and args.verbose:
                print(json.loads(line.removeprefix("data: "))["line"], file=sys.stderr)

    if outcome == 0:
        engine = (await client.get("/api/engine")).json()
        for key in ("available_kv_cache_gib", "kv_cache_size_tokens",
                    "maximum_concurrency", "peak_activation_gib"):
            if key in engine["facts"]:
                print(f"  {key:<26} {engine['facts'][key]}")
        for reclaimed in engine["reclaimed_shm"]:
            print(f"reclaimed abandoned shared memory: {reclaimed}", file=sys.stderr)
        if not args.once:
            print("\nserving; ctrl-c to stop", file=sys.stderr)
            with contextlib.suppress(KeyboardInterrupt):
                while (await client.get("/api/engine")).json()["state"] == "ready":
                    await asyncio.sleep(1.0)
    await client.post("/api/engine/stop")
    return outcome


def cmd_serve(args) -> int:
    """Run the manager. Not an API call -- it is what serves the API."""
    import uvicorn

    from vllm_untwisted.api import create_app
    from vllm_untwisted.store import Store

    uvicorn.run(create_app(Store(args.store)), host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="untwisted")
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ap.add_argument("--url", default=DEFAULT_URL,
                    help="a running manager; without it the API is mounted in-process")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="run the manager")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.set_defaults(sync=cmd_serve)

    p = sub.add_parser("run", help="start a stored configuration")
    p.add_argument("ref")
    p.add_argument("--once", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true", help="stream engine output")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("import-sh", help="read invocations out of shell scripts")
    p.add_argument("files", nargs="+")
    p.add_argument("-n", "--dry-run", action="store_true")
    p.set_defaults(func=cmd_import_sh)

    p = sub.add_parser("ls"); p.set_defaults(func=cmd_ls)

    p = sub.add_parser("show"); p.add_argument("ref")
    p.add_argument("--runs", type=int, default=5); p.set_defaults(func=cmd_show)

    p = sub.add_parser("lint"); p.add_argument("ref", nargs="?")
    p.set_defaults(func=cmd_lint)

    p = sub.add_parser("export"); p.add_argument("ref", nargs="?")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("import-json"); p.add_argument("file")
    p.set_defaults(func=cmd_import_json)

    p = sub.add_parser("rename"); p.add_argument("ref"); p.add_argument("name")
    p.set_defaults(func=cmd_rename)

    p = sub.add_parser("rm"); p.add_argument("ref"); p.set_defaults(func=cmd_rm)

    p = sub.add_parser("api", help="call any endpoint")
    p.add_argument("path")
    p.add_argument("-X", "--method", default="GET")
    p.add_argument("-d", "--data")
    p.set_defaults(func=cmd_api)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(args, "sync"):
        return args.sync(args)

    async def go() -> int:
        async with connect(args) as client:
            return await args.func(client, args)

    return asyncio.run(go())


if __name__ == "__main__":
    sys.exit(main())
