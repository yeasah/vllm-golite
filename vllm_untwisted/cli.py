"""Command-line access to the configuration store.

Required from the first version rather than added later: replacing a pile of shell
scripts has to work before any frontend exists, and a store you can only reach through a
UI that is not written yet is not a replacement for anything. Export doubles as the
report surface and as the thing an owner can back up and commit.

This is deliberately *not* the general CLI the design note describes for the manager API
-- that one is a thin generic client over HTTP, and mirroring a UI's command surface is
the thing to avoid. This is the store, which exists first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from vllm_untwisted.engine import EngineState, Supervisor
from vllm_untwisted.engine.config import EngineConfig
from vllm_untwisted.store import Store
from vllm_untwisted.store.shell import parse

DEFAULT_STORE = Path(
    os.environ.get("UNTWISTED_STORE")
    or Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    / "vllm-untwisted" / "configs.db"
)

STATUS_MARK = {"known-good": "ok", "regressed": "!!", "draft": "--"}


def cmd_import_sh(store: Store, args: argparse.Namespace) -> int:
    total = 0
    for path in args.files:
        result = parse(path)
        for warning in result.warnings:
            print(f"warning: {warning}", file=sys.stderr)
        for item in result.configs:
            name = item.config.name
            if store.get(name) is not None:
                print(f"skip   {name} (already present)", file=sys.stderr)
                continue
            if args.dry_run:
                print(f"would add {name}")
            else:
                store.add(item.config, note=item.note)
                print(f"added  {name}")
            total += 1
    print(f"{total} configuration(s){' (dry run)' if args.dry_run else ''}",
          file=sys.stderr)
    return 0


def cmd_run(store: Store, args: argparse.Namespace) -> int:
    """What replaces `sh run-whatever.sh`."""
    import asyncio
    import contextlib

    from vllm_untwisted.manager import Manager

    async def go() -> int:
        manager = Manager(store, Supervisor(start_timeout=args.timeout))
        started = await manager.start(args.ref)
        record = started.record

        if not started.ok:
            failure = record.failure
            print(f"failed: {failure.kind} -- {failure.summary}", file=sys.stderr)
            for line in failure.log_tail[-15:]:
                print(f"  {line}", file=sys.stderr)
            await manager.stop()
            return 1

        print(f"ready on port {record.port} in {record.startup_seconds:.1f}s "
              f"(compile cache: {manager.supervisor.compile_state})")
        for reclaimed in record.reclaimed_shm:
            print(f"reclaimed abandoned shared memory: {reclaimed}", file=sys.stderr)
        for key in ("available_kv_cache_gib", "kv_cache_size_tokens",
                    "maximum_concurrency", "peak_activation_gib"):
            if key in record.facts:
                print(f"  {key:<26} {record.facts[key]}")

        if not args.once:
            print("\nserving; ctrl-c to stop", file=sys.stderr)
            with contextlib.suppress(KeyboardInterrupt):
                while manager.supervisor.state is EngineState.READY:
                    await asyncio.sleep(0.5)
        await manager.stop()
        return 0

    return asyncio.run(go())


def cmd_ls(store: Store, args: argparse.Namespace) -> int:
    entries = store.list()
    if not entries:
        print("no configurations", file=sys.stderr)
        return 0
    width = max(len(e.name) for e in entries)
    for e in entries:
        runs = f"{e.run_count} run{'s' if e.run_count != 1 else ''}"
        last = ""
        if e.last_run and e.last_run.startup_seconds:
            last = f"  {e.last_run.startup_seconds:.0f}s start"
        print(f"{STATUS_MARK.get(e.status, '??')} {e.name:<{width}}  {e.id}  {runs}{last}")
    return 0


def cmd_show(store: Store, args: argparse.Namespace) -> int:
    entry = store.get(args.ref)
    if entry is None:
        print(f"no configuration {args.ref!r}", file=sys.stderr)
        return 1
    print(f"{entry.name}  [{entry.id}]  {entry.status}")
    if entry.note:
        print(f"  note     {entry.note}")
    print(f"  origin   {entry.origin}" + (f" by {entry.derived_by}" if entry.derived_by else ""))
    print(f"  created  {entry.created_at}")
    print()
    print(entry.config.command_line())
    runs = store.runs(entry.id, limit=args.runs)
    if runs:
        print(f"\n{len(runs)} most recent run(s):")
        for r in runs:
            detail = r.failure_kind or (f"{r.startup_seconds:.1f}s" if r.startup_seconds else "")
            print(f"  {r.started_at}  {r.outcome:<7} {detail}"
                  + (f"  compile={r.compile_state}" if r.compile_state else ""))
    return 0


def cmd_lint(store: Store, args: argparse.Namespace) -> int:
    from vllm_untwisted.store.lint import lint

    entries = [store.get(args.ref)] if args.ref else store.list()
    if entries == [None]:
        print(f"no configuration {args.ref!r}", file=sys.stderr)
        return 1
    found = 0
    for entry in entries:
        findings = lint(entry)
        if not findings:
            continue
        found += sum(f.severity == "warn" for f in findings)
        print(entry.name)
        for f in findings:
            print(f"  {f.severity}  {f.rule}: {f.message}")
    if not found:
        print("nothing to report", file=sys.stderr)
    return 1 if found else 0


def cmd_export(store: Store, args: argparse.Namespace) -> int:
    entries = [store.get(args.ref)] if args.ref else store.list()
    if entries == [None]:
        print(f"no configuration {args.ref!r}", file=sys.stderr)
        return 1
    doc = [{"name": e.name, "note": e.note, "origin": e.origin,
            "status": e.status, **e.config.to_doc()} for e in entries]
    print(json.dumps(doc, indent=2))
    return 0


def cmd_import_json(store: Store, args: argparse.Namespace) -> int:
    doc = json.loads(Path(args.file).read_text())
    for item in doc:
        name = item["name"]
        if store.get(name) is not None:
            print(f"skip   {name} (already present)", file=sys.stderr)
            continue
        store.add(EngineConfig.from_doc(name, item), note=item.get("note"))
        print(f"added  {name}")
    return 0


def cmd_rename(store: Store, args: argparse.Namespace) -> int:
    store.rename(args.ref, args.name)
    return 0


def cmd_rm(store: Store, args: argparse.Namespace) -> int:
    store.delete(args.ref)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="untwisted", description=__doc__.splitlines()[0])
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE,
                    help=f"path to the store (default: {DEFAULT_STORE})")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("import-sh", help="read invocations out of shell scripts")
    p.add_argument("files", nargs="+")
    p.add_argument("-n", "--dry-run", action="store_true")
    p.set_defaults(func=cmd_import_sh)

    p = sub.add_parser("run", help="start a stored configuration")
    p.add_argument("ref", help="name or id")
    p.add_argument("--once", action="store_true",
                   help="stop as soon as it is healthy (a warmup or measurement pass)")
    p.add_argument("--timeout", type=float, default=900.0)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("ls", help="list configurations")
    p.set_defaults(func=cmd_ls)

    p = sub.add_parser("show", help="show one configuration and its runs")
    p.add_argument("ref", help="name or id")
    p.add_argument("--runs", type=int, default=5)
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("lint", help="check configurations against known traps")
    p.add_argument("ref", nargs="?")
    p.set_defaults(func=cmd_lint)

    p = sub.add_parser("export", help="write configurations as JSON")
    p.add_argument("ref", nargs="?")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("import-json", help="read configurations from JSON")
    p.add_argument("file")
    p.set_defaults(func=cmd_import_json)

    p = sub.add_parser("rename", help="give a configuration a new name")
    p.add_argument("ref")
    p.add_argument("name")
    p.set_defaults(func=cmd_rename)

    p = sub.add_parser("rm", help="delete a configuration and its runs")
    p.add_argument("ref")
    p.set_defaults(func=cmd_rm)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with Store(args.store) as store:
        try:
            return args.func(store, args)
        except (KeyError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(main())
