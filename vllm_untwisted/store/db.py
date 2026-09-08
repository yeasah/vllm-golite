"""Named configurations, and the evidence that they work.

Replaces a directory of shell scripts whose alternatives live in comments. The point is
not storage -- a directory of files stores fine -- it is that a comment cannot say
whether it ever worked, on what, or when, so a pile of them rots silently. Here every
entry carries where it came from and what happened when it ran.

Two shapes in one store, as `docs/design.md` sets out:

- **Configurations** are few, sometimes hand-written, and change slowly.
- **Runs** are many, always machine-written, and only accumulate.

They are one object conceptually and two tables practically, and it is the second that
eventually justifies a database at all.

Identity is an opaque id, not the name. Runs and derived configurations reference a
configuration for as long as it exists, and a collection large enough to be worth having
is one whose names get revised. A name is a unique handle, and a mutable one.

The document is JSON in a column with a few things promoted beside it for querying, so
the shape can change without a migration for every field. Migrations then cost something
only when a field is promoted, which is the point at which one is warranted anyway.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from vllm_untwisted.engine.config import EngineConfig

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS configs (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,
    doc          TEXT NOT NULL,
    origin       TEXT NOT NULL,
    derived_by   TEXT,
    derived_from TEXT REFERENCES configs(id) ON DELETE SET NULL,
    note         TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id              TEXT PRIMARY KEY,
    config_id       TEXT NOT NULL REFERENCES configs(id) ON DELETE CASCADE,
    started_at      TEXT NOT NULL,
    outcome         TEXT NOT NULL,
    failure_kind    TEXT,
    failure_summary TEXT,
    startup_seconds REAL,
    compile_state   TEXT,
    facts           TEXT NOT NULL,
    fingerprint     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_by_config ON runs (config_id, started_at DESC);
CREATE INDEX IF NOT EXISTS runs_by_outcome ON runs (config_id, outcome);
"""

#: A configuration that has never started successfully. Most of the cure for the rot the
#: shell scripts have: a commented-out invocation and a working one look identical.
DRAFT = "draft"
#: Has started at least once.
KNOWN_GOOD = "known-good"
#: Has started before, and its most recent start failed.
REGRESSED = "regressed"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class RunRecord:
    id: str
    config_id: str
    started_at: str
    outcome: str
    failure_kind: str | None
    failure_summary: str | None
    startup_seconds: float | None
    compile_state: str | None
    facts: dict[str, str]
    #: What the run happened on -- box and software versions. Opaque here on purpose:
    #: which fields belong in it is a fit question, and putting the policy in the schema
    #: would freeze an answer this project has not finished arguing about.
    fingerprint: dict[str, str]


@dataclass(frozen=True, slots=True)
class ConfigEntry:
    id: str
    name: str
    config: EngineConfig
    origin: str
    derived_by: str | None
    derived_from: str | None
    note: str | None
    created_at: str
    updated_at: str
    status: str
    run_count: int
    last_run: RunRecord | None

    @property
    def is_draft(self) -> bool:
        return self.status == DRAFT


class Store:
    """A configuration store on SQLite.

    Synchronous: writes are small and rare, and a manager that needs them off its event
    loop can hand a call to a thread. Pretending otherwise would buy nothing.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            # Survive a crash mid-write, which is the whole reason this is not a file.
            self.db.execute("PRAGMA journal_mode = WAL")
        self.db.executescript(SCHEMA)
        self.db.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- configurations ---------------------------------------------------------

    def add(
        self,
        config: EngineConfig,
        *,
        origin: str = "authored",
        derived_by: str | None = None,
        derived_from: str | None = None,
        note: str | None = None,
    ) -> str:
        """Store a configuration and return its id. The name must be free."""
        cid = uuid.uuid4().hex[:12]
        now = _now()
        try:
            self.db.execute(
                "INSERT INTO configs (id, name, doc, origin, derived_by, derived_from,"
                " note, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (cid, config.name, json.dumps(config.to_doc()), origin, derived_by,
                 derived_from, note, now, now),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"a configuration named {config.name!r} already exists") from exc
        self.db.commit()
        return cid

    def get(self, ref: str) -> ConfigEntry | None:
        """Look up by id or by name. Ids win, so a name shaped like an id cannot shadow."""
        row = self.db.execute("SELECT * FROM configs WHERE id = ?", (ref,)).fetchone()
        if row is None:
            row = self.db.execute("SELECT * FROM configs WHERE name = ?", (ref,)).fetchone()
        return None if row is None else self._entry(row)

    def list(self) -> list[ConfigEntry]:
        rows = self.db.execute("SELECT * FROM configs ORDER BY name").fetchall()
        return [self._entry(r) for r in rows]

    def rename(self, ref: str, new_name: str) -> None:
        """The reason identity is not the name."""
        entry = self._require(ref)
        try:
            self.db.execute("UPDATE configs SET name = ?, updated_at = ? WHERE id = ?",
                            (new_name, _now(), entry.id))
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"a configuration named {new_name!r} already exists") from exc
        self.db.commit()

    def update(self, ref: str, config: EngineConfig) -> None:
        """Replace the invocation. Runs already recorded stay attached, and stay true of
        what they measured -- which is why they carry their own fingerprint."""
        entry = self._require(ref)
        self.db.execute(
            "UPDATE configs SET doc = ?, name = ?, updated_at = ? WHERE id = ?",
            (json.dumps(config.to_doc()), config.name, _now(), entry.id),
        )
        self.db.commit()

    def delete(self, ref: str) -> None:
        entry = self._require(ref)
        self.db.execute("DELETE FROM configs WHERE id = ?", (entry.id,))
        self.db.commit()

    # -- runs -------------------------------------------------------------------

    def record_run(
        self,
        ref: str,
        *,
        outcome: str,
        facts: dict[str, str] | None = None,
        fingerprint: dict[str, str] | None = None,
        failure_kind: str | None = None,
        failure_summary: str | None = None,
        startup_seconds: float | None = None,
        compile_state: str | None = None,
    ) -> str:
        entry = self._require(ref)
        rid = uuid.uuid4().hex[:12]
        self.db.execute(
            "INSERT INTO runs (id, config_id, started_at, outcome, failure_kind,"
            " failure_summary, startup_seconds, compile_state, facts, fingerprint)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (rid, entry.id, _now(), outcome, failure_kind, failure_summary,
             startup_seconds, compile_state, json.dumps(facts or {}),
             json.dumps(fingerprint or {})),
        )
        self.db.commit()
        return rid

    def runs(self, ref: str, limit: int = 50) -> list[RunRecord]:
        entry = self._require(ref)
        rows = self.db.execute(
            "SELECT * FROM runs WHERE config_id = ? ORDER BY started_at DESC, rowid DESC"
            " LIMIT ?", (entry.id, limit)).fetchall()
        return [self._run(r) for r in rows]

    # -- internals --------------------------------------------------------------

    def _require(self, ref: str) -> ConfigEntry:
        entry = self.get(ref)
        if entry is None:
            raise KeyError(f"no configuration {ref!r}")
        return entry

    def _entry(self, row: sqlite3.Row) -> ConfigEntry:
        last = self.db.execute(
            "SELECT * FROM runs WHERE config_id = ? ORDER BY started_at DESC, rowid DESC"
            " LIMIT 1", (row["id"],)).fetchone()
        total = self.db.execute(
            "SELECT COUNT(*) c, SUM(outcome = 'ready') ok FROM runs WHERE config_id = ?",
            (row["id"],)).fetchone()
        ever_ready = (total["ok"] or 0) > 0
        if not ever_ready:
            status = DRAFT
        elif last is not None and last["outcome"] != "ready":
            status = REGRESSED
        else:
            status = KNOWN_GOOD
        return ConfigEntry(
            id=row["id"], name=row["name"],
            config=EngineConfig.from_doc(row["name"], json.loads(row["doc"])),
            origin=row["origin"], derived_by=row["derived_by"],
            derived_from=row["derived_from"], note=row["note"],
            created_at=row["created_at"], updated_at=row["updated_at"],
            status=status, run_count=total["c"] or 0,
            last_run=None if last is None else self._run(last),
        )

    @staticmethod
    def _run(row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            id=row["id"], config_id=row["config_id"], started_at=row["started_at"],
            outcome=row["outcome"], failure_kind=row["failure_kind"],
            failure_summary=row["failure_summary"],
            startup_seconds=row["startup_seconds"], compile_state=row["compile_state"],
            facts=json.loads(row["facts"]), fingerprint=json.loads(row["fingerprint"]),
        )

    def __iter__(self) -> Iterator[ConfigEntry]:
        return iter(self.list())
