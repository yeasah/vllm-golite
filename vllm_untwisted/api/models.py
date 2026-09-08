"""Request and response shapes.

Kept thin. The endpoints accrete against a real client rather than being designed up
front, so these describe what is actually exchanged today and are expected to move.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ConfigDoc(BaseModel):
    model: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    launcher: list[str] = Field(default_factory=lambda: ["vllm", "serve"])


class ConfigIn(BaseModel):
    name: str
    note: str | None = None
    model: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    launcher: list[str] = Field(default_factory=lambda: ["vllm", "serve"])


class ConfigOut(BaseModel):
    id: str
    name: str
    status: str
    origin: str
    note: str | None
    run_count: int
    created_at: str
    command_line: str
    doc: ConfigDoc


class RunOut(BaseModel):
    id: str
    started_at: str
    outcome: str
    became_ready: bool
    failure_kind: str | None
    startup_seconds: float | None
    compile_state: str | None
    facts: dict[str, str]
    fingerprint: dict[str, str]


class FindingOut(BaseModel):
    rule: str
    severity: str
    message: str


class RenameIn(BaseModel):
    name: str


class ImportShIn(BaseModel):
    #: The client reads the file and sends its contents: in a container the manager
    #: cannot see the caller's filesystem, and parsing stays server-side so there is one
    #: implementation of it.
    filename: str
    content: str
    #: Report what would be added without storing it.
    dry_run: bool = False


class ImportShOut(BaseModel):
    added: list[str]
    skipped: list[str]
    warnings: list[str]


class StartIn(BaseModel):
    ref: str


class EngineOut(BaseModel):
    state: str
    config: str | None = None
    config_id: str | None = None
    port: int | None = None
    pid: int | None = None
    startup_seconds: float | None = None
    compile_state: str | None = None
    failure_kind: str | None = None
    failure_summary: str | None = None
    facts: dict[str, str] = Field(default_factory=dict)
    reclaimed_shm: list[str] = Field(default_factory=list)
    #: Engines a previous manager left running and this one killed at startup.
    reclaimed_orphans: list[str] = Field(default_factory=list)
