"""Read `vllm serve` invocations out of the shell scripts they currently live in.

The scripts already contain everything the store wants; it is only in a form nothing can
select. Each file holds one live invocation and several commented-out alternatives, each
introduced by a comment saying what it is for -- "3.00bpw w/turboquant, long context",
"4.00bpw, real tight" -- and grouped under banner comments that say whether the group has
been verified. Those are names and provenance, sitting in a place no tool can reach.

Two decisions worth stating, because both could reasonably go the other way:

**Commented-out invocations are imported too.** They are the alternatives the pile exists
to hold, and dropping them would lose most of the content. They arrive as ordinary
entries, and the store's own rule already puts them where they belong: never started,
therefore a draft.

**`export VAR=value` applies; bare `VAR=value` does not.** In `sh` an assignment on its
own line sets a shell variable *without* exporting it, so it never reaches `vllm`, while
`export` does and so does prefixing it to the command. The distinction is invisible on
sight and the scripts have had it both ways. So an export is folded into every
invocation below it, and a bare assignment is reported instead -- importing it would
quietly change what the configuration does, and dropping it silently would lose a knob
the author meant to set.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from vllm_untwisted.engine.config import EngineConfig

#: A banner comment: a run of #s, used in these scripts to head a group.
BANNER = re.compile(r"^#{4,}\s*(.*)$")
ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


@dataclass
class Imported:
    config: EngineConfig
    note: str
    active: bool
    source: str


@dataclass
class ImportResult:
    configs: list[Imported] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


#: Long enough to stay descriptive, short enough to type and to fit a table.
SLUG_MAX = 48


def _slug(text: str, fallback: str) -> str:
    text = re.sub(r"--?\s*\d+(\.\d+)?\s*t/s.*$", "", text)  # drop "-- 48t/s @ 187K"
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if len(slug) > SLUG_MAX:
        # Cut on a word boundary; a truncated word reads like a typo.
        slug = slug[:SLUG_MAX].rsplit("-", 1)[0]
    return slug or fallback


def _expand(value: str, path: Path, lineno: int, result: ImportResult) -> str:
    """Expand a leading `~`, because sh does and the engine will not.

    Left as a warning as well as an expansion: the result is now specific to whoever ran
    the import, which is a thing to know when a stored configuration moves.
    """
    if value.startswith("~"):
        expanded = str(Path(value).expanduser())
        result.warnings.append(
            f"{path.name}:{lineno}: expanded `{value}` to `{expanded}`; sh would have "
            f"done this and the engine would not.")
        return expanded
    return value


def parse(path: str | Path) -> ImportResult:
    path = Path(path)
    result = ImportResult()
    banner = ""
    comment = ""
    #: `export`ed variables apply to every invocation after them in the file.
    exported: dict[str, str] = {}
    seen: set[str] = set()

    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#!"):
            continue

        if m := BANNER.match(line):
            banner, comment = m.group(1).strip(), ""
            continue

        commented = line.startswith("#")
        body = line.lstrip("#").strip() if commented else line
        if not body:
            continue

        if not commented and body.startswith("unset "):
            result.warnings.append(
                f"{path.name}:{lineno}: `{body}` clears an ambient variable. The store "
                f"has no unset; untwisted controls the base environment instead.")
            continue

        exporting = not commented and body.startswith("export ")
        if exporting:
            body = body.removeprefix("export ").strip()

        try:
            tokens = shlex.split(body)
        except ValueError:
            if commented:
                comment = body  # an ordinary prose comment that happens to hold a quote
            else:
                result.warnings.append(f"{path.name}:{lineno}: unparsable")
            continue

        # Peel leading NAME=value tokens. This has to happen *after* tokenizing: a
        # command line prefixed with assignments starts with the same text as a bare
        # assignment, and a regex over the whole line cannot tell them apart.
        env: dict[str, str] = {}
        i = 0
        while i < len(tokens) and (m := ASSIGNMENT.match(tokens[i])):
            env[m.group(1)] = _expand(m.group(2), path, lineno, result)
            i += 1
        rest = tokens[i:]

        if not rest:
            # Assignments and nothing else.
            if exporting:
                exported.update(env)
            elif env:
                names = ", ".join(env)
                result.warnings.append(
                    f"{path.name}:{lineno}: `{names}` is assigned without `export`, so "
                    f"sh does not pass it to the engine. Not imported -- add `export`, "
                    f"or prefix it to the command, if it was meant to apply.")
            elif commented:
                comment = body
            continue

        if rest[:2] != ["vllm", "serve"]:
            if commented:
                comment = body  # an ordinary comment; the next invocation's name
            continue

        if len(rest) < 3:
            result.warnings.append(f"{path.name}:{lineno}: no model argument")
            continue

        model, args = rest[2], [_expand(a, path, lineno, result) for a in rest[3:]]
        env = {**exported, **env}  # a prefix on the command line wins

        # --port belongs to the supervisor: a stored config that pins one cannot run twice.
        if "--port" in args:
            j = args.index("--port")
            del args[j : j + 2]
            result.warnings.append(
                f"{path.name}:{lineno}: dropped --port; the supervisor assigns it")

        name = _slug(comment, f"{path.stem}-{lineno}")
        while name in seen:
            name = f"{name}-{lineno}"
        seen.add(name)

        note = " | ".join(filter(None, [
            comment, f"section: {banner}" if banner else "",
            "commented out in source" if commented else "active in source",
            f"from {path.name}:{lineno}",
        ]))
        result.configs.append(Imported(
            config=EngineConfig(name=name, model=model, args=args, env=env),
            note=note, active=not commented, source=f"{path}:{lineno}",
        ))
        comment = ""

    return result
