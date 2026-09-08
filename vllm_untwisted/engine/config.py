"""What gets launched: an argument vector and an environment.

The storage shape is settled in `docs/design.md` and is read off the shell scripts this
replaces. Arguments are a *list*, never a string: the invocations in `~/ckpt/run-*.sh`
carry multi-valued flags (`--cudagraph-capture-sizes 1 2 4`), JSON values with embedded
quotes (`--kv-transfer-config '{...}'`), and two syntaxes for passing a value
(`--kv-cache-memory=N` beside `--gpu-memory-utilization 0.97`). Any of those survives a
round trip through a list and none survives being re-split from a string.

`command_line()` exists because the *presentation* must stay a literal command line --
being able to paste one into a shell is what makes this a replacement for the scripts
rather than another place configurations get trapped.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field, replace


@dataclass(frozen=True, slots=True)
class EngineConfig:
    #: Human-chosen name. In the scripts this lived in a comment above the line,
    #: which is why nothing could select it.
    name: str
    #: The model as vLLM takes it: a checkpoint directory or a hub id.
    model: str
    #: Everything after `vllm serve <model>`, already split.
    args: list[str] = field(default_factory=list)
    #: The program in front of the model. A field rather than a constant because the
    #: shipped image may front it with a wrapper or a specific interpreter, and because
    #: tests need to point the whole lifecycle at something that is not vLLM.
    launcher: tuple[str, ...] = ("vllm", "serve")
    #: Environment *additions*. Some of these are load-bearing before the torch
    #: import (`PYTORCH_CUDA_ALLOC_CONF`), which is why they belong to the
    #: configuration rather than to the manager's ambient environment.
    env: dict[str, str] = field(default_factory=dict)

    def argv(self, port: int) -> list[str]:
        """The process to spawn. `--port` is the supervisor's to assign, not the
        configuration's -- a stored config that pins a port cannot be run twice."""
        return [*self.launcher, self.model, "--port", str(port), *self.args]

    def command_line(self, port: int | None = None) -> str:
        """The invocation as a human would write it, environment prefix included."""
        env = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(self.env.items()))
        argv = self.argv(port) if port is not None else [*self.launcher, self.model, *self.args]
        return " ".join(filter(None, [env, shlex.join(argv)]))

    def to_doc(self) -> dict:
        """The stored form. Plain JSON types only, and the same shape the CLI exports."""
        return {"model": self.model, "launcher": list(self.launcher),
                "args": list(self.args), "env": dict(self.env)}

    @classmethod
    def from_doc(cls, name: str, doc: dict) -> EngineConfig:
        return cls(name=name, model=doc["model"], args=list(doc.get("args", [])),
                   env=dict(doc.get("env", {})),
                   launcher=tuple(doc.get("launcher", ("vllm", "serve"))))

    def flag(self, name: str) -> str | None:
        """The value of `--name`, however it was written.

        Both syntaxes appear in the same script (`--kv-cache-memory=N` beside
        `--gpu-memory-utilization 0.97`), so anything reading a flag has to accept both.
        Returns "" for a flag present without a value, and None when absent -- which
        distinguishes `--enable-prefix-caching` from a flag that is not there.
        """
        opt = "--" + name.replace("_", "-")
        for i, arg in enumerate(self.args):
            if arg == opt:
                nxt = self.args[i + 1] if i + 1 < len(self.args) else None
                return "" if nxt is None or nxt.startswith("-") else nxt
            if arg.startswith(opt + "="):
                return arg.split("=", 1)[1]
        return None

    def with_args(self, **flags: str | None) -> EngineConfig:
        """Return a copy with `--flag value` set or removed.

        This is the operation the fit tiers exist to perform, and the reason args are
        stored structured: rewriting one flag by editing a rendered command line means
        parsing shell, and the syntax is not consistent enough to make that safe.
        """
        args = list(self.args)
        for flag, value in flags.items():
            opt = "--" + flag.replace("_", "-")
            # Drop any existing occurrence in either syntax.
            out: list[str] = []
            i = 0
            while i < len(args):
                if args[i] == opt:
                    i += 2 if i + 1 < len(args) and not args[i + 1].startswith("-") else 1
                    continue
                if args[i].startswith(opt + "="):
                    i += 1
                    continue
                out.append(args[i])
                i += 1
            args = out
            if value is not None:
                args += [opt, str(value)]
        return replace(self, args=args)
