# The manager is the commodity; the sizing is the product

golite serves one engine on one box and makes it easy to point a client at. That
much exists five times over. What does not exist anywhere is the thing that
decides *what the engine's command line should be* -- and on the hardware this is
aimed at, that decision is the whole difference between a model that runs and one
that does not.

This note records the initial design and the reasoning behind the calls, so the
ones made for a reason survive contact with the first refactor.

## Scope, and the boundary below it

golite is downstream of `vllm-exl3-plugin` and its siblings. It does not pull
stock vLLM: the engine environment is the patched fork plus the plugins, which is
why "pull an image, pass a model id" -- vLLMManager's model -- does not transfer.
The container *is* the engine environment.

The corollary is that model selection cannot be "any HuggingFace model." It has to
route quant format to backend, and know which formats the shipped plugins serve.
That routing table is appliance knowledge; nothing upstream has it.

## What the neighbours do, and the seam between them

| project | what it does | what to take |
|---|---|---|
| [Easy-vLLM](https://github.com/inboxpraveen/Easy-vLLM) | static page: parses `config.json`, estimates fit, emits a `vllm serve` line | the *framing* -- the output is a verdict ("77 concurrent at 5,120 tokens"), not a flag form |
| [vLLMManager](https://github.com/ddunford/vLLMManager) | React/Express/SQLite, one Docker container per instance | port-range allocation; config persisted so state survives a manager restart |
| [vllm-playground](https://github.com/micytao/vllm-playground) | FastAPI UI, omni/multimodal in a minimal env | the engine runtime abstracted three ways: subprocess \| container \| remote |
| [guidellm](https://github.com/vllm-project/guidellm) | load generator; TTFT/ITL distributions, sweep mode | take as a dependency. It measures and explicitly does not recommend |
| [vllm-tuner](https://github.com/kryvokhyzha/vllm-tuner) | Optuna over `gpu_memory_utilization`, `max_num_seqs`, `max_num_batched_tokens` | the trial lifecycle, not the optimizer -- it is the supervisor's state machine |
| [llama.cpp fit-params](https://github.com/ggml-org/llama.cpp/tree/master/tools/fit-params) | proposes params, allocates, refines; prints fitted args to stdout | fitting belongs in a *library*; the CLI is a thin printer |

The seam: llama-swap answers "which process should be running," Easy-vLLM answers
"will it fit" (statically, in a browser), and **nothing joins them.** llama-swap's
config is a hand-written `cmd:` line per model -- the user still writes the
`vllm serve` invocation themselves, which is the two-hundred-flag problem Easy-vLLM
named and did not solve for anyone actually serving. golite computes the command
llama-swap makes you type.

`llama_params_fit()` is the closest prior art, and the difference is what makes
this hard: llama.cpp can iterate cheaply because ggml graph allocation is
computable for a given `-ngl`/`-c` without running the model. vLLM has no such
path. Its profiler runs a real forward pass, so **every candidate costs a whole
process launch** (see the leak below -- an engine start cannot be amortized by
reusing a process), and the profiler is least reliable exactly at the margin being
fitted against. That makes pre-launch shortlisting and result caching load-bearing
rather than nice.

## v1 serves one engine

Not a staging decision -- a scope decision, and it is the right one on this
hardware:

- Swap latency is real and is paid by the user in the foreground. Automatic
  swapping only makes sense single-user.
- A swap policy safe enough to enable would have to refuse to stop recently-used
  engines and refuse to launch unless the new engine fits. Under a memory budget
  tight enough to need golite at all, those two rules compose to *almost never
  launch*. The complexity buys a case that rarely fires.
- On-demand swapping is inseparable from multi-engine support, which is the
  complexity not worth introducing first.

**The latency does not disappear with the feature.** With one engine, start cost is
paid in full on every model change, and it becomes the appliance's dominant UX
cost -- the number that decides whether "change models" reads as a setting or an
outage. It is unmeasured (see `TODO: start-latency`).

### Deliberately deferred, and the shape each should take

- **Multi-engine.** Deferred, not foreclosed. The cost of foreclosing it is one
  mistake: letting "the engine" become a global singleton. An engine record with an
  id, and a router with a table that happens to hold one row, cost nothing now.
- **A small always-resident embedding model.** The one co-residency case with
  demonstrated use. It is *not* multi-engine and must not be built as it: it is a
  **budget line item** -- subtracted off the top before the generative model is
  sized -- never evicted, never scheduled, never arbitrated. Designing to that
  asymmetry avoids growing a scheduler.
- **Autotuning.** vllm-tuner's loop over a supervisor that already exists. For an
  audience at low utilization, fitting at all dominates throughput tuning.
- **Multi-node.** Out of scope. The `EngineRuntime` seam is what keeps it possible.

## One container, engines as child processes, one endpoint

**Single container.** Separate per-engine containers are not merely out of scope,
they are wrong here: the value proposition is a shared, correct memory budget on
one box, and once each engine is its own container, each believes it owns the GPU
and nothing arbitrates. The ledger has to be built either way -- containers only
remove the ability to see into the processes. Sharing the HF cache is also free
in-container.

**Engines as supervised child processes, not in-process.** In-process couples an
engine segfault to the manager, and the manager is what has to notice and report
it. Put a thin `EngineRuntime` interface at that boundary with a subprocess
implementation only.

**This is forced, not merely preferred: vLLM leaks in-process across engine
starts**, badly enough that serving in-process is not available even if the
supervision argument went the other way. The rule that follows is short and will be
violated by the first person who tries to make the fit loop faster:

> **One engine start per process, always.** No process is reused across engine
> constructions -- not in the supervisor, not in the fit loop, not in the
> characterizer.

That forecloses the obvious optimization on `fit-shortlist`'s successor: keeping a
warm process and re-constructing the engine with new parameters to skip interpreter
start, imports and CUDA context creation. It is not available. Budget a full process
launch per candidate configuration and design the loop around that cost rather than
around removing it.

**Open question, and it is an appliance question:** does the leak have a
per-*request* component, or is it only per-engine-construction? An appliance is
supposed to run unattended for weeks. If a long-lived engine process drifts, the
supervisor needs uptime and RSS monitoring with a restart policy, and that is a v1
feature rather than a later one.

The prior is that it does not -- a per-request leak would hit ordinary vLLM operators
hard enough to have been found. That prior is good but weaker for us than for them:
**we do not run their code.** A fork plus several plugins with custom kernels and
cache dtypes is exactly where a leak survives that mainstream serving would have
flushed out. Which also cuts the other way -- if it is ours, it is both more
tractable and more our job.

**Fixing rather than routing around it is on the table**, since a fork is already
being carried for EXL3. What prices that work is `TODO: start-latency`: the leak's
cost is process launch minus engine construction, paid per engine start. If weight
load dominates, fixing buys little; if interpreter start, imports and CUDA context
dominate, it buys the fit loop and every model change. Measure before committing.

**One OpenAI-compatible endpoint, not a port per instance.** vLLMManager's
port-per-instance leaks manager state into every client config. A stable address
that survives engine restarts means openwebui and agent configs are not rewritten
on every model change, and gives somewhere to answer a coherent 503 when nothing
is loaded rather than connection-refused. It is also the seam multi-engine lands
on later, which is the second reason to have it on day one.

Two ways to get the router wrong, both of which look fine in a smoke test:

- **Buffering SSE.** Pass streamed chunks through unbuffered or every token
  arrives in a clump.
- **Swallowing client disconnects.** vLLM aborts generation when the client goes
  away. A proxy that does not propagate the disconnect leaves orphaned requests
  generating on a box with no spare GPU to burn.

## Fit is four tiers, because the cheap ones are structurally blind

The static tier cannot be fixed by better arithmetic. Each of these is a startup
or mid-session fact, invisible to any amount of `config.json` reading (measured
2026-09-02, see re-verification note below):

- `gpu_memory_utilization` is a fraction of *total* but must fit within *free*.
  The ceiling is the ratio of two numbers only the startup line prints
  (15.28/15.51 = 0.985, hence 0.98). The gap is driver/context overhead and is not
  recoverable.
- vLLM's memory profiler can fail at a configuration **strictly cheaper** than one
  that works: `max_num_batched_tokens` 128 profiles and serves, 256 over-allocates
  and OOMs, over a true difference of 61 MiB against a 0.78 GiB margin.
  Reproducible, so not allocator noise.
- `enforce_eager` can cost *more* memory than CUDA graphs. Qwen3.8-27B in 15.9 GiB:
  graphs reported 0.04 GiB of capture memory and usable KV went **up**, 46K to 67K
  tokens. The common assumption that graphs are the memory-hungry option and eager
  the safe fallback is not reliable; price it.

And the launch tier is not sufficient either, which is the tier most likely to be
skipped: **a transient that scales with cached context is invisible to the
profiler**, which varies `max_num_batched_tokens` and nothing else. vLLM reports
the same peak activation for a 4K session and a 130K one. Configs of that shape OOM
*mid-session*, and no short test prompt can detect it. `max_num_seqs` has the same
signature from the other direction -- essentially zero static cost, later OOM
because the scheduler admits a batch the config was never sized for.

| tier | cost | answers | instrument |
|---|---|---|---|
| 0 -- headers | kilobytes, no GPU, no download | can this checkpoint be split at the degrees this box has? is it worth the bandwidth? | `tp_preflight --remote`, `checkpoint_survey` |
| 1 -- arithmetic | a `config.json` read | which candidates are worth spending a launch on? | new; Easy-vLLM's formula is the floor, not the answer |
| 2 -- allocation | one **process launch** per candidate | does it start, and how much KV does it *actually* get? | vLLM's own startup reporting |
| 3 -- depth | a load run per survivor | does it survive a full-context session at N concurrent? | guidellm as generator; `memprof` on failure |

Tier 3 is the appliance verdict and the one nobody else produces. It is also
guidellm's real job here -- not throughput numbers, but walking context depth and
concurrency until the mid-session cliff appears.

### Traps for the config generator

- **Do not apply vLLM's own `--kv-cache-memory=` suggestion.** It subtracts CUDA
  graph memory and a deliberate 150 MiB redundancy buffer that the profiler did not
  count, so it lands routinely *below* what the running engine is already surviving
  on -- handing back cache the config demonstrably did not need to give up. Both
  numbers are self-consistent; neither is a bug. Verified to the reported digits on
  a 15.5 GiB card: 4.24 running, 4.03 "fully utilize", 3.95 "requested".
- **Pinning `--kv-cache-memory` suppresses the profile run and all memory
  reporting.** So the shrink above is one-shot -- unless the manager re-measures by
  unpinning and repinning, which turns it into a ratchet.
- **`--max-model-len auto` makes "Maximum concurrency: 1.00x" a tautology**, not an
  observation: `max_model_len` is set *to* the KV capacity. Any `max_num_seqs > 1`
  on such a config is overcommitted by construction.

## Most of tier 0 already exists

`vllm-exl3-plugin/tools/` holds four tools written to one philosophy -- screen
cheaply before spending something expensive (rental hours, bandwidth, GPU time).
golite is the first consumer that ties them to a serving decision rather than a
benchmarking one.

| tool | tier | why it is not replaceable by arithmetic |
|---|---|---|
| `checkpoint_survey.py` | 0 | screens a checkpoint before the bandwidth |
| `tp_preflight.py` | 0 | EXL3's Hadamard transform is block-diagonal in 128s, so a tensor may only be cut on a multiple of 128 per rank -- against *stored* dimensions, which are not the model's, because exllamav3 pads before quantizing. `config.json` cannot answer it |
| `host_survey.py` | 0 | stdlib-only, runs on a bare box before torch exists |
| `memprof.py` | 3 | attributes a snapshot's *peak* to call sites, and ranks sites by largest single allocation -- "which is what a margin has to cover" |

**Share, do not move.** The plugin needs these for benchmark provenance, and its
boundary note says appliance-specific reasoning must not leak upward. A small
extracted package both repos depend on keeps the plugin's docstrings honest about
why the tools exist. See `TODO: preflight-share`.

One precision on reuse: `host_survey`'s output-relevant-vs-throughput-only split is
calibrated for **token reproducibility**. Fit needs a different list -- VRAM, GPU
count, driver, vLLM and plugin version matter; PCIe width and host CPU do not.
Overlapping, not identical. What transfers is the tool's actual innovation: keeping
that list explicit and checkable rather than remembered. The fit list is also the
cache key for tier 2 and 3 results, which is what makes per-candidate engine starts
affordable across sessions.

## The transient characterizer

Transients cannot be characterized once and shipped as a table -- not every
constellation on every system. The tool goes *in* the appliance. Both halves exist
as working code: `tools/memprof.py` reduces a snapshot, and `~/ckpt/profile-completion.py`
gathers one. What they are not yet is a thing that predicts.

What the gathering half already gets right, and must keep:

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set **before** the torch import.
- The record window bounded to a single `chat()` call --
  `_record_memory_history(clear_history=True)` immediately before, `_dump_snapshot`
  and disable immediately after.
- A `prompt_prefix` that busts the prefix cache. Without it a repeat run hits cache
  and never allocates the transient being measured.

What has to change:

- **Sweep depth, regress the slope.** It measures one depth per edit-and-run. Peak
  at one depth does not predict mid-session OOM; bytes-per-token does. Run the same
  config at several `target_toks` and fit per call site -- that is what turns "this
  site took 300 MB" into "this site needs 786 MB at your 128K `max_model_len`."
- **Persist the reduction, not the snapshot.** A ~45K-token prompt produces ~700K
  allocation records and a 138K one over 3M; at full detail the standard viewer
  OOMs a browser VM. The appliance keeps the per-site slopes and discards the pickle.
- **Parameters, not commented-out blocks.** The config under test is currently
  chosen by editing the file.

Two constraints to design around rather than discover:

- **It requires a launch topology the appliance cannot serve with.** Profiling
  needs `VLLM_ENABLE_V1_MULTIPROCESSING=0` to keep vLLM in-process, and the
  in-process leak means that mode cannot be the serving mode. So characterization is
  permanently a distinct mode, and **whether a profile taken in-process transfers to
  the multiprocessing path is assumed, not shown.** The gap cannot be closed by
  choosing to serve in-process -- that option does not exist -- so it has to be
  closed by measurement or lived with knowingly.
- **The sweep splits across two axes, and only one of them fits in a process.**
  Depth is in-process: one engine start, several `chat()` calls at increasing
  `target_toks`, a record window around each -- which is the shape
  `profile-completion.py` already has. Configuration is not: each candidate config is
  a new engine construction and therefore a new process. Do not let the config axis
  migrate inside the process to save time.
- **`_record_memory_history` is not free.** This is a deliberate "characterize this
  config" mode, never always-on telemetry.

## Platform

Python backend (uvicorn/FastAPI) and a React frontend, held as weak preferences.
The one argument that is not a preference: a Python manager can import
`huggingface_hub` and `transformers` directly and reuse vLLM's own accounting,
where Easy-vLLM had to reimplement config parsing in JavaScript. Sizing is where
silent wrongness lives, and reimplementation is where it enters. Keep the manager
in its own venv from the engine's, in the same container.

`golite` is a working name.

## The shippable is the product's real entry point

This is a complicated system to bootstrap by hand -- a forked vLLM, a forked
exllamav3, several plugins, and a CUDA/torch stack that has to agree with all of
them. If a user has to assemble that, none of the rest matters. **The container is
not packaging applied at the end; it is the thing being shipped**, and the manager is
what runs inside it.

Contents: the vLLM fork, the exllamav3 fork, the plugins we choose to ship (EXL3
today, `kv-pager` and others if they land), the manager and frontend, on the CUDA and
torch versions the forks were built against. Models are **not** in the image -- the
HF cache is a volume, so the image is pullable and cacheable without them.

### Half of this is already built

`~/podman/vast-vllm` is a working podman-compose wheel builder: it produces both the
patched vLLM wheel and the exllamav3 wheel for cp312/cu130, broad-arch
(`TORCH_CUDA_ARCH_LIST='7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0'`), via
`pip wheel . --no-build-isolation --no-deps` into a mounted `/dist`. **Stage one
exists. The runtime image that installs those wheels is the new work.**

Two things to carry over from it, and one not to:

- **Carry: the wheel/image split.** Compiling in the shipped image would drag the
  CUDA toolchain into it. Build wheels in a builder, install them in a runtime image.
- **Carry: broad `TORCH_CUDA_ARCH_LIST`.** An appliance does not know the customer's
  card. Breadth costs build time and image size and is the right trade here.
- **Do not carry the base image.** vast-vllm builds on `vastai/pytorch:cuda-13.0.3-auto`
  because the wheels had to run on *vast's* runtime image. That was correct there and
  is meaningless here. What transfers is the invariant, not the tag: **the build base
  must agree with the runtime base**, and for golite we choose both.

[`vllm-fork/docker/Dockerfile`](../vllm-fork/docker/Dockerfile) is the reference for
the half that does not exist yet -- pinned `CUDA_VERSION`/`PYTHON_VERSION` (13.0.3,
3.12), wheels built against the same glibc floor as PyTorch's published wheels, and a
runtime lineage (`vllm-runtime-base` -> `vllm-base` -> `vllm-openai`) that keeps the
toolchain out of the serving image. golite's image belongs at **`vllm-base`, not
`vllm-openai`**: it replaces the entrypoint rather than wrapping it.

Read it as a reference, not as text to copy. The fork rebases on upstream; copied
stage text drifts silently, and this project has already paid three times over for
duplicating what upstream was maintaining.

It is also stale -- pinned to `VLLM_VERSION=v0.27.0` and applying
`vllm-exl3-plugin/patches/vllm-*.patch`, a directory that no longer exists now that
both dependencies are submodules and the vLLM fork sits on a branch named, aptly,
`appliance/v0.28.0`. That is the rental workflow's problem, not golite's: golite is
not blocked on it, and the pile is vast-specific enough that generalizing it would
cost more than starting from the invariants above.

### Two venvs, one container

The manager keeps its own venv, separate from the engine's, in the same image. The
container makes this nearly free, and it stops FastAPI/uvicorn/frontend dependencies
from perturbing an engine environment whose resolution is already delicate and gets
re-litigated on every vLLM bump.

### Podman is the native path here, not an alternative

The development box is CentOS Stream 10 with podman and `nvidia-ctk` installed and
**no docker at all**. The image is OCI and will run either place, but the commands
around it are not interchangeable: builds are `podman build`, and GPU access is CDI
(`nvidia-ctk cdi generate`, then `--device nvidia.com/gpu=all`) rather than docker's
`--gpus all`. Whatever run scripts and docs golite ships have to lead with the podman
form, because it is the only one we can actually test.

### The bootstrap floor, stated honestly

One host prerequisite cannot be containerized away: an NVIDIA driver and a working
container GPU path (toolkit or CDI). The image removes assembly complexity down to
that floor and no further. Say so in the README rather than letting a user discover
it after a multi-gigabyte pull.

### The image has to describe itself

The fit cache is keyed partly on vLLM and plugin versions, so the image must be able
to report exactly what it contains -- both fork commits, every shipped plugin's
version, torch and CUDA. This is also the only honest way to attach a measurement to
a build. Pin by submodule commit at build time and record it in the image.

## What is fresh, and what might not be

The whole project is early, and obviously so on contact; nothing here is offered at
higher confidence than that. The operating stance is **trust nothing** -- every
measurement quoted is dated, was taken against a moving vLLM, and may have been
overtaken by it.

The dates are not a disclaimer, they are the useful part. They are what lets us tell
a fact that was checked last week from one carried forward for a month, which is the
only distinction that matters while the tree underneath moves this fast. Keep dating
anything measured; re-verification happens when a decision leans hard on a number,
not on a schedule.
