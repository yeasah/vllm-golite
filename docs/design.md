# The manager is the commodity; the sizing is the product

untwisted serves one engine on one box and makes it easy to point a client at. That
much exists five times over. What does not exist anywhere is the thing that
decides *what the engine's command line should be* -- and on the hardware this is
aimed at, that decision is the whole difference between a model that runs and one
that does not.

This note records the initial design and the reasoning behind the calls, so the
ones made for a reason survive contact with the first refactor.

## The bugs worth fixing here are the ones that scale inversely with VRAM

0.4 GiB of misattributed activation is a rounding error on a datacenter card and 14% of
context on a 16 GiB one. The same defect, the same patch, and a completely different
case for spending a week on it.

That asymmetry is not a complaint about upstream -- it is a correct prioritization given
who upstream serves, and it will keep being correct. It is the reason this project finds
things worth fixing that are not worth fixing to anyone else, and the reason carrying a
fork is a standing cost rather than a temporary one.

Two working rules follow, and they pull in opposite directions:

- **Still check upstream first.** This stack has repeatedly paid for maintaining what
  upstream was already maintaining, and staleness has cost more than churn.
- **But do not wait on a fix whose magnitude only matters here.** When a defect's
  significance scales inversely with the reporter's VRAM budget, "reported upstream" is
  not a plan. Report it, then carry the workaround, and expect to carry it.

The corollary for measurement: the appliance has to be able to detect these itself,
because nobody else's CI will. That is most of the argument for the fit tiers being
instruments rather than estimators.

## Scope, and the boundary below it

untwisted is downstream of `vllm-exl3-plugin` and its siblings. It does not pull
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
named and did not solve for anyone actually serving. untwisted computes the command
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
  tight enough to need untwisted at all, those two rules compose to *almost never
  launch*. The complexity buys a case that rarely fires.
- On-demand swapping is inseparable from multi-engine support, which is the
  complexity not worth introducing first.

**The latency does not disappear with the feature.** With one engine, start cost is
paid in full on every model change, and it becomes the appliance's dominant UX
cost -- the number that decides whether "change models" reads as a setting or an
outage.

First measurement, through the supervisor on 2026-09-07 (Qwen3.8-27B EXL3 3.00bpw,
turboquant KV, one 16 GiB card): **74.2 s to healthy**, of which weight load was 2.6 s
and `torch.compile` plus the profiling/warmup run was **50.3 s**. Compilation, not
loading, is the cost -- and vLLM already breaks the phases down in its own startup log,
so `start-latency` is mostly a parsing job rather than an instrumentation one.

**And most of that cost is a development artifact, not an appliance one.** That run
inherited `VLLM_DISABLE_COMPILE_CACHE=1`, which is set on the development box because
plugin work invalidates the compile cache without changing anything the cache keys on --
a stale artifact is worse than a slow start. The shipped image has a frozen plugin set,
so it has no such problem: **the appliance should enable the compile cache.**

Measured on the same configuration, with `VLLM_DISABLE_COMPILE_CACHE` as the only
variable and every other cache left warm, interleaved cold/warm/cold/warm:

| | cold | warm |
|---|---|---|
| start to healthy | 32.1 s | **24.1 s** |
| compilation | 9.4 s | 0.4 s |
| peak activation, KV cache, resolved context | identical | identical |

A genuinely cold box is worse than the cold arm above -- the first start on this
machine took 74-81 s with compilation reported at 22-50 s, because torch's own inductor
cache (`/tmp/torchinductor_ypell`, 227 MB here) was cold too and is not the same cache.
**That is the appliance's first-boot condition**, so both numbers matter: a fresh
container pays the large one once per model, and every model change after pays the
small one.

### The profiling run is contaminated by compilation, and it sizes the KV cache

vLLM profiles peak activation in the same process that has just compiled the model, and
sizes the KV cache from what it measures. When substantial compilation lands in that
process, the profiler counts compilation workspace as model activation, and the engine
gets a smaller cache for the life of the run.

Measured on one configuration and card (Qwen3.8-27B EXL3 3.00bpw, turboquant KV):

| compile-cache state | graph compile | compile + warmup | peak activation | KV cache | `auto` context |
|---|---|---|---|---|---|
| disabled, torch caches also cold | 8.71 s | 50.27 s combined | **0.79 GiB** | 3.01 GiB | 172,032 |
| populating (collects AOT artifacts) | 13.90 s | 22.3 s + **34.04 s** warmup | **0.79 GiB** | 3.01 GiB | 172,032 |
| disabled, torch caches warm | 0.71 s | 9.40 s combined | 0.40 GiB | 3.41 GiB | 196,608 |
| hit | -- | 0.35 s + 1.69 s warmup | 0.40 GiB | 3.41 GiB | 196,608 |

The populating row shows the mechanism most clearly: AOT artifacts are collected during
the warmup run, so that run takes 34 s instead of 1.7 s and the transient it measures is
correspondingly larger.

**`disabled` is not a safe state**, which is the counterintuitive part and the reason
this is written as compilation volume rather than as a cache rule. The first and third
rows are both cache-disabled and differ only in whether *torch's* caches -- separate from
vLLM's, at `/tmp/torchinductor_ypell` here -- were warm. Same vLLM setting, 14% apart.

**So the first start of a configuration on a fresh box gets a worse cache than every
start after it**, silently, and `--max-model-len auto` resolves against the contaminated
figure. Pinning `--kv-cache-memory` from that start freezes it permanently, since pinning
also suppresses the profile run that would later have found the larger number.

`logscan` reports `compile_state` (`hit` / `populating` / `disabled` / `unknown`) plus
the compile and warmup timings, so a measurement can be labelled with the state it was
taken in. It identifies the case it can name and does not pretend to detect the rest:
nothing in vLLM's log says whether torch's own caches were warm.

The two caches are complementary rather than redundant, which is worth knowing before
reaching for either: torch's caches kernel codegen, vLLM's caches the traced and
AOT-compiled callable. A cache-disabled start with torch's caches warm compiles the graph
in 0.71 s but still spends 5.2 s in Dynamo; the cache-hit start has no Dynamo line at all.

This looks like a vLLM defect rather than an untwisted problem -- a memory profiler should
not be measuring the compiler -- and is worth reporting upstream. It is also, realistically,
**a defect upstream has little reason to prioritize**, which is the next section.

### Warm before measuring, and warm before serving

The practical lesson from the capacity result is not mainly about caches, it is that
**warmup should be applied wherever it can be**, and the measurement protocol should
assume it has not been.

- **Tier 2 does not trust a first start.** Discard it, measure the second, and require a
  third to agree before a result is stored as certified rather than provisional. That
  costs two extra process launches on a fresh box and nothing thereafter, which is cheap
  against the alternative of storing a number that is 14% wrong and cannot be told apart
  from a right one.
- **The appliance warms on provisioning, not on first request.** A fresh container has
  neither vLLM's compile cache nor torch's inductor cache, so the first start of a model
  costs 74-81 s where later ones cost 24 s. That belongs to whoever set the appliance up,
  not to whoever first asks it a question.

**And the two must match, or the measurement lies.** Capacity is fixed at engine start,
so a fit certified against a warm start does not hold for a cold one -- the cold engine
gets the smaller cache and the certified context is simply unavailable. Warming only for
measurement would produce exactly the failure the tiers exist to prevent: a stored number
that was true when taken and is not true when used. So warming is a property of how the
appliance starts engines, and the measurement protocol inherits it rather than the other
way round.

### Traps for the config generator

- **vLLM prints two `--kv-cache-memory=` suggestions, and they are not
  interchangeable.** One "to fit into requested memory", one "to fully utilize gpu
  memory". Measured on the 2026-09-07 capture (Qwen3.8-27B EXL3 3.00bpw, turboquant
  KV): **2.82 GiB requested, 3.01 GiB actually in use, 3.84 GiB to fully utilize** --
  the two bracket the running value. An earlier measurement on the same card had both
  below it (4.24 running, 4.03 "fully utilize", 3.95 "requested").

  The direction is *not* fixed, and it is not noise either: it is set by how
  over-reserved `peak_activation` is, which is a property of the **KV/attention backend**
  rather than of the card. This capture is turboquant, whose vLLM attention backend has
  significant prefill transients -- a full-context uncached prompt spikes VRAM that must
  be reserved and then mostly sits empty -- so this configuration cannot reach higher
  utilization at all. The fp8 configurations for the same model target 0.97 and may sit
  above the bracket instead. So the rule "the suggestion is lower than what is running"
  holds for the requested figure and not reliably for the other. What is
  stable is the mechanism: both subtract CUDA graph memory and a deliberate 150 MiB
  redundancy buffer that the profiler did not count, and the running config survives on
  `peak_activation` being over-reserved. Both numbers are self-consistent; neither is a
  bug. **Harvest both, apply neither blindly**, and never treat one as "the"
  suggestion -- which is what a config generator reading the first match would do.
- **Pinning `--kv-cache-memory` suppresses the profile run and all memory
  reporting.** So the shrink above is one-shot -- unless the manager re-measures by
  unpinning and repinning, which turns it into a ratchet.
- **`--max-model-len auto` makes "Maximum concurrency: 1.00x" a tautology**, not an
  observation: `max_model_len` is set *to* the KV capacity. Any `max_num_seqs > 1`
  on such a config is overcommitted by construction.

## Most of tier 0 already exists

`vllm-exl3-plugin/tools/` holds four tools written to one philosophy -- screen
cheaply before spending something expensive (rental hours, bandwidth, GPU time).
untwisted is the first consumer that ties them to a serving decision rather than a
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
  must agree with the runtime base**, and for untwisted we choose both.

[`vllm-fork/docker/Dockerfile`](../vllm-fork/docker/Dockerfile) is the reference for
the half that does not exist yet -- pinned `CUDA_VERSION`/`PYTHON_VERSION` (13.0.3,
3.12), wheels built against the same glibc floor as PyTorch's published wheels, and a
runtime lineage (`vllm-runtime-base` -> `vllm-base` -> `vllm-openai`) that keeps the
toolchain out of the serving image. untwisted's image belongs at **`vllm-base`, not
`vllm-openai`**: it replaces the entrypoint rather than wrapping it.

Read it as a reference, not as text to copy. The fork rebases on upstream; copied
stage text drifts silently, and this project has already paid three times over for
duplicating what upstream was maintaining.

It is also stale -- pinned to `VLLM_VERSION=v0.27.0` and applying
`vllm-exl3-plugin/patches/vllm-*.patch`, a directory that no longer exists now that
both dependencies are submodules and the vLLM fork sits on a branch named, aptly,
`appliance/v0.28.0`. That is the rental workflow's problem, not untwisted's: untwisted is
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
`--gpus all`. Whatever run scripts and docs untwisted ships have to lead with the podman
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
