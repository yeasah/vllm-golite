# Open tasks

What is outstanding, our current understanding of it, and the approach we think is
the best candidate today. Nothing else.

<!--
POLICY -- read before adding to this file.

The test: **if it would still be true and worth reading after the item is closed,
it does not belong here.** Measurements, ruled-out hypotheses, post-mortems and
chronology all fail that test. They go to the matching note in docs/ when written --
not here first for migration later, because that migration never happens.

Each item carries four things and stops:
  1. the outcome wanted;
  2. one or two sentences on what it unblocks;
  3. the current best-candidate approach, and the one-line reason it is the
     candidate;
  4. a pointer to the note holding the evidence.

Headings carry a **stable slug**, and cross-references from code and docs use the
slug, never the position. Prefer pointing at the docs/ note wherever the sentence
works either way -- a reader almost always needs the subject, not the queue
position.
-->

## `engine-supervisor` -- Start, stop and observe one engine

One engine, launched as a supervised child process, with a lifecycle that reports
honestly: starting, healthy, failed-and-why. Everything else in the project is
downstream of this, including every fit tier above 1, which pays an engine start
per candidate.

**Candidate approach:** vllm-tuner's trial lifecycle, which is the same state
machine -- start, health check with timeout, run, **parse logs for OOM**, clean up.
It is the candidate because that log-parsing step is not incidental: vLLM's
interesting failures are reported rather than raised, and a supervisor that only
watches the exit code learns nothing from them.

Put the process boundary behind a thin `EngineRuntime` interface with one
implementation. An engine record carries an id even though there is only ever one.
**One engine start per process, always** -- vLLM's in-process leak makes process
reuse unavailable, and this is the layer that has to enforce it.

**Scope may be larger than it looks:** if that leak turns out to have a per-request
component and not only a per-construction one, an unattended appliance needs uptime
and RSS monitoring with a restart policy here, in v1 rather than later. Establishing
which it is comes before deciding. See [docs/design.md](docs/design.md).

## `config-store` -- Named configurations, and the place derived answers land

Name and store a configuration -- a `vllm serve` invocation plus its environment --
so one model can be run in several shapes and a test matrix can be selected by name.
Unblocks `engine-supervisor` (which needs something to launch) and every fit tier
(which needs somewhere to put an answer that cost an engine start to produce).

**Wanted now, independent of the guidance work.** The mechanism today is
`~/ckpt/run-*.sh`: one live invocation per file and three or four commented-out
alternates whose intent survives only as a comment. It is unpleasant to use and rots
silently, because a commented block records no evidence that it ever worked.

**Candidate approach:** present the literal command line, store it structured -- args
as a list, environment as a map with explicit unsets. Presentation is the candidate
because paste-ability into a shell is what makes this a real replacement for the
scripts; structured storage is the candidate because the fit layer has to rewrite
individual flags, and the existing scripts already contain multi-valued flags,
JSON-valued flags with embedded quotes, and two different syntaxes for passing a
value -- all of which defeat text surgery on a stored line.

Two properties worth having from the first version, both cheap and both expensive to
retrofit: **every entry carries provenance** (hand-written or derived by which tier,
when, against which box fingerprint and fork/plugin versions) so staleness is
mechanical rather than remembered; and **an entry that has never launched is a
draft**, which is most of the cure for the rot.

Then `config-lint`: the documented traps are statically checkable against a stored
entry with no GPU and no engine start (`--max-model-len auto` with `--max-num-seqs >
1`, `--gpu-memory-utilization` above the box's free/total ratio, a `--kv-cache-memory`
pin inherited from vLLM's own low-biased suggestion). It is the knowledge layer in its
cheapest possible form and a good test of whether the store holds enough structure.
See [docs/design.md](docs/design.md).

## `manager-api` -- One contract, and no way around it

The interface the frontend and the CLI both speak. Unblocks both of them, and makes
golite scriptable into existing workflows rather than a place work has to be done by
hand.

**Candidate approach:** plain HTTP for state changes and queries, one multiplexed SSE
stream with typed events for everything pushed. It is the candidate because nothing in
the management surface is bidirectional -- commands are POSTs, events are
server-to-client -- and SSE keeps the CLI a `curl` away while reusing the SSE
discipline the router already needs. WebSockets stays the escape hatch if something
genuinely bidirectional appears, an interactive engine console being the plausible case.

**The rule that has to hold from the first endpoint:** the manager has no internal path
that bypasses its own API. Retrofitting it means unpicking every shortcut the UI took.

Do not design the schema up front. Pick the transports, which are expensive to change,
and let endpoints accrete against the CLI -- which is the first consumer, since the UI
will not exist for a while. Watch for polling: it means something belongs on the event
stream. See [docs/design.md](docs/design.md).

## `frontend-foundation` -- The UI shell, before it is needed

Build pipeline, the bundle served by the manager process itself, event-stream client
plumbing, and a dev loop. Unblocks every feature not worth expressing as a command --
which is most of what comes after the fit tiers start producing things to look at.

**Candidate approach:** React (a weak preference, not a finding), bundled and served by
the same uvicorn process -- one port, one process, no separate node server in the
container -- with HMR against a running manager for development. Single-process serving
is the candidate because it is what the container wants; the dev loop is called out
because it is the difference between UI work being pleasant and being miserable.

Slot it early. A CLI can carry the load for a while, but the point at which it stops
being worth expressing that way arrives sooner than the frontend can be stood up from
nothing. See [docs/design.md](docs/design.md).

## `auto-context` -- make concurrency an input to sizing, not an afterthought

`--max-model-len auto:N` -- the largest context that leaves room for `N` concurrent
requests. Unblocks the trap `config-lint` can only detect: `auto` alone sets
`max_model_len` *to* the KV capacity, so "Maximum concurrency: 1.00x" is a tautology
and any `--max-num-seqs > 1` on such a config is overcommitted by construction. Lint
flags it; this removes it.

**Candidate approach:** parse the suffix and call vLLM's own
`estimate_max_model_len(vllm_config, kv_cache_spec, available_memory)` in
`vllm/v1/core/kv_cache_utils.py` with `available_memory // N`. It already binary
searches for exactly this quantity and restores the config it borrowed; today it is
called only to make a "doesn't fit" error friendlier. That is the candidate because the
arithmetic exists and is upstream's own, and because the call needs a profiled engine --
so it belongs inside engine init rather than reimplemented in the manager.

**It belongs upstream**, and is worth more there than here. Carry it in the fork
meanwhile, as a patch shaped for submission rather than a golite feature -- this project
has paid repeatedly for maintaining what upstream would have taken.

Migrated from `vllm-virtualkv-plugin`'s TODO, where it was recorded so as not to be lost
but was explicitly not that plugin's business. Removing it there is still outstanding.
See [docs/design.md](docs/design.md).

## `router` -- One endpoint that survives engine restarts

A stable OpenAI-compatible address, so client configs are not rewritten on every
model change, and so there is somewhere to answer a coherent 503 when nothing is
loaded.

**Candidate approach:** a reverse proxy with a single upstream slot and no routing
logic -- a table that happens to hold one row. It is the candidate because it is the
seam multi-engine lands on later, and it costs almost nothing to have on day one.

Two failure modes to test for deliberately, since both pass a smoke test: buffered
SSE, and swallowed client disconnects (vLLM aborts generation on disconnect; a proxy
that drops it leaves requests generating for nobody). See
[docs/design.md](docs/design.md).

## `fit-shortlist` -- Answer "will it fit" without spending a launch

Given a model and this box, return a ranked shortlist of candidate configurations
worth actually starting. Unblocks every downstream tier by keeping the number of
engine starts small, which is what makes fitting affordable at all.

**Candidate approach:** tiers 0 and 1 together -- header reads first
(`tp_preflight --remote`, `checkpoint_survey`), then `config.json` arithmetic.
Header-first is the candidate because it is the only tier that can answer before a
download, and because TP divisibility is not derivable from `config.json` at all.

Treat tier 1's arithmetic as a *ranking* signal, never a verdict. The facts that
decide fit -- total-vs-free, profiler failures at strictly cheaper configs, eager
sometimes costing more than graphs -- are invisible to it by construction. See
[docs/design.md](docs/design.md).

## `fit-certify` -- Prove a config survives a full-context session

A config that starts is not a config that serves. Certification means a run at
realistic context depth and concurrency that does not OOM mid-session, producing a
statement the appliance can stand behind ("N concurrent at C tokens").

**Candidate approach:** guidellm as the load generator, walking depth and
concurrency until the cliff appears; `memprof` invoked on failure to attribute it.
guidellm is the candidate because it already produces the distributions and
explicitly declines to interpret them, which is precisely the half worth owning.

This is the tier nothing else in the ecosystem produces, and the reason the static
tier cannot be the product. See [docs/design.md](docs/design.md).

## `transient-characterizer` -- Predict the profiler-invisible allocations

Turn the existing pair -- `vllm-exl3-plugin/tools/memprof.py` (reduce) and
`~/ckpt/profile-completion.py` (gather) -- into something that predicts rather than
reports. Unblocks `fit-certify`: it is what converts "OOMs at 60K context" into
"this site costs ~6 KB/token," which is a config change rather than a failure report.

**Candidate approach:** sweep `target_toks` across several depths, run `memprof` on
each snapshot, and regress bytes-per-token per call site. The slope is the candidate
because peak-at-one-depth is exactly the measurement the vLLM profiler already
takes, and it is the one that misses this class.

Keep what the gathering script gets right (alloc conf before torch import, record
window bounded to one call, prefix-cache-busting prefix). The depth sweep stays
in-process -- one engine start, several calls -- because **the config axis must not
migrate inside the process**: vLLM's in-process leak means one engine start per
process, everywhere.

**Open question first:** profiling requires `VLLM_ENABLE_V1_MULTIPROCESSING=0`, and
the leak means that mode can never be the serving mode -- so whether these numbers
transfer to the MP path is assumed, cannot be resolved by serving in-process, and
should be checked before any of them are trusted for sizing. See
[docs/design.md](docs/design.md).

## `preflight-share` -- One home for the screening tools

`checkpoint_survey.py`, `tp_preflight.py`, `host_survey.py` and `memprof.py` are
needed by both repos. Unblocks tier 0 without forking them.

**Candidate approach:** a small extracted package both repos depend on, rather than
moving them out of the plugin. It is the candidate because the plugin genuinely uses
them for benchmark provenance, and its boundary note says appliance-specific
reasoning must not leak upward -- a move would make the plugin's own docstrings
dishonest about why the tools exist.

## `start-latency` -- Measure what a model change costs

With one engine, every model change pays a full engine start in the foreground.
Unblocks the UX question of whether "change models" is a setting or an outage, and
scopes whether load-time work (weight load, capture, compile cache) is worth
optimizing.

**Candidate approach:** instrument the supervisor and break the wall time down by
phase, on a checkpoint of each shape golite claims to serve. Phase breakdown is the
candidate because the aggregate number does not say which lever to pull, and the
levers differ (page cache, capture sizes, compile cache).

**Largely answered for one configuration** (Qwen3.8-27B EXL3 3.00bpw + turboquant, one
16 GiB card, 2026-09-07): 81.2 s cold, **24.1 s warm**, and compilation is essentially
all of the difference. vLLM prints its own phase breakdown, so `logscan` harvests it on
every start rather than needing a special run. What remains is coverage -- other model
shapes and quantizations, where weight load may dominate instead -- and the action the
measurement implies, which is that the shipped image should enable the compile cache.

**The largest obvious lever is unavailable**: process reuse across engine starts is
ruled out by vLLM's in-process leak, so interpreter start, imports and CUDA context
creation are paid every time and cannot be amortized. The warm figure above is what that
floor actually costs.

The warm/cold gap also moves `peak_activation` and therefore the KV cache size, which is
a `fit-shortlist` and `fit-certify` concern before it is a latency one. See
[docs/design.md](docs/design.md).

## `format-routing` -- Know which backend serves which checkpoint

Model selection cannot be "any HuggingFace model": golite ships specific plugins,
and the mapping from a checkpoint's quantization format to the backend that serves
it is knowledge no upstream project has.

**Candidate approach:** derive from the checkpoint rather than the model name, via
the survey tools that already read headers. Names are not a reliable statement of
format, and getting this wrong is the silent-corruption case rather than a startup
error. See [docs/design.md](docs/design.md).

## `shippable-image` -- A container that runs on a box with nothing on it

One image carrying the vLLM fork, the exllamav3 fork, the shipped plugins, the
manager and the frontend, on agreed CUDA and torch. Unblocks everything else being
worth building: a system this hard to assemble by hand has no users without it.

**Candidate approach:** keep `~/podman/vast-vllm`'s wheel/image split -- build wheels
in a builder, install them in a runtime image derived from the fork's `vllm-base`
stage. It is the candidate because half of it already works, and because compiling
in the shipped image would drag the CUDA toolchain along with it.

**Not blocked on vast-vllm**, and do not try to generalize it: that pile is
vast-specific down to its base image, and its own staleness belongs to the rental
workflow rather than here. It is a reference for the wheel half and nothing more.
golite builds its own wheels against its own runtime base -- the invariant that
transfers is only that the two agree.

Podman and CDI are the tested path; there is no docker on the development box. The
image must be able to report both fork commits and every plugin version, since the
fit cache is keyed on them. See [docs/design.md](docs/design.md).
