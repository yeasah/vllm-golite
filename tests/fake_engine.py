#!/usr/bin/env python3
"""A stand-in for `vllm serve`, so the whole lifecycle is testable without a GPU.

It exists to exercise the paths that matter and are otherwise only reachable by
breaking a real engine: exiting before health, hanging past the deadline, dying while
serving, and leaving a child process behind.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

# Real lines, lifted from tests/data/vllm-start-qwen3.8-27b.log so the fake exercises
# the patterns the way an engine actually writes them -- including the (EngineCore pid=)
# prefix, which anything anchored to line start would miss.
BANNER = [
    "(EngineCore pid=1) INFO [gpu_model_runner.py:5515] Model loading took 10.24 GiB "
    "memory and 2.623740 seconds",
    "(EngineCore pid=1) INFO [interface.py:986] Setting attention block size to 3072 "
    "tokens to ensure that attention page size is >= mamba page size.",
    "(EngineCore pid=1) INFO [gpu_worker.py:578] Available KV cache memory: 3.01 GiB",
    "(EngineCore pid=1) INFO [kv_cache_utils.py:1883] GPU KV cache size: 172,032 tokens, "
    "Maximum concurrency for 172,032 tokens per request: 1.00x",
    "(EngineCore pid=1) INFO [gpu_worker.py:804] Free memory on device (15.28/15.51 GiB) "
    "on startup. Desired GPU memory utilization is (0.92, 14.27 GiB). Actual usage is "
    "10.47 GiB for consumed memory (weights + non-torch), 0.79 GiB for peak activation, "
    "and 0.04 GiB for CUDAGraph memory. Replace gpu_memory_utilization config with "
    "`--kv-cache-memory=3031561913` (2.82 GiB) to fit into requested memory, or "
    "`--kv-cache-memory=4121221120` (3.84 GiB) to fully utilize gpu memory. Current kv "
    "cache memory in use is 3.01 GiB.",
]


class Health(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        code = 200 if self.path == "/health" else 404
        self.send_response(code)
        self.end_headers()

    def log_message(self, *a):  # keep the fake's own logging out of the stream
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--ready-delay", type=float, default=0.0)
    ap.add_argument("--exit-before-ready", type=int, default=None)
    ap.add_argument("--emit-oom", action="store_true")
    ap.add_argument("--hang", action="store_true", help="never serve health")
    ap.add_argument("--die-after-ready", type=float, default=None)
    ap.add_argument("--spawn-child", action="store_true",
                    help="leave a grandchild behind, as vLLM's EngineCore does")
    args, _unknown = ap.parse_known_args()

    for line in BANNER:
        print(line, flush=True)

    if args.spawn_child:
        # Deliberately not tracked by the supervisor: the point is that killing only
        # the parent would strand it, still holding its resources.
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
        print("INFO spawned EngineCore child", flush=True)

    if args.emit_oom:
        print("ERROR torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to "
              "allocate 2.00 GiB", flush=True)
    if args.exit_before_ready is not None:
        return args.exit_before_ready

    if args.hang:
        while True:
            time.sleep(1)

    time.sleep(args.ready_delay)
    server = HTTPServer(("127.0.0.1", args.port), Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"INFO serving on {args.port}", flush=True)

    if args.die_after_ready is not None:
        time.sleep(args.die_after_ready)
        os._exit(9)

    while True:
        time.sleep(1)


if __name__ == "__main__":
    sys.exit(main())
