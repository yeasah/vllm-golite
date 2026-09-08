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

# Shaped after the lines the scanner looks for. Not a claim about vLLM's real output --
# the calibration path for that is `python -m golite.engine.logscan` on a real capture.
BANNER = [
    "INFO vllm: starting engine",
    "INFO gpu memory: 15.28 GiB is free of 15.51 GiB total",
    "INFO GPU KV cache size: 67,584 tokens",
    "INFO Maximum concurrency for 262,144 tokens per request: 1.03x",
    "INFO To fully utilize gpu memory pass --kv-cache-memory=1323302912",
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
