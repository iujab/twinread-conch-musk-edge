#!/usr/bin/env python
"""Start, use and stop a host x86 llama-server built at the pinned commit, with the board's server flags."""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BIN = REPO / "data/llm/finetune/build-host-server/bin/llama-server"
BASE_GGUF = REPO / "data/llm/gen/Qwen3-0.6B-Q4_0-embQ8.gguf"


class Server:
    def __init__(self, model, port=1791, threads=4, ctx=1024, log=None):
        self.model, self.port, self.threads, self.ctx = str(model), port, threads, ctx
        self.log_path = Path(log) if log else None
        self.p = None

    def __enter__(self):
        logf = open(self.log_path, "w") if self.log_path else subprocess.DEVNULL
        self.p = subprocess.Popen(
            [str(BIN), "-m", self.model, "-c", str(self.ctx), "-t", str(self.threads), "-tb", str(self.threads), "-np", "1",
             "--host", "127.0.0.1", "--port", str(self.port), "--no-webui", "--cache-ram", "0", "--ctx-checkpoints", "0",
             "--reasoning", "off"], stdout=logf, stderr=subprocess.STDOUT, env={"OMP_WAIT_POLICY": "passive", "PATH": "/usr/bin:/bin"})
        t0 = time.time()
        while time.time() - t0 < 120:
            try:
                if json.loads(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2).read()).get("status") == "ok":
                    return self
            except Exception:
                pass
            if self.p.poll() is not None:
                raise RuntimeError(f"server exited with {self.p.returncode}")
            time.sleep(0.5)
        raise RuntimeError("server did not come up")

    def __exit__(self, *_):
        self.p.terminate()
        try:
            self.p.wait(10)
        except subprocess.TimeoutExpired:
            self.p.kill()

    def post(self, path, body, timeout=600):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def chat(self, body):
        b = dict(body)
        b["stream"] = False
        return self.post("/v1/chat/completions", b)


if __name__ == "__main__":
    sys.exit("import this module")
