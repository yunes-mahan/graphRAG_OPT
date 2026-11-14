"""Local Qwen backend for the V12 open-model experiments (C1-C4).  Nothing is sent to an external API.
Model: Ollama, qwen3:14b (digest bdbd181c33f2, 14.8B, Q4_K_M), thinking off.
Defaults: temperature 0, seed 0, num_ctx = $QWEN_NUM_CTX (default 32768; the V11 runs used 16384 - global
map-reduce answers need more context).  `sampled(temperature, seed)` overrides the sampling for the A/A test only
(independent samples of the same input).  Every result file records SETTINGS.
"""
import contextlib
import json
import os
import threading
import urllib.request

MODEL = "qwen3:14b"
NUM_CTX = int(os.environ.get("QWEN_NUM_CTX", "32768"))
SETTINGS = dict(runtime="ollama 0.40.15", model=MODEL, digest="bdbd181c33f2", parameters="14.8B", quantization="Q4_K_M",
                temperature=0, seed=0, think=False, num_ctx=NUM_CTX, parallel=1)
_local = threading.local()


@contextlib.contextmanager
def sampled(temperature, seed):
    _local.opt = dict(temperature=temperature, seed=seed)
    try:
        yield
    finally:
        _local.opt = None


def call(model_ignored, messages, json_mode=False, max_tokens=1200):
    opt = getattr(_local, "opt", None) or dict(temperature=0, seed=0)
    body = dict(model=MODEL, messages=messages, stream=False, think=False,
                options=dict(num_ctx=NUM_CTX, num_predict=min(int(max_tokens), 2000), **opt))
    if json_mode:
        body["format"] = "json"
    req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    err = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                out = json.loads(r.read())
            return out["message"]["content"], dict(model=MODEL, inp=out.get("prompt_eval_count", 0),
                                                   out=out.get("eval_count", 0), **opt)
        except Exception as e:  # noqa: BLE001
            err = e
    raise RuntimeError(f"ollama call failed: {err}")
