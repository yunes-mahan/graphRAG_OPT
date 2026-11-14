"""Local Qwen backend for the VLDB-revision open-model experiments (Block C).  Every LLM call of C1/C2 goes through
`call` below; nothing is sent to an external API.  All outputs are cached by the calling scripts (separate cache files,
never mixed with the gpt-5.6 caches), so every number can be regenerated without running the model.

Settings (also written into every result file via SETTINGS):
  runtime   Ollama (local HTTP API), model tag qwen3:14b, digest bdbd181c33f2, 14.8B parameters, Q4_K_M quantization
  sampling  temperature 0, seed 0 (greedy decoding), thinking disabled (think=False)
  context   num_ctx 16384 tokens; output limit num_predict = the caller's max_tokens (<= 1500)
"""
import json
import subprocess
import urllib.request

MODEL = "qwen3:14b"
NUM_CTX = 16384
SETTINGS = dict(runtime="ollama", model=MODEL, digest="bdbd181c33f2", parameters="14.8B", quantization="Q4_K_M",
                temperature=0, seed=0, think=False, num_ctx=NUM_CTX)


def ollama_version():
    try:
        return subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    except Exception:  # noqa: BLE001
        return "unknown"


def call(model_ignored, messages, json_mode=False, max_tokens=1200):
    body = dict(model=MODEL, messages=messages, stream=False, think=False,
                options=dict(temperature=0, seed=0, num_ctx=NUM_CTX, num_predict=min(int(max_tokens), 1500)))
    if json_mode:
        body["format"] = "json"
    req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                out = json.loads(r.read())
            return out["message"]["content"], dict(model=MODEL, inp=out.get("prompt_eval_count", 0),
                                                   out=out.get("eval_count", 0))
        except Exception as e:  # noqa: BLE001
            err = e
    raise RuntimeError(f"ollama call failed: {err}")
