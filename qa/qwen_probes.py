"""Staleness probe with a fully local pipeline: Qwen3-14B (Ollama, thinking off, temperature 0) writes the questions,
answers them and judges key-point coverage; the reports themselves are unchanged (gpt-5.6-luna, cached).
Run once on the never-refreshed stores (calibration against the gpt pipeline, probes.json) and once on the reports
that the Jaccard rule keeps at tau = 0.9 (reports that gained >= 4 papers with Jaccard >= 0.9).
usage: python qwen_probes.py 0.0 | 0.9     -> probes_qwen_never.json / probes_qwen_tau09.json
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wikidata"))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
HERE = config.RESULTS
tau = sys.argv[1]
os.environ["PROBE_TAU"] = tau
os.environ["PROBE_ONLY"] = "staleness"
os.environ["PROBE_SUFFIX"] = "_qwen_never" if tau == "0.0" else "_qwen_tau09"
os.environ.setdefault("OPENAI_API_KEY", "unused")
import qa_run as qr  # noqa: E402

_orig_init = qr.Cache.__init__


def _init(self, name):
    _orig_init(self, name + "_qwen" if name.endswith("_probe") else name)


def _call(model, messages, json_mode=False, max_tokens=1200):
    body = dict(model="qwen3:14b", messages=messages, stream=False, think=False,
                options=dict(temperature=0, num_ctx=16384, num_predict=min(max_tokens, 1200)))
    if json_mode:
        body["format"] = "json"
    req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.loads(r.read())
    return out["message"]["content"], dict(model="qwen3:14b", inp=out.get("prompt_eval_count", 0),
                                           out=out.get("eval_count", 0))


qr.Cache.__init__ = _init
qr.call = _call
import probes  # noqa: E402

probes.qr.call = _call
probes.main("run")
