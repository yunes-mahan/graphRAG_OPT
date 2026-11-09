"""Second ANSWERING model: Qwen3-14B (local, Ollama, thinking off) reads the same reports with the same prompt and
retrieval as gpt-5.6-luna; the judge stays gpt-5.6-terra, so only the reader changes.
Stores: FRESH, recompute/warm-start/hybrid at tau=0.9, hybrid and recompute never refreshed; 300 local questions.
  python qwen_reader.py answer   -> local answers only (no API calls), cache qa/cache_answers_qwenreader.json
  python qwen_reader.py judge    -> gpt-5.6-terra judging (API), writes qa/results_qwenreader.json
"""

import concurrent.futures as cf
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wikidata"))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
HERE = config.RESULTS
import qa_run as qr  # noqa: E402

QA = HERE / "qa"
STORES = ["FRESH|recompute all", "recompute_leiden|tau=0.9", "warmstart_leiden|tau=0.9", "hybrid|tau=0.9",
          "hybrid|tau=0.0", "recompute_leiden|tau=0.0"]
_api_call = qr.call


def ollama(model, messages, json_mode=False, max_tokens=1200):
    body = dict(model="qwen3:14b", messages=messages, stream=False, think=False,
                options=dict(temperature=0, num_ctx=16384, num_predict=max_tokens))
    req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.loads(r.read())
    return out["message"]["content"], dict(model="qwen3:14b", inp=out.get("prompt_eval_count", 0),
                                           out=out.get("eval_count", 0))


def main(mode):
    plan = json.loads((QA / "plan.json").read_text())
    node_t = np.load(QA / "node_t.npy")
    present = np.where(node_t >= 0)[0]
    stores = {}
    for m in ["recompute_leiden", "warmstart_leiden", "hybrid"]:
        for tau, v in plan["methods"][m]["taus"].items():
            stores[f"{m}|tau={tau}"] = v["final_store"]
    comm = {}
    for v, c in zip(present, plan["methods"]["recompute_leiden"]["labels"][-1]):
        comm.setdefault(int(c), []).append(int(v))
    stores["FRESH|recompute all"] = [sorted(x) for x in comm.values() if len(x) >= 5]
    cs = qr.Cache("summaries")
    items = qr.Cache("questions").get("questions")["items"] + qr.Cache("questions_more").get("questions")["items"]
    ca = qr.Cache("answers_qwenreader")

    def ans(a):
        n, it = a
        qr.call = ollama
        return n, it, qr.answer(it["question"], [cs.get(qr.key_of(s))["text"] for s in stores[n]], ca, "qwenreader|" + n)

    jobs = [(n, it) for n in STORES for it in items]
    if mode == "answer":
        with cf.ThreadPoolExecutor(2) as ex:
            for k, _ in enumerate(ex.map(ans, jobs)):
                if k % 100 == 0:
                    print(f"{k}/{len(jobs)}", flush=True)
        return
    qr.call = _api_call
    cj = qr.Cache("judge_qwenreader")
    rows = []

    def work(a):
        n, it = a
        hit = ca.get(__import__("hashlib").sha1(("qwenreader|" + n + "|" + it["question"]).encode()).hexdigest())
        j = qr.judge(it, hit["text"], cj)
        return dict(store=n, q=it["question"], recall=float(np.mean(j["covered"])))

    with cf.ThreadPoolExecutor(12) as ex:
        rows = list(ex.map(work, jobs))
    print(pd.DataFrame(rows).groupby("store").recall.agg(["mean", "count"]).round(3))
    (QA / "results_qwenreader.json").write_text(json.dumps(dict(per_question=rows), indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
