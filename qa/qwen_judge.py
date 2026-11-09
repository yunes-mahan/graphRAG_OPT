"""Second judge from another model family: Qwen3-14B, run locally via Ollama (no API cost).

Re-runs the three QA evaluations with every judge call routed to qwen3:14b (temperature 0, thinking off, JSON
output). Summaries, questions and answers come from the existing caches; any non-judge call that would reach the
OpenAI API raises instead, so the second judge sees exactly the answers the first judge saw.
Judge caches: qa/cache_<name>_qwen.json.  Outputs: qa/results_qwen.json, qa_global_qwen.json,
entity_graph/qa_entity_qwen.json.
usage: python qwen_judge.py entity|global|local
"""

import json
import pathlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wikidata"))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
HERE = config.RESULTS
import qa_run as qr  # noqa: E402

LOCAL = "qwen3:14b"
_orig_cache_init = qr.Cache.__init__


def _cache_init(self, name):
    _orig_cache_init(self, name + "_qwen" if name.startswith("judge") else name)


def _call(model, messages, json_mode=False, max_tokens=1200):
    if model != qr.JUDGE_MODEL or not json_mode:
        raise RuntimeError(f"cache miss for a non-judge call ({model}); refusing to call the API")
    body = dict(model=LOCAL, messages=messages, stream=False, think=False, format="json",
                options=dict(temperature=0, num_ctx=16384, num_predict=400))
    req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.loads(r.read())
    return out["message"]["content"], dict(model=LOCAL, inp=out.get("prompt_eval_count", 0), out=out.get("eval_count", 0))


_orig_write = pathlib.Path.write_text
REDIRECT = {"results.json": "results_qwen.json", "qa_global.json": "qa_global_qwen.json",
            "qa_entity.json": "qa_entity_qwen.json"}


def _write(self, data, *a, **k):
    if self.name in REDIRECT:
        self = self.with_name(REDIRECT[self.name])
    return _orig_write(self, data, *a, **k)


CRIT = ["comprehensiveness", "diversity", "empowerment", "directness"]
_abs_cache = None


def score_answer(q, a):
    """Absolute 1-10 rating of one answer per criterion (no pairwise presentation, hence no position bias)."""
    import hashlib
    global _abs_cache
    if _abs_cache is None:
        _abs_cache = qr.Cache("abs_qwen")
    k = hashlib.sha1(("abs|" + q + "|" + a).encode()).hexdigest()
    hit = _abs_cache.get(k)
    if hit:
        return hit["s"]
    prompt = (f"Question: {q}\n\nAnswer:\n{a}\n\nRate the answer on four criteria, each from 1 (very poor) to 10 "
              "(excellent): comprehensiveness (covers all aspects of the question), diversity (varied perspectives and "
              "insights), empowerment (helps the reader understand and make informed judgements), directness "
              "(addresses the question specifically and clearly). Be strict and use the full scale. Return JSON "
              '{"comprehensiveness": int, "diversity": int, "empowerment": int, "directness": int}.')
    for _ in range(3):
        text, usage = _call(qr.JUDGE_MODEL, [{"role": "user", "content": prompt}], json_mode=True)
        try:
            s = {c: int(json.loads(text)[c]) for c in CRIT}
            break
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            s = None
    if s is None:
        raise RuntimeError("absolute judge failed")
    _abs_cache.put(k, dict(s=s, usage=usage))
    return s


def compare_abs(q, a, b, cache):
    sa, sb = score_answer(q, a), score_answer(q, b)
    return dict(v={c: 1 if sa[c] > sb[c] else 2 if sb[c] > sa[c] else 0 for c in CRIT})


qr.Cache.__init__ = _cache_init
qr.call = _call
pathlib.Path.write_text = _write

if __name__ == "__main__":
    which = sys.argv[1]
    import qa_global
    qa_global.compare = compare_abs
    if which == "entity":
        import qa_entity
        qa_entity.main("run")
    elif which == "global":
        qa_global.main()
    else:
        qr.main()
