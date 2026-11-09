"""QA experiment, stage 2 (LLM calls): summaries, questions, answers, key-point judging.

Models (OpenAI): SUMMARY/ANSWER = gpt-5.6-luna, QUESTION/JUDGE = gpt-5.6-terra.
Stores evaluated at the final version: every maintainer x staleness tolerance (from plan.json) plus
FRESH = recomputed Leiden partition with all summaries regenerated at the final version (reference).
Question generation is independent of all stores: a seed paper from 2019-2020 and its 8 nearest papers
(TF-IDF over title+abstract) -> one topical question + 4 key points grounded in those papers.
Answers use only the top-6 summaries of a store (TF-IDF retrieval). Judge marks covered key points.
All LLM outputs and token usage are cached in qa/cache_*.json (resumable).
"""

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
_sys.path.insert(0, str(_P(__file__).resolve().parents[1] / 'benchmark'))
import config  # noqa: E402


import concurrent.futures as cf
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

sys.path.insert(0, str(Path(__file__).parent))
import qa_plan as qp  # noqa: E402

HERE = config.RESULTS
QA = HERE / "qa"
SUM_MODEL, ANS_MODEL, Q_MODEL, JUDGE_MODEL = "gpt-5.6-luna", "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-terra"
N_QUESTIONS, TOP_K, MAX_CHARS = 120, 6, 24000
client = OpenAI()
lock = threading.Lock()


class Cache:
    def __init__(self, name):
        self.path = QA / f"cache_{name}.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def get(self, k):
        return self.data.get(k)

    def put(self, k, v):
        with lock:
            self.data[k] = v
            self.path.write_text(json.dumps(self.data))


def call(model, messages, json_mode=False, max_tokens=1200):
    for attempt in range(6):
        try:
            kw = dict(model=model, messages=messages, max_completion_tokens=max_tokens)
            if json_mode:
                kw["response_format"] = {"type": "json_object"}
            r = client.chat.completions.create(**kw)
            return r.choices[0].message.content, dict(model=model, inp=r.usage.prompt_tokens, out=r.usage.completion_tokens)
        except Exception as e:  # noqa: BLE001
            time.sleep(min(60, 5 * 2 ** attempt))
            err = e
    raise RuntimeError(f"LLM call failed: {err}")


def key_of(members):
    return hashlib.sha1(",".join(map(str, sorted(members))).encode()).hexdigest()


def summarise(members, docs, cache):
    k = key_of(members)
    hit = cache.get(k)
    if hit:
        return hit
    parts, used = [], 0
    for v in sorted(members):
        t, a = docs[v]
        s = f"- {t}: {a[:600]}"
        if used + len(s) > MAX_CHARS:
            break
        parts.append(s)
        used += len(s)
    prompt = ("You write community reports for a GraphRAG index over computer-science papers. Below are the "
              f"papers of one community ({len(members)} papers, {len(parts)} shown). Write a report of at most "
              "200 words: a short title, the main research themes, typical methods, and recent trends. "
              "Use only the given papers.\n\n" + "\n".join(parts))
    text, usage = call(SUM_MODEL, [{"role": "user", "content": prompt}], max_tokens=1500)
    out = dict(text=text, usage=usage, n=len(members))
    cache.put(k, out)
    return out


def make_questions(docs, years, cache):
    if cache.get("questions"):
        return cache.get("questions")["items"]
    idx = np.arange(len(docs))
    corpus = [docs[i][0] + " " + docs[i][1] for i in idx]
    X = TfidfVectorizer(stop_words="english", max_features=50000, sublinear_tf=True).fit_transform(corpus)
    rng = np.random.default_rng(7)
    recent = idx[years >= 2019]
    seeds = rng.choice(recent, N_QUESTIONS * 2, replace=False)
    items, usage = [], []
    for s in seeds:
        if len(items) >= N_QUESTIONS:
            break
        sims = linear_kernel(X[s], X).ravel()
        nb = np.argsort(-sims)[:8]
        ctx = "\n".join(f"[{j}] {docs[j][0]}: {docs[j][1][:700]}" for j in nb)
        prompt = ("From the papers below, write ONE broad, topic-level question that a user of a research "
                  "knowledge base might ask (e.g. 'What are the main approaches and trends in ...?'), and exactly "
                  "4 short key points that a good answer should mention. Each key point must be a general theme, "
                  "method family or trend that is supported by at least two of the papers; do NOT use numbers, "
                  "dataset names, system names or details of a single paper. Return JSON "
                  "{\"question\": str, \"key_points\": [str, str, str, str]}.\n\n" + ctx)
        text, u = call(Q_MODEL, [{"role": "user", "content": prompt}], json_mode=True)
        usage.append(u)
        try:
            j = json.loads(text)
            if isinstance(j.get("question"), str) and len(j.get("key_points", [])) == 4:
                items.append(dict(seed=int(s), papers=[int(x) for x in nb], **j))
        except json.JSONDecodeError:
            pass
    cache.put("questions", dict(items=items, usage=usage))
    return items


def answer(question, summaries, cache, store_name):
    k = hashlib.sha1((store_name + "|" + question).encode()).hexdigest()
    hit = cache.get(k)
    if hit:
        return hit
    if not summaries:
        out = dict(text="No information available.", usage=dict(model=ANS_MODEL, inp=0, out=0), used=[])
        cache.put(k, out)
        return out
    vec = TfidfVectorizer(stop_words="english").fit(summaries + [question])
    sims = linear_kernel(vec.transform([question]), vec.transform(summaries)).ravel()
    top = list(np.argsort(-sims)[:TOP_K])
    ctx = "\n\n".join(f"Report {i + 1}:\n{summaries[j]}" for i, j in enumerate(top))
    prompt = ("Answer the question using ONLY the community reports below. Be specific and concise "
              "(at most 150 words). If the reports do not contain the information, say so.\n\n"
              f"Question: {question}\n\n{ctx}")
    text, usage = call(ANS_MODEL, [{"role": "user", "content": prompt}], max_tokens=1200)
    out = dict(text=text, usage=usage, used=[int(t) for t in top])
    cache.put(k, out)
    return out


def judge(item, ans_text, cache):
    k = hashlib.sha1((item["question"] + "|" + ans_text).encode()).hexdigest()
    hit = cache.get(k)
    if hit:
        return hit
    kp = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(item["key_points"]))
    prompt = ("You grade an answer against 4 key points. A key point counts as covered only if the answer "
              "states it (paraphrase is fine). Return JSON {\"covered\": [true|false x4]}.\n\n"
              f"Question: {item['question']}\nKey points:\n{kp}\n\nAnswer:\n{ans_text}")
    text, usage = call(JUDGE_MODEL, [{"role": "user", "content": prompt}], json_mode=True, max_tokens=800)
    try:
        cov = [bool(x) for x in json.loads(text)["covered"]][:4]
    except Exception:  # noqa: BLE001
        cov = [False] * 4
    out = dict(covered=cov, usage=usage)
    cache.put(k, out)
    return out


def main(limit_q=None, limit_stores=None):
    plan = json.loads((QA / "plan.json").read_text())
    nodes = pd.read_csv(QA / "nodes.csv")
    texts = qp.load_texts(nodes.paper_id.values)
    docs = {int(i): texts[int(p)] for i, p in zip(nodes.idx, nodes.paper_id)}
    years = nodes.year.values
    cs, cq, ca, cj = Cache("summaries"), Cache("questions"), Cache("answers"), Cache("judge")

    # stores: method x tau final summaries + FRESH reference
    stores = {}
    for m in ["recompute_leiden", "warmstart_leiden", "frontier_leiden", "hybrid"]:
        for tau, v in plan["methods"][m]["taus"].items():
            stores[f"{m}|tau={tau}"] = dict(final=v["final_store"], events=v["events"])
    lab = plan["methods"]["recompute_leiden"]["labels"][-1]
    node_t = np.load(QA / "node_t.npy")
    final_nodes = np.where(node_t >= 0)[0]
    comm = {}
    for v, c in zip(final_nodes, lab):
        comm.setdefault(int(c), []).append(int(v))
    fresh = [sorted(m_) for m_ in comm.values() if len(m_) >= qp.MIN_SIZE]
    stores["FRESH|recompute all"] = dict(final=fresh, events=[(plan["T"] - 1, s) for s in fresh])
    if limit_stores:
        stores = dict(list(stores.items())[:limit_stores])

    # 1) summaries: every member set that any store ever generated (cost) or holds at the end
    need = {key_of(s): s for st in stores.values() for s in [e[1] for e in st["events"]] + st["final"]}
    print(f"summaries needed: {len(need)} (cached {sum(1 for k in need if cs.get(k))})", flush=True)
    with cf.ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda s: summarise(s, docs, cs), need.values()))

    # 2) questions
    items = make_questions(docs, years, cq)[: (limit_q or N_QUESTIONS)]
    print(f"questions: {len(items)}", flush=True)

    # 3) answers + 4) judgements
    rows = []

    def work(args):
        name, st, it = args
        sums = [cs.get(key_of(s))["text"] for s in st["final"]]
        a = answer(it["question"], sums, ca, name)
        j = judge(it, a["text"], cj)
        return dict(store=name, q=it["question"], recall=float(np.mean(j["covered"])))

    jobs = [(n, st, it) for n, st in stores.items() for it in items]
    with cf.ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(work, jobs))
    res = pd.DataFrame(rows)

    # cost per store: tokens of all summary generations it triggered over the stream (excluding version 0,
    # which is identical for all maintainers) + per-update averages
    summary_rows = []
    for n, st in stores.items():
        ev = [e for e in st["events"] if e[0] > 0]
        inp = sum(cs.get(key_of(s))["usage"]["inp"] for _, s in ev)
        out = sum(cs.get(key_of(s))["usage"]["out"] for _, s in ev)
        r = res[res.store == n].recall
        boot = [np.mean(np.random.default_rng(b).choice(r.values, len(r))) for b in range(2000)]
        summary_rows.append(dict(store=n, recall=r.mean(), ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5),
                                 n_q=len(r), update_generations=len(ev), update_input_tokens=inp, update_output_tokens=out,
                                 final_summaries=len(st["final"])))
    out = pd.DataFrame(summary_rows).sort_values("recall", ascending=False)
    print(out.round(3).to_string(index=False))
    tot = {}
    for c in (cs, cq, ca, cj):
        for k, v in c.data.items():
            us = [v["usage"]] if isinstance(v, dict) and "usage" in v and isinstance(v["usage"], dict) else \
                (v.get("usage", []) if isinstance(v, dict) else [])
            for u in us:
                t = tot.setdefault(u["model"], [0, 0])
                t[0] += u["inp"]
                t[1] += u["out"]
    print("actual token usage by model:", tot)
    (QA / "results.json").write_text(json.dumps(dict(summary=out.to_dict("records"), per_question=rows,
                                                     usage=tot), indent=1))


if __name__ == "__main__":
    lq = int(sys.argv[1]) if len(sys.argv) > 1 else None
    ls = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(lq, ls)
