"""How much input does a community summary need?  (cost lever per regeneration)

For the final communities of the FRESH reference and of the hybrid tau=0.9 store, write summaries from cheaper
inputs and let gpt-5.6-terra rate each against the full summary (titles + abstracts, up to 24k characters, as in
qa_run.py) with the same 1-5 scale and judge prompt as delta_summary.py. The noise floor is an independent second
full summary (cached by delta_summary.py).
Variants: titles of up to 300 members | titles + abstracts of 10 members | titles + abstracts of 20 members.
Members are taken in index order as in qa_run.py (index is unrelated to year).
usage: python input_variants.py
"""

import concurrent.futures as cf
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
import delta_summary as ds  # noqa: E402
import qa_plan as qp  # noqa: E402
import qa_run as qr  # noqa: E402

HERE = config.RESULTS
QA = HERE / "qa"
VARIANTS = {"titles<=300": dict(abstracts=0, titles=300), "abstracts x10": dict(abstracts=10, titles=0),
            "abstracts x20": dict(abstracts=20, titles=0)}


def summarise_variant(members, docs, name, cache):
    k = f"{name}|{qr.key_of(members)}"
    hit = cache.get(k)
    if hit:
        return hit
    v = VARIANTS[name]
    mem = sorted(members)
    if v["abstracts"]:
        parts = [f"- {docs[m][0]}: {docs[m][1][:600]}" for m in mem[:v["abstracts"]]]
    else:
        parts = [f"- {docs[m][0]}" for m in mem[:v["titles"]]]
    prompt = ("You write community reports for a GraphRAG index over computer-science papers. Below are the "
              f"papers of one community ({len(members)} papers, {len(parts)} shown). Write a report of at most "
              "200 words: a short title, the main research themes, typical methods, and recent trends. "
              "Use only the given papers.\n\n" + "\n".join(parts))
    text, usage = qr.call(qr.SUM_MODEL, [{"role": "user", "content": prompt}], max_tokens=1500)
    out = dict(text=text, usage=usage)
    cache.put(k, out)
    return out


def main():
    plan = json.loads((QA / "plan.json").read_text())
    node_t = np.load(QA / "node_t.npy")
    nodes = pd.read_csv(QA / "nodes.csv")
    texts = qp.load_texts(nodes.paper_id.values)
    docs = {int(i): texts[int(p)] for i, p in zip(nodes.idx, nodes.paper_id)}
    cs, cv, cr = qr.Cache("summaries"), qr.Cache("variants"), qr.Cache("delta_rate")
    sets = {}
    for m in ["recompute_leiden", "hybrid"]:
        for s in plan["methods"][m]["taus"]["0.9"]["final_store"]:
            sets[qr.key_of(s)] = sorted(s)
    sets = list(sets.values())
    jobs = [(s, v) for s in sets for v in VARIANTS]
    with cf.ThreadPoolExecutor(8) as ex:
        outs = list(ex.map(lambda j: summarise_variant(j[0], docs, j[1], cv), jobs))
    with cf.ThreadPoolExecutor(8) as ex:
        rates = list(ex.map(lambda jo: ds.rate(jo[1]["text"], qr.summarise(jo[0][0], docs, cs)["text"], cr)["change"],
                            zip(jobs, outs)))
    rows = [dict(variant=v, n=len(s), inp=o["usage"]["inp"], full_inp=qr.summarise(s, docs, cs)["usage"]["inp"],
                 change=r) for (s, v), o, r in zip(jobs, outs, rates)]
    noise = [r for k, r in cr.data.items()]  # not used directly; floor taken from delta_summary.json
    floor = pd.DataFrame(json.loads((HERE / "delta_summary.json").read_text())["fidelity"])
    floor = floor[floor.kind == "full-again"]
    df = pd.DataFrame(rows)
    df["substantive"] = df.change >= 3
    out = df.groupby("variant").agg(n=("change", "size"), mean_change=("change", "mean"),
                                    substantive=("substantive", "mean"), input_tokens=("inp", "mean"),
                                    full_input_tokens=("full_inp", "mean"))
    out["input_share"] = out.input_tokens / out.full_input_tokens
    out.loc["full again (noise floor)"] = [len(floor), floor.change.mean(), (floor.change >= 3).mean(),
                                           np.nan, np.nan, 1.0]
    print(out.round(3).to_string())
    big = df[df.n >= 40].groupby("variant").agg(n=("change", "size"), substantive=("substantive", "mean"))
    print("communities with >= 40 members:\n", big.round(3).to_string())
    del noise
    (HERE / "input_variants.json").write_text(json.dumps(dict(rows=rows, summary=out.reset_index().to_dict("records")),
                                                         indent=1, default=float))


if __name__ == "__main__":
    main()
