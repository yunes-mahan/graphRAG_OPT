"""Human check of the LLM judge (Cohen's kappa).
  python human_check.py make   -> human_check/keypoints.csv (100 key-point verdicts) and human_check/changes.csv
                                  (50 report pairs: 25 judged substantive, 25 not), judge verdicts kept separately
  python human_check.py kappa  -> Cohen's kappa between the filled-in human labels and the judge
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wikidata"))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
HERE = config.RESULTS
OUT = config.REPO / "human_check"
QA = HERE / "qa"


def load(n):
    return json.loads((QA / f"cache_{n}.json").read_text())


def make():
    rng = np.random.default_rng(42)
    res = json.loads((QA / "results.json").read_text())["per_question"]
    items = {it["question"]: it for it in load("questions")["questions"]["items"]}
    ans, jud = load("answers"), load("judge")
    pick = [res[i] for i in rng.choice(len(res), 25, replace=False)]
    rows, key = [], []
    for r in pick:
        a = ans[hashlib.sha1((r["store"] + "|" + r["q"]).encode()).hexdigest()]["text"]
        j = jud[hashlib.sha1((r["q"] + "|" + a).encode()).hexdigest()]["covered"]
        for k, kp in enumerate(items[r["q"]]["key_points"]):
            rid = f"K{len(rows) + 1:03d}"
            rows.append(dict(id=rid, question=r["q"], key_point=kp, answer=a, covered_yes_no=""))
            key.append(dict(id=rid, judge=int(bool(j[k]))))
    with open(OUT / "keypoints.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    st, sm = load("stale_judge"), load("summaries")
    pos = [k for k, v in st.items() if v["stale"]]
    neg = [k for k, v in st.items() if not v["stale"]]
    sel = list(rng.choice(pos, 25, replace=False)) + list(rng.choice(neg, 25, replace=False))
    rng.shuffle(sel)
    rows2 = []
    for k in sel:
        ko, kn = k.split("|")
        rid = f"C{len(rows2) + 1:03d}"
        rows2.append(dict(id=rid, summary_A_old=sm[ko]["text"], summary_B_current=sm[kn]["text"], substantive_yes_no=""))
        key.append(dict(id=rid, judge=int(st[k]["stale"])))
    with open(OUT / "changes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows2[0]))
        w.writeheader()
        w.writerows(rows2)
    (HERE / "human_check_key.json").write_text(json.dumps(key, indent=1))
    print(len(rows), "key-point rows,", len(rows2), "change pairs")


def kappa():
    from sklearn.metrics import cohen_kappa_score
    key = {r["id"]: r["judge"] for r in json.loads((HERE / "human_check_key.json").read_text())}
    out = {}
    for fn, col in (("keypoints.csv", "covered_yes_no"), ("changes.csv", "substantive_yes_no")):
        h, j = [], []
        for r in csv.DictReader(open(OUT / fn)):
            v = r[col].strip().lower()
            if v in ("yes", "y", "1", "no", "n", "0"):
                h.append(int(v in ("yes", "y", "1")))
                j.append(key[r["id"]])
        out[fn] = dict(n=len(h), kappa=float(cohen_kappa_score(h, j)), agreement=float(np.mean(np.array(h) == np.array(j))))
    print(out)
    (HERE / "human_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"make": make, "kappa": kappa}[sys.argv[1]]()
