"""Check that every number the paper reports equals the output of the generating script.

Runs paper_assets/make_assets.py in a scratch directory on the cached results (artifacts/ or $PAPER_RESULTS) and
compares, NUMBER BY NUMBER, the regenerated macros (numbers.tex: every number quoted in the text) and every table with
the versions used in the paper (artifacts/expected_numbers.tex, artifacts/expected_tables/*.tex). Every mismatch is
listed with file, position and both values.  Exit code 1 if anything differs.
usage: python paper_assets/check_numbers.py            (about 2-5 minutes, no LLM calls)
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
NUM = re.compile(r"-?\d[\d{},]*\.?\d*")


def numbers(text):
    text = re.sub(r"\\(begin|end)\{[^}]*\}(\{[^}]*\})?", " ", text)
    return [t.replace("{,}", ",") for t in NUM.findall(text)]


def main():
    res = Path(os.environ.get("PAPER_RESULTS", REPO / "artifacts")).resolve()
    exp_num, exp_tab = REPO / "artifacts" / "expected_numbers.tex", REPO / "artifacts" / "expected_tables"
    tmp = Path(tempfile.mkdtemp(prefix="check_numbers_"))
    shutil.copy(HERE / "make_assets.py", tmp / "make_assets.py")
    r = subprocess.run([sys.executable, "make_assets.py"], cwd=tmp, capture_output=True, text=True,
                       env=dict(os.environ, PAPER_RESULTS=str(res)))
    if r.returncode != 0:
        print(r.stderr[-3000:])
        sys.exit("make_assets.py failed")
    bad, n_macros, n_table = [], 0, 0
    ma = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", exp_num.read_text()))
    mb = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (tmp / "numbers.tex").read_text()))
    for k in sorted(set(ma) | set(mb)):
        n_macros += 1
        if ma.get(k) != mb.get(k):
            bad.append(("numbers.tex", k, ma.get(k), mb.get(k)))
    for f in sorted(exp_tab.glob("*.tex")):
        g = tmp / "tables" / f.name
        if not g.exists():
            bad.append((f.name, "-", "table not produced", ""))
            continue
        na, nb = numbers(f.read_text()), numbers(g.read_text())
        n_table += len(na)
        if len(na) != len(nb):
            bad.append((f.name, "count", len(na), len(nb)))
        bad += [(f.name, f"number #{i + 1}", x, y) for i, (x, y) in enumerate(zip(na, nb)) if x != y]
    print(f"checked {n_macros} macros (numbers quoted in the text) and {n_table} numbers in "
          f"{len(list(exp_tab.glob('*.tex')))} tables against a fresh run of make_assets.py on {res}")
    if bad:
        print(f"MISMATCHES ({len(bad)}):")
        for b in bad:
            print("  ", b)
    else:
        print("OK: every number matches the script output")
    shutil.rmtree(tmp)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
