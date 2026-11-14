"""C3 (V12): 300 local questions for the two cheapest stores, warm-start and frontier-restricted Leiden with
"never" refreshed reports, against fresh reports - fully local Qwen3-14B, identical pipeline and settings to V11's C2
(analysis/v2/c2_qwen_qa.py: qa_run.summarise / answer (TF-IDF top 6) / judge (key-point recall), num_ctx 16384,
temperature 0, seed 0), same TOST (margin 0.05, 90% CI).  The 300 questions are the C2 question set (qa_run.make_questions
rng 7 -> 120, qa_more.more_questions rng 8 -> 180; already generated and cached by C2).
The C2 caches are COPIED to v12_qwen/cache_*_c3.json and only the copies are extended; the V11 caches and the V11 result
file are not touched.  Output: v12_qwen/c3_never_300.json
"""
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
A = HERE.parent
sys.path[:0] = [str(A), str(A / "v2")]
os.environ.setdefault("OPENAI_API_KEY", "unused-local-only")
for name in ("summaries", "questions", "questions_more", "answers", "judge"):
    src, dst = A / "qa" / f"cache_{name}_c2qwen.json", HERE / f"cache_{name}_c3.json"
    if src.exists() and not dst.exists():
        shutil.copy(src, dst)
import c2_qwen_qa as c2  # noqa: E402   (patches qa_run.call -> local Qwen with the V11 C2 settings)

_base_init = c2._orig_init


def _init(self, name):
    self.path = HERE / f"cache_{name}_c3.json"
    import json
    self.data = json.loads(self.path.read_text()) if self.path.exists() else {}


c2.qr.Cache.__init__ = _init
c2.STORES = ["FRESH|recompute all", "warmstart_leiden|tau=0.0", "frontier_leiden|tau=0.0"]
c2.HERE = HERE                                   # result goes to v12_qwen/c2_qwen_qa.json, renamed below

if __name__ == "__main__":
    c2.main(300)
    (HERE / "c2_qwen_qa.json").rename(HERE / "c3_never_300.json")
