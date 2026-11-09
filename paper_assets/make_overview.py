"""Overview figure (Fig. 1): where summary-aware maintenance acts in a GraphRAG index on an evolving KG."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).parent
plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42})
BLUE, GREEN, INK, GREY, LIGHT = "#2a78d6", "#1baf7a", "#0b0b0b", "#52514e", "#f6f5f1"

fig, ax = plt.subplots(figsize=(4.75, 1.45))
ax.set_xlim(-0.6, 101.0)
ax.set_ylim(0, 30)
ax.axis("off")
W, GAP, Y, H = 21.0, 5.6, 1.0, 27.0


def box(i, title, lines, edge=GREY):
    x = i * (W + GAP)
    ax.add_patch(FancyBboxPatch((x + 0.4, Y), W - 0.8, H, boxstyle="round,pad=0.3,rounding_size=1.5",
                                fc=LIGHT, ec=edge, lw=0.9))
    ax.text(x + W / 2, Y + H - 2.4, title, ha="center", va="top", fontsize=6.6, weight="bold", color=INK)
    y = Y + H - 8.6
    for txt, col, it in lines:
        ax.text(x + W / 2, y, txt, ha="center", va="top", fontsize=5.4, color=col,
                style="italic" if it else "normal")
        y -= 3.6
    return x


xs = [box(0, "Graph update", [("version $G_t$:", GREY, 0), ("new nodes,", GREY, 0), ("new or removed", GREY, 0),
                               ("edges", GREY, 0)]),
      box(1, "Communities", [("warm-start or", GREY, 0), ("frontier Leiden", GREY, 0), ("+ summary-priced", BLUE, 1),
                              ("rollback (new)", BLUE, 1)], edge=BLUE),
      box(2, "Summaries", [("an LLM summary", GREY, 0), ("per community", GREY, 0), ("+ topic-anchored", GREEN, 1),
                            ("reuse (new)", GREEN, 1)], edge=GREEN),
      box(3, "Answers", [("global questions:", GREY, 0), ("all summaries", GREY, 0), ("local questions:", GREY, 0),
                          ("document search", GREY, 0)])]
for x in xs[:-1]:
    ax.add_patch(FancyArrowPatch((x + W - 0.2, Y + H / 2), (x + W + GAP + 0.2, Y + H / 2), arrowstyle="-|>",
                                 mutation_scale=7, lw=0.8, color=GREY))
fig.savefig(HERE / "figures" / "overview.pdf", bbox_inches="tight", pad_inches=0.02)
print("ok")
