# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["laya>=0.3.22"]
# ///
"""Run Laya once over the eval set and cache its routing labels, so scoring doesn't need PyTorch.

First run downloads the checkpoint (~800 MB, about a minute).

Usage:
  uv run label_laya.py
"""

import json
from collections import Counter
from pathlib import Path

import routers
from eval_set import build

HERE = Path(__file__).parent
OUT = HERE / "results" / "laya_labels.json"


def main() -> None:
    items = build()
    routers.laya("warm-up")  # load weights before timing
    labels, right = {}, Counter()
    for it in items:
        label, conf, secs, a = routers.laya(it["question"])
        labels[it["id"]] = {"label": label, "confidence": conf, "seconds": secs,
                            "p_day_plan": a["day_plan"]["noul"], "task": a["task"]["choice"]}
        right[(it["style"], label == it["kind"])] += 1
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"questions": routers.LAYA_QUESTIONS, "plan_threshold": routers.PLAN_THRESHOLD,
                               "labels": labels}, indent=2))
    ms = sum(v["seconds"] for v in labels.values()) / len(labels) * 1000
    for style in ("template", "natural"):
        n = right[(style, True)] + right[(style, False)]
        print(f"Laya picked the intended kind for {right[(style, True)]}/{n} {style} questions")
    print(f"{ms:.0f} ms per decision · wrote {OUT.relative_to(HERE)}")


if __name__ == "__main__":
    main()
