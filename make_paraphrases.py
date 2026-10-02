# /// script
# requires-python = ">=3.11"
# dependencies = ["google-genai>=2.25"]
# ///
"""One-off: reword 30 eval questions the way attendees actually type, same ground truth.

The templated questions give routers easy keywords ("Plan my day", "Which talks").
These don't. Output goes to paraphrases.json; check it by hand before using it,
because a reworded question that drops a constraint has the wrong answer key.

Usage:
  uv run make_paraphrases.py
"""

import json
import os
import random
import sys
from pathlib import Path

from eval_set import build

HERE = Path(__file__).parent
OUT = HERE / "paraphrases.json"
PER_KIND = 10

PROMPT = """Rewrite each question the way a real conference attendee would type it into a chat app on their phone.

Rules:
- Keep every fact and constraint exactly: times, topics, levels, rooms, must-see talks, preferences and their order, "no X talks", Level 5 / stairs, arrival and leaving times, "before/after lunch".
- A talk title may be shortened, but only so it still clearly names that one talk.
- Vary the style: casual, first person, sometimes no question mark, sometimes context first ("first devfest, ...").
- Avoid the template phrases: "Plan my day", "Which talks", "What talks", "List every", "Give the talks in time order", "time slot".
- For day plans, keep the meaning of: one talk in every slot while they're there, no clashes, their preference order, and "anything else only if nothing preferred is on".
- One or two sentences, under 60 words.

Return JSON: a list of {"id": ..., "question": ...} in the same order.

Questions:
"""


def main() -> None:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        sys.exit("Set GOOGLE_CLOUD_PROJECT.")
    from google import genai
    from google.genai import types

    rng = random.Random(7)
    items = build(natural=False)
    picked = [it for k in ("lookup", "filter", "plan")
              for it in rng.sample([i for i in items if i["kind"] == k], PER_KIND)]
    listing = "\n".join(f'{it["id"]}: {it["question"]}' for it in picked)

    client = genai.Client(vertexai=True, project=project, location="global")
    resp = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=[PROMPT + listing],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema={
                "type": "array",
                "items": {"type": "object", "properties": {"id": {"type": "string"}, "question": {"type": "string"}},
                          "required": ["id", "question"]},
            },
            labels={"demo": "routing", "strategy": "paraphrase"},
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    out = {r["id"]: r["question"] for r in json.loads(resp.text)}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    by_id = {it["id"]: it for it in picked}
    for i, q in out.items():
        print(f"{i}\n  was: {by_id[i]['question']}\n  now: {q}\n")
    print(f"wrote {OUT.relative_to(HERE)} ({len(out)} questions). Review before running the eval.")


if __name__ == "__main__":
    main()
