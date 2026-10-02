# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Eval set: ~130 DevFest Helper questions, generated and graded by code.

Every question has a ground truth computed from the schedule JSON, so there's no
LLM-as-judge and no hand labelling. Three kinds, known by construction:
  lookup  a single fact (room, start time, who's talking, what's on)   -> ideal tier: Flash-Lite
  filter  a set of talks matching some conditions                        -> ideal tier: Flash
  plan    a full day under constraints, one talk per slot, best choice   -> ideal tier: Pro

The model answers in one JSON shape for every question (ANSWER_SCHEMA), because the
live helper doesn't know the kind in advance.

Usage:
  uv run eval_set.py            # build, self-test graders, write eval_set.json
  uv run eval_set.py --show 5   # also print 5 sample questions per kind
"""

import argparse
import hashlib
import json
import random
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
# data/schedule.json is the 1 Oct 2026 snapshot of Sessionize's public schedule API, trimmed to talks, rooms,
# topics and speaker names (no bios, photos or links). Delete it to download the live schedule instead.
SCHEDULE_URL = "https://gdg-melbourne-devfest-2026.sessionize.com/api/schedule"
SCHEDULE = HERE / "data" / "schedule.json"
# Fingerprint of the 1 Oct 2026 schedule that results/ was made with (see schedule_fingerprint).
SNAPSHOT = "439f7e55dfcee4f9"
OUT = HERE / "eval_set.json"
PARAPHRASES = HERE / "paraphrases.json"  # reworded questions, same answer keys (make_paraphrases.py)
SEED = 2026

LUNCH_START, LUNCH_END = "12:55", "13:40"

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "session_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "IDs of the talks that answer the question. For a day plan, in time order. Empty if not applicable.",
        },
        "room": {"type": "string", "description": "Exact room name, if the question asks for a room. Otherwise empty."},
        "time": {"type": "string", "description": "24-hour HH:MM, if the question asks for a time. Otherwise empty."},
        "answer": {"type": "string", "description": "A short answer for a human, one or two sentences."},
    },
    "required": ["session_ids", "room", "time", "answer"],
}

INSTRUCTIONS = (
    "Answer from the schedule JSON only. A 'talk' is a session with isServiceSession = false; "
    "never include service sessions (registration, breaks, lunch, drinks) in session_ids. "
    "'Before lunch' means the talk ends by 12:55; 'after lunch' means it starts at 13:40 or later. "
    "Level 5 rooms are The Auditorium and Flagstaff; the Lecture Theatre is on Level 3."
)


# ---------------------------------------------------------------- schedule model

def hhmm(ts: str) -> str:
    return ts[11:16]


def load_talks(path: Path = SCHEDULE) -> list[dict]:
    d = json.loads(path.read_text())
    rooms = {r["id"]: r["name"] for r in d["rooms"]}
    speakers = {s["id"]: s["fullName"] for s in d["speakers"]}
    items = {}
    for cat in d["categories"]:
        for it in cat["items"]:
            items[it["id"]] = (cat["title"].strip(), it["name"])
    talks = []
    for s in d["sessions"]:
        if s["isServiceSession"]:
            continue
        cats = dict(items[i] for i in s["categoryItems"])
        talks.append({
            "id": s["id"],
            "title": s["title"].strip(),
            "start": hhmm(s["startsAt"]),
            "end": hhmm(s["endsAt"]),
            "room": rooms[s["roomId"]],
            "speakers": [speakers[p] for p in s["speakers"]],
            "topic": cats.get("Topic Category"),
            "level": cats.get("Target Level"),
            "length": cats.get("Talk length"),
        })
    return sorted(talks, key=lambda t: (t["start"], t["room"]))


def schedule_fingerprint(path: Path = SCHEDULE) -> str:
    """Hash of everything the questions and answer keys are built from (talks and break times)."""
    breaks = {s["title"]: s["startsAt"] for s in json.loads(path.read_text())["sessions"] if s["isServiceSession"]}
    blob = json.dumps({"talks": load_talks(path), "breaks": breaks}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def fetch_schedule() -> None:
    if SCHEDULE.exists():
        return
    print(f"No {SCHEDULE.relative_to(HERE)}: downloading the live schedule from {SCHEDULE_URL}…", file=sys.stderr)
    try:
        with urllib.request.urlopen(SCHEDULE_URL, timeout=30) as r:
            data = r.read()
        json.loads(data)
    except Exception as e:
        sys.exit(f"Couldn't download the schedule ({e}). Save it to {SCHEDULE} by hand and re-run.")
    SCHEDULE.parent.mkdir(exist_ok=True)
    SCHEDULE.write_bytes(data)
    if schedule_fingerprint() != SNAPSHOT:
        warn_snapshot()


def warn_snapshot() -> None:
    print("WARNING: this schedule differs from the 1 Oct 2026 snapshot that results/ was made with "
          "(talks moved or changed after it was taken), so questions and answer keys differ too. "
          "Scores of the saved results/ files won't be valid; run a fresh eval instead.", file=sys.stderr)


fetch_schedule()  # on import, so every script has a schedule before it reads it


def slots(talks: list[dict]) -> dict[str, list[dict]]:
    by_start = defaultdict(list)
    for t in talks:
        by_start[t["start"]].append(t)
    return dict(sorted(by_start.items()))


def fmt12(t: str) -> str:
    h, m = map(int, t.split(":"))
    suffix = "am" if h < 12 else "pm"
    h = h % 12 or 12
    return f"{h}{suffix}" if m == 0 else f"{h}:{m:02d}{suffix}"


def short_room(room: str) -> str:
    name = room.split(" (")[0].replace("The ", "")
    return name if name == "Flagstaff" else "the " + name


def minutes(t: dict) -> int:
    (h1, m1), (h2, m2) = (map(int, t[k].split(":")) for k in ("start", "end"))
    return (h2 * 60 + m2) - (h1 * 60 + m1)


LEVEL5 = {"The Auditorium (Level 5)", "Flagstaff (Level 5)"}


# ---------------------------------------------------------------- generators

def gen_lookups(talks, rng) -> list[dict]:
    out = []
    sample = lambda n: rng.sample(talks, n)
    for t in sample(12):
        out.append({"kind": "lookup", "sub": "room_of", "question": f"Which room is \"{t['title']}\" in?",
                    "expect": {"room": t["room"]}})
    for t in sample(10):
        out.append({"kind": "lookup", "sub": "time_of", "question": f"What time does \"{t['title']}\" start?",
                    "expect": {"time": t["start"]}})
    solo = [t for t in talks if len(t["speakers"]) >= 1]
    for t in rng.sample(solo, 10):
        who = rng.choice(t["speakers"])
        out.append({"kind": "lookup", "sub": "talk_by", "question": f"What is {who} talking about?",
                    "expect": {"session_ids": [t["id"]]}})
    for t in sample(8):
        h, m = map(int, t["start"].split(":"))
        eh, em = map(int, t["end"].split(":"))
        mid = (h * 60 + m + eh * 60 + em) // 2
        at = f"{mid // 60:02d}:{mid % 60:02d}"
        out.append({"kind": "lookup", "sub": "on_at",
                    "question": f"What's on in {short_room(t['room'])} at {fmt12(at)}?",
                    "expect": {"session_ids": [t["id"]]}})
    return out


def gen_breaks() -> list[dict]:
    """Questions about service sessions (breaks), answered with a start time."""
    d = json.loads(SCHEDULE.read_text())
    start = {s["title"]: hhmm(s["startsAt"]) for s in d["sessions"] if s["isServiceSession"]}
    asks = [("When's lunch?", "Lunch"), ("When is morning tea?", "Morning Tea"),
            ("What time does afternoon tea start?", "Afternoon Tea"),
            ("When do the after event drinks start?", "After event drinks"), ("What time is the welcome?", "Welcome")]
    return [{"kind": "lookup", "sub": "break", "question": q, "expect": {"time": start[title]}} for q, title in asks]


def gen_filters(talks, rng) -> list[dict]:
    cands = []

    def add(sub, question, pred):
        ids = sorted(t["id"] for t in talks if pred(t))
        if ids:
            cands.append({"kind": "filter", "sub": sub, "question": question, "expect": {"session_ids": ids}})

    topics = sorted({t["topic"] for t in talks})
    levels = sorted({t["level"] for t in talks})
    for tp in topics:
        add("topic", f"List every {tp} talk.", lambda t, tp=tp: t["topic"] == tp)
    for tp in topics:
        for lv in levels:
            aim = "a non-technical audience" if lv == "Non-technical" else f"{lv.lower()} level"
            add("topic_level", f"Which {tp} talks are aimed at {aim}?",
                lambda t, tp=tp, lv=lv: t["topic"] == tp and t["level"] == lv)
    for tp in topics:
        add("topic_window", f"Which {tp} talks are after lunch?",
            lambda t, tp=tp: t["topic"] == tp and t["start"] >= LUNCH_END)
        add("topic_window", f"Which {tp} talks are before lunch?",
            lambda t, tp=tp: t["topic"] == tp and t["end"] <= LUNCH_START)
    for room in sorted({t["room"] for t in talks}):
        add("room_window", f"What talks are in {short_room(room)} before lunch?",
            lambda t, room=room: t["room"] == room and t["end"] <= LUNCH_START)
        add("room_window", f"What talks are in {short_room(room)} after lunch?",
            lambda t, room=room: t["room"] == room and t["start"] >= LUNCH_END)
    # Scheduled duration, not the "Talk length" field: one 15-minute talk sits in a 10-minute slot.
    add("length", "Which talks are only 10 minutes long?", lambda t: minutes(t) == 10)
    add("length", "Which talks are shorter than 35 minutes?", lambda t: minutes(t) < 35)
    add("speakers", "Which talks have more than one speaker?", lambda t: len(t["speakers"]) > 1)
    add("level5", "Which AI talks are on Level 5?", lambda t: t["topic"] == "AI" and t["room"] in LEVEL5)
    add("title", "Which talks have 'agent' or 'agentic' in the title?", lambda t: "agent" in t["title"].lower())
    rng.shuffle(cands)
    return cands[:35]


def best_plan(talks, c) -> list[dict]:
    """Reference solution: one allowed talk per slot, best preference rank, must-see forced."""
    plan = []
    for start, options in slots(talks).items():
        allowed = [t for t in options if allowed_talk(t, c)]
        if not allowed:
            continue
        must = [t for t in allowed if t["id"] in c["must"]]
        plan.append(must[0] if must else min(allowed, key=lambda t: rank(t, c)))
    return plan


def allowed_talk(t, c) -> bool:
    return (t["start"] >= c["arrive"] and t["end"] <= c["leave"]
            and (not c["level5"] or t["room"] in LEVEL5)
            and t["topic"] not in c["avoid"])


def rank(t, c) -> int:
    return c["prefer"].index(t["topic"]) if t["topic"] in c["prefer"] else len(c["prefer"])


def gen_plans(talks, rng, n=25) -> list[dict]:
    topics = sorted({t["topic"] for t in talks})
    out, seen = [], set()
    while len(out) < n:
        c = {
            "arrive": rng.choice(["09:00", "09:00", "10:00", "11:00"]),
            "leave": rng.choice(["19:00", "15:00", "16:05", "14:15"]),
            "prefer": rng.sample(topics, rng.choice([1, 2, 2, 3])),
            "avoid": [],
            "level5": rng.random() < 0.35,
            "must": [],
        }
        rest = [tp for tp in topics if tp not in c["prefer"]]
        if rng.random() < 0.4:
            c["avoid"] = [rng.choice(rest)]
        pool = [t for t in talks if allowed_talk(t, c)]
        if rng.random() < 0.4 and pool:
            c["must"] = [rng.choice(pool)["id"]]
        plan = best_plan(talks, c)
        key = json.dumps(c, sort_keys=True)
        if len(plan) < 3 or key in seen:
            continue
        seen.add(key)
        out.append({"kind": "plan", "sub": "day", "question": plan_question(talks, c), "expect": {"constraints": c}})
    return out


def plan_question(talks, c) -> str:
    title = {t["id"]: t["title"] for t in talks}
    parts = ["Plan my day."]
    if c["arrive"] > "09:00":
        parts.append(f"I arrive at {fmt12(c['arrive'])}.")
    if c["leave"] < "19:00":
        parts.append(f"I have to leave by {fmt12(c['leave'])}.")
    parts.append("I want one talk in every time slot I'm here for, with no clashes.")
    pref = c["prefer"]
    if len(pref) == 1:
        parts.append(f"Pick {pref[0]} talks whenever there is one.")
    else:
        parts.append(f"Prefer {pref[0]} talks, then {', then '.join(pref[1:])}, and anything else only if nothing I prefer is on.")
    if c["avoid"]:
        parts.append(f"No {c['avoid'][0]} talks at all.")
    if c["level5"]:
        parts.append("I can't do stairs, so Level 5 rooms only.")
    if c["must"]:
        parts.append(f"I must see \"{title[c['must'][0]]}\".")
    parts.append("Give the talks in time order.")
    return " ".join(parts)


# ---------------------------------------------------------------- grading

def grade(item: dict, answer: dict, talks: list[dict]) -> tuple[bool, str]:
    """True if the answer solves the task. The reason explains a failure in a few words."""
    exp = item["expect"]
    ids = [str(i).strip() for i in answer.get("session_ids") or []]
    if "room" in exp:
        norm = lambda r: r.lower().split(" (")[0].removeprefix("the ").strip()
        got = answer.get("room") or ""
        ok = norm(got) == norm(exp["room"])
        return ok, "" if ok else f"room {got!r}"
    if "time" in exp:
        got = (answer.get("time") or "").strip()
        got = got if len(got) != 4 else "0" + got  # 9:20 -> 09:20
        return got == exp["time"], "" if got == exp["time"] else f"time {got!r}"
    if "session_ids" in exp:
        want, got = set(exp["session_ids"]), set(ids)
        if got == want:
            return True, ""
        return False, f"{len(got & want)}/{len(want)} right, {len(got - want)} extra"
    return grade_plan(exp["constraints"], ids, talks)


def grade_plan(c: dict, ids: list[str], talks: list[dict]) -> tuple[bool, str]:
    by_id = {t["id"]: t for t in talks}
    if any(i not in by_id for i in ids):
        return False, "unknown or service session"
    chosen = [by_id[i] for i in ids]
    if any(not allowed_talk(t, c) for t in chosen):
        return False, "breaks a constraint"
    starts = [t["start"] for t in chosen]
    if len(set(starts)) != len(starts):
        return False, "clash"
    for m in c["must"]:
        if m not in ids:
            return False, "missed must-see"
    need = {s for s, opts in slots(talks).items() if any(allowed_talk(t, c) for t in opts)}
    if set(starts) != need:
        return False, f"{len(need - set(starts))} slot(s) missing"
    for t in chosen:
        if t["id"] in c["must"]:
            continue
        best = min(rank(o, c) for o in slots(talks)[t["start"]] if allowed_talk(o, c))
        if rank(t, c) != best:
            return False, f"worse choice at {t['start']}"
    return True, ""


def reference_answer(item: dict, talks: list[dict]) -> dict:
    exp = item["expect"]
    ans = {"session_ids": [], "room": "", "time": "", "answer": ""}
    if "constraints" in exp:
        ans["session_ids"] = [t["id"] for t in best_plan(talks, exp["constraints"])]
    else:
        ans.update({k: v for k, v in exp.items()})
    return ans


# ---------------------------------------------------------------- build

def build(natural: bool = True) -> list[dict]:
    """Templated questions, plus reworded copies (id prefixed "n") if paraphrases.json exists."""
    talks = load_talks()
    rng = random.Random(SEED)
    items = gen_lookups(talks, rng) + gen_filters(talks, rng) + gen_plans(talks, rng)
    for n, it in enumerate(items, 1):
        it["id"] = f"{it['kind'][0]}{n:03d}"
        it["style"] = "template"
    # Added after the first full run (the live helper misrouted "When's lunch?"): own ids, so nothing shifts.
    for n, it in enumerate(gen_breaks(), 1):
        items.append({**it, "id": f"s{n:02d}", "style": "template"})
    if natural and PARAPHRASES.exists():
        by_id = {it["id"]: it for it in items}
        for src, q in json.loads(PARAPHRASES.read_text()).items():
            items.append({**by_id[src], "id": "n" + src, "question": q, "style": "natural", "source": src})
    return items


def self_test(items, talks) -> None:
    """Reference answers must pass; obviously wrong ones must fail."""
    for it in items:
        ok, why = grade(it, reference_answer(it, talks), talks)
        assert ok, (it["id"], why)
        wrong = {"session_ids": [], "room": "Nowhere", "time": "00:00", "answer": ""}
        assert not grade(it, wrong, talks)[0], it["id"]
    # A plan with one slot swapped for a worse option must fail.
    plans = [it for it in items if it["kind"] == "plan"]
    broken = 0
    for it in plans:
        c = it["expect"]["constraints"]
        ref = best_plan(talks, c)
        for i, t in enumerate(ref):
            worse = [o for o in slots(talks)[t["start"]] if o["id"] != t["id"] and o["id"] not in c["must"]
                     and (not allowed_talk(o, c) or rank(o, c) > rank(t, c))]
            if worse and t["id"] not in c["must"]:
                ids = [x["id"] for x in ref]
                ids[i] = worse[0]["id"]
                assert not grade_plan(c, ids, talks)[0], it["id"]
                broken += 1
                break
    print(f"self-test ok: {len(items)} reference answers pass, {len(items)} wrong answers fail, "
          f"{broken} swapped plans fail")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", type=int, default=0, help="print N sample questions per kind")
    args = ap.parse_args()

    talks = load_talks()
    items = build()
    self_test(items, talks)

    counts = defaultdict(int)
    for it in items:
        counts[f"{it['style']} {it['kind']}"] += 1
    print("questions:", dict(counts), "total", len(items))

    if args.show:
        for kind in ("lookup", "filter", "plan"):
            print(f"\n{kind}:")
            for it in [i for i in items if i["kind"] == kind][: args.show]:
                print(f"  {it['id']}  {it['question']}")

    OUT.write_text(json.dumps({"seed": SEED, "schema": ANSWER_SCHEMA, "instructions": INSTRUCTIONS,
                               "items": items}, indent=2))
    print(f"wrote {OUT.relative_to(HERE)}")


if __name__ == "__main__":
    main()
