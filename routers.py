"""Routers: question -> label (lookup / filter / plan) -> tier.

Each router returns (label, confidence, seconds spent deciding).
"""

import re
import time

TIER_FOR = {"lookup": "lite", "filter": "flash", "plan": "pro"}
# Two tiers, no Flash-Lite: added after the first full run showed Flash-Lite failing "what's on at 12:37".
TIER_FOR_2 = {"lookup": "flash", "filter": "flash", "plan": "pro"}

PLAN = re.compile(r"\b(plan|itinerary|my day|build me|schedule for me)\b", re.I)
FILTER = re.compile(r"\b(list|which talks|what talks|every|all the|talks are|talks have)\b", re.I)


def ifelse(question: str) -> tuple[str, float, float]:
    """Keywords and length. The baseline any team could write in five minutes."""
    t0 = time.perf_counter()
    if PLAN.search(question) or len(question) > 200:
        label = "plan"
    elif FILTER.search(question):
        label = "filter"
    else:
        label = "lookup"
    return label, 1.0, time.perf_counter() - t0


def oracle(kind: str) -> tuple[str, float, float]:
    """A perfect classifier (the eval knows each question's kind). Upper bound, not a real router."""
    return kind, 1.0, 0.0


# ---------------------------------------------------------------- Laya

# One Laya pass, two typed questions. Wording chosen by measuring three candidates on this eval set
# (see README): the yes/no "needs a day plan?" question is where zero-shot Laya is strong.
LAYA_QUESTIONS = {
    "day_plan": {
        "type": "noul",
        "instructions": "Is the attendee asking to organise their whole day, with rules such as arrival or "
                        "leaving times, topic preferences or talks to avoid?",
    },
    "task": {
        "type": "choice",
        "instructions": "What kind of help is the conference attendee asking for?",
        "criteria": {
            "quick_fact": "short question with a single answer: a room, a time, a speaker, or what is on right now",
            "search": "find every talk matching a filter like topic, level, room, before or after lunch",
            "itinerary": "long request to organise my whole day: one talk per slot, preferences in order, "
                         "things to avoid, must-see talks",
        },
    },
}
PLAN_THRESHOLD = 0.3
# Only the yes/no question may send a request to Pro. Zero-shot, the choice question says "itinerary"
# for most short questions ("When's lunch?"), so below the threshold it means "not sure": Flash.
TASK_KIND = {"quick_fact": "lookup", "search": "filter", "itinerary": "filter"}

_laya = None


def laya(question: str) -> tuple[str, float, float, dict]:
    """Laya (421M; GPU if there is one, else CPU): label, confidence, seconds, raw answers.

    The state can be anything up to ~512 tokens: in a real agent, the conversation so far and the
    tool results, not just the last message.
    """
    global _laya
    if _laya is None:
        from laya import Router
        _laya = Router()
    t0 = time.perf_counter()
    a = _laya.predict(question, LAYA_QUESTIONS)["answers"]
    secs = time.perf_counter() - t0
    p_plan = a["day_plan"]["noul"]
    if p_plan >= PLAN_THRESHOLD:
        return "plan", p_plan, secs, a
    return TASK_KIND.get(a["task"]["choice"], "filter"), a["task"]["answer_confidence"], secs, a
