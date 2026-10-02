"""Shared by every script: tiers, prices, cost and the answer validator."""

import re

# Cheapest to dearest. Every tier must be on the same location: these three are global / us / eu only.
TIERS = ["lite", "flash", "pro"]
MODELS = {
    "lite": "gemini-3.5-flash-lite",
    "flash": "gemini-3.8-flash",
    "pro": "gemini-3.1-pro-preview",
}
THINKING = {"lite": "minimal", "flash": "low", "pro": "high"}
LOCATION = "global"

# Vertex AI, global, <=200k context, $ per 1M tokens. Output covers thinking tokens.
# Storage = explicit cache, per 1M tokens per hour. Checked 30 Sep 2026.
PRICES = {
    "today": {
        "lite": {"input": 0.30, "cached": 0.03, "output": 2.50, "storage": 1.00},
        "flash": {"input": 0.75, "cached": 0.075, "output": 3.75, "storage": 1.00},
        "pro": {"input": 2.00, "cached": 0.20, "output": 12.00, "storage": 4.50},
    },
    # 3.7 / 3.8 Flash double on 1 Jan 2027; the others stay. Storage assumed unchanged.
    "jan": {
        "lite": {"input": 0.30, "cached": 0.03, "output": 2.50, "storage": 1.00},
        "flash": {"input": 1.50, "cached": 0.15, "output": 7.50, "storage": 1.00},
        "pro": {"input": 2.00, "cached": 0.20, "output": 12.00, "storage": 4.50},
    },
}


# Router compute: the Laya share of a Cloud Run instance, priced as 1 vCPU + 2 GiB with request-based
# billing ($0.000024/vCPU-s + $0.0000025/GiB-s). The deployed service is 4 vCPU / 8 GiB (see README).
# Not checked against today's price list. A min-instance kept warm costs extra and isn't included.
ROUTER_PER_SECOND = 0.000024 + 2 * 0.0000025


def call_cost(c: dict, p: dict) -> float:
    """Token cost of one logged call (prompt, cached, output, thinking counts)."""
    return ((c["prompt"] - c["cached"]) * p["input"] + c["cached"] * p["cached"]
            + (c["output"] + c["thinking"]) * p["output"]) / 1e6


TIME = re.compile(r"^\d{1,2}:\d{2}$")


def validate(answer: dict | None, talks: list[dict], label: str) -> tuple[bool, str]:
    """Cheap checks with no answer key, i.e. what a live system can check before replying.

    A failure here is what triggers escalation to the next tier.
    """
    if not isinstance(answer, dict):
        return False, "not JSON"
    by_id = {t["id"]: t for t in talks}
    ids = [str(i) for i in answer.get("session_ids") or []]
    room = (answer.get("room") or "").strip()
    time = (answer.get("time") or "").strip()
    if not (ids or room or time):
        return False, "empty"
    if any(i not in by_id for i in ids):
        return False, "unknown session id"
    rooms = {t["room"].lower().split(" (")[0].removeprefix("the ") for t in talks}
    if room and room.lower().split(" (")[0].removeprefix("the ").strip() not in rooms:
        return False, "unknown room"
    if time and not TIME.match(time):
        return False, "bad time"
    if label == "plan":
        starts = [by_id[i]["start"] for i in ids]
        if len(ids) < 2:
            return False, "plan too short"
        if len(set(starts)) != len(starts):
            return False, "clash"
        if starts != sorted(starts):
            return False, "not in time order"
    return True, ""
