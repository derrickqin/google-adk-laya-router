# /// script
# requires-python = ">=3.11"
# dependencies = ["google-genai>=2.25", "rich>=13"]
# ///
"""Scoreboard: run every eval question once on every tier, then score strategies offline.

Each (question, tier) is called once, with the schedule in an explicit cache per model.
Strategies (always-X, routers, escalation, cascade) are then computed from those answers,
so adding a strategy or re-pricing for January costs nothing.

Usage:
  uv run run_eval.py --per-kind 1          # smoke test: 3 questions x 3 tiers
  uv run run_eval.py                       # full run: 130 questions x 3 tiers
  uv run run_eval.py --resume <file>       # finish an interrupted run
  uv run run_eval.py --score <file>        # scoreboard only, no API calls
  uv run run_eval.py --score <file> --style natural   # reworded questions only
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from rich.console import Console

import routers
from common import LOCATION, MODELS, PRICES, ROUTER_PER_SECOND, THINKING, TIERS, call_cost, validate
from eval_set import (ANSWER_SCHEMA, INSTRUCTIONS, SCHEDULE, SNAPSHOT, build, grade, load_talks,
                      schedule_fingerprint, warn_snapshot)

HERE = Path(__file__).parent
RESULTS = HERE / "results"
LAYA_LABELS = RESULTS / "laya_labels.json"  # written by label_laya.py
LAYA: dict = {}
SYSTEM = "You are DevFest Helper for GDG Melbourne DevFest 2026 on 3 October 2026. " + INSTRUCTIONS

console = Console()


# ---------------------------------------------------------------- run

def make_cache(client, types, tier: str, schedule: str):
    return client.caches.create(
        model=MODELS[tier],
        config=types.CreateCachedContentConfig(
            system_instruction=SYSTEM,
            contents=[types.Content(role="user", parts=[types.Part(text="DevFest schedule (JSON):\n" + schedule)])],
            ttl="3600s",
            display_name=f"devfest-eval-{tier}",
        ),
    )


def ask(client, types, tier: str, cache_name: str, item: dict) -> dict:
    config = types.GenerateContentConfig(
        cached_content=cache_name,
        response_mime_type="application/json",
        response_json_schema=ANSWER_SCHEMA,
        thinking_config=types.ThinkingConfig(thinking_level=THINKING[tier]),
        labels={"demo": "routing", "tier": tier, "strategy": "eval"},
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    err = ""
    for attempt in range(4):
        try:
            t0 = time.perf_counter()
            resp = client.models.generate_content(model=MODELS[tier], contents=[item["question"]], config=config)
            latency = time.perf_counter() - t0
            break
        except Exception as e:  # 429 / 5xx: back off and retry
            err = str(e)[:200]
            time.sleep(2 ** attempt * 2)
    else:
        return {"qid": item["id"], "tier": tier, "error": err, "answer": None,
                "prompt": 0, "cached": 0, "output": 0, "thinking": 0, "latency": 0.0}
    u = resp.usage_metadata
    try:
        answer = json.loads(resp.text)
    except Exception:
        answer = None
    return {
        "qid": item["id"], "tier": tier, "error": "", "answer": answer,
        "prompt": u.prompt_token_count or 0, "cached": u.cached_content_token_count or 0,
        "output": u.candidates_token_count or 0, "thinking": u.thoughts_token_count or 0,
        "latency": latency,
    }


def run(args, items: list[dict], talks: list[dict], out: Path, saved: dict) -> dict:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        sys.exit("Set GOOGLE_CLOUD_PROJECT (and run: gcloud auth application-default login).")
    from google import genai
    from google.genai import types

    client = genai.Client(vertexai=True, project=project, location=LOCATION)
    schedule = SCHEDULE.read_text()
    done = {(c["qid"], c["tier"]) for c in saved["calls"] if not c["error"]}
    saved["calls"] = [c for c in saved["calls"] if not c["error"]]
    by_id = {it["id"]: it for it in items}

    try:
        for tier in args.tiers:
            todo = [it for it in items if (it["id"], tier) not in done]
            if not todo:
                continue
            console.print(f"\n[bold]{tier} · {MODELS[tier]}[/] [dim]thinking={THINKING[tier]} · {len(todo)} questions[/]")
            created = time.time()
            cache = make_cache(client, types, tier, schedule)
            try:
                with ThreadPoolExecutor(args.workers) as pool:
                    futs = [pool.submit(ask, client, types, tier, cache.name, it) for it in todo]
                    for f in as_completed(futs):
                        c = f.result()
                        saved["calls"].append(c)
                        print_call(c, by_id[c["qid"]], talks)
            finally:
                tokens = cache.usage_metadata.total_token_count if cache.usage_metadata else 0
                hours = (time.time() - created) / 3600
                s = saved["storage"].setdefault(tier, {"tokens": tokens, "hours": 0.0})
                s["hours"] += hours
                client.caches.delete(name=cache.name)
    finally:
        out.write_text(json.dumps(saved, indent=2))
        console.print(f"\n[dim]Saved {out.relative_to(HERE)}[/]")
    return saved


def print_call(c: dict, item: dict, talks: list[dict]) -> None:
    if c["error"]:
        console.print(f"  [red]ERR[/] {c['tier']:<5} {item['id']}  {c['error'][:60]}")
        return
    ok, why = grade(item, c["answer"] or {}, talks)
    mark = "[green]✓[/]" if ok else "[red]✗[/]"
    cost = call_cost(c, PRICES["today"][c["tier"]])
    console.print(
        f"  {mark} {c['tier']:<5} {item['id']}  {item['kind']:<6}  "
        f"{c['output'] + c['thinking']:>6,} out  ${cost:.4f}  {c['latency']:>5.1f}s  [dim]{why}[/]"
    )


# ---------------------------------------------------------------- score

def strategies(items: list[dict]) -> dict:
    """Strategy name -> function(item) -> (starting tier, validator label, escalate?, router seconds)."""
    ie = lambda it: routers.ifelse(it["question"])[0]
    s = {f"Always {t.capitalize() if t != 'lite' else 'Flash-Lite'}": (lambda it, t=t: (t, "any", False, 0.0)) for t in TIERS}
    s["if/else 3-tier"] = lambda it: _routed(ie(it), False)
    s["if/else 3-tier + esc."] = lambda it: _routed(ie(it), True)
    if LAYA:
        s["Laya 3-tier"] = lambda it: _routed(LAYA[it["id"]]["label"], False, secs=LAYA[it["id"]]["seconds"])
        s["Laya 3-tier + esc."] = lambda it: _routed(LAYA[it["id"]]["label"], True, secs=LAYA[it["id"]]["seconds"])
    s["Cascade (Lite → up)"] = lambda it: ("lite", "any", True, 0.0)
    s["Perfect 3-tier*"] = lambda it: _routed(it["kind"], False)
    s["if/else Flash/Pro"] = lambda it: _routed(ie(it), False, routers.TIER_FOR_2)
    if LAYA:
        s["Laya Flash/Pro"] = lambda it: _routed(LAYA[it["id"]]["label"], False, routers.TIER_FOR_2, LAYA[it["id"]]["seconds"])
        s["Laya Flash/Pro + esc."] = lambda it: _routed(LAYA[it["id"]]["label"], True, routers.TIER_FOR_2, LAYA[it["id"]]["seconds"])
    s["Perfect Flash/Pro*"] = lambda it: _routed(it["kind"], False, routers.TIER_FOR_2)
    return s


def _routed(label: str, escalate: bool, tiers: dict = routers.TIER_FOR, secs: float = 0.0):
    return tiers[label], label, escalate, secs


def score(saved: dict, items: list[dict], talks: list[dict], prices: dict, failure_cost: float) -> list[dict]:
    res = {(c["qid"], c["tier"]): c for c in saved["calls"] if not c["error"]}
    calls_per_tier = {t: sum(1 for c in res.values() if c["tier"] == t) for t in TIERS}
    storage_per_call = {
        t: (s["tokens"] * prices[t]["storage"] * s["hours"] / 1e6) / max(calls_per_tier[t], 1)
        for t, s in saved["storage"].items()
    }
    usable = [it for it in items if all((it["id"], t) in res for t in TIERS)]
    rows = []
    for name, pick in strategies(usable).items():
        total, wins, secs, mix = 0.0, 0, 0.0, {t: 0 for t in TIERS}
        for it in usable:
            tier, label, escalate, router_secs = pick(it)
            total += router_secs * ROUTER_PER_SECOND
            secs += router_secs
            while True:
                c = res[(it["id"], tier)]
                total += call_cost(c, prices[tier]) + storage_per_call.get(tier, 0.0)
                mix[tier] += 1
                secs += c["latency"]
                valid = validate(c["answer"], talks, label)[0]
                if valid or not escalate or tier == TIERS[-1]:
                    break
                tier = TIERS[TIERS.index(tier) + 1]
            wins += grade(it, c["answer"] or {}, talks)[0]
        rows.append({"name": name, "n": len(usable), "wins": wins, "total": total,
                     "per_success": total / wins if wins else float("inf"),
                     "with_failures": (total + (len(usable) - wins) * failure_cost) / wins if wins else float("inf"),
                     "latency": secs / len(usable), "mix": mix})
    return rows


def show_scoreboard(saved: dict, items: list[dict], talks: list[dict], failure_cost: float) -> None:
    res = {(c["qid"], c["tier"]): c for c in saved["calls"] if not c["error"]}
    usable = [it for it in items if all((it["id"], t) in res for t in TIERS)]
    if not usable:
        console.print("[red]No question has answers from every tier yet.[/]")
        return

    console.print(f"\n[bold]Success by kind[/]  [dim]{len(usable)} questions[/]")
    console.print()
    kinds = ["lookup", "filter", "plan"]
    console.print("  " + f"{'':<16}" + "".join(f"{k:>9}" for k in kinds) + f"{'all':>9}")
    for t in TIERS:
        cells = []
        for k in kinds + [None]:
            sub = [it for it in usable if k is None or it["kind"] == k]
            wins = sum(grade(it, res[(it["id"], t)]["answer"] or {}, talks)[0] for it in sub)
            cells.append(f"{wins / len(sub):>8.0%} " if sub else f"{'-':>9}")
        console.print(f"  {MODELS[t].removeprefix('gemini-'):<16}" + "".join(cells))

    for style in ("template", "natural"):
        sub = [it for it in usable if it.get("style") == style]
        if sub:
            right = sum(routers.ifelse(it["question"])[0] == it["kind"] for it in sub)
            line = f"  [dim]{style}: if/else picked the intended kind for {right}/{len(sub)}"
            if LAYA:
                line += f", Laya {sum(LAYA[it['id']]['label'] == it['kind'] for it in sub)}/{len(sub)}"
            console.print(line + "[/]")

    for when, title in (("today", "today's prices"), ("jan", "January 2027 prices (Flash ×2)")):
        rows = score(saved, items, talks, PRICES[when], failure_cost)
        base = next(r for r in rows if r["name"] == "Always Pro")["with_failures"]
        console.print(f"\n[bold]Cost per successful task · {title}[/]\n")
        console.print(f"  {'strategy':<24}{'success':>8}{'tokens $':>10}{'$/success':>11}"
                      f"{'+failures':>11}{'vs Pro':>8}{'avg s':>7}   calls L/F/P")
        for r in rows:
            vs = "" if r["name"] == "Always Pro" or not r["wins"] else f"{r['with_failures'] / base - 1:+.0%}"
            m = r["mix"]
            console.print(
                f"  {r['name']:<24}{r['wins'] / r['n']:>8.0%}{'$' + format(r['total'], '.4f'):>10}"
                f"{'$' + format(r['per_success'], '.5f'):>11}{'$' + format(r['with_failures'], '.4f'):>11}"
                f"[bold]{vs:>8}[/]{r['latency']:>7.1f}   {m['lite']}/{m['flash']}/{m['pro']}"
            )
    console.print("\n  [dim]$/success = every call made (incl. escalations and cache storage) ÷ tasks solved"
                  f"\n  +failures = the same, plus ${failure_cost:.2f} per wrong answer (--failure-cost); vs Pro uses this"
                  "\n  Laya rows include its CPU time (latency and $) · * Perfect = knows each question's kind: an upper bound[/]\n")


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-kind", type=int, help="only the first N questions of each kind")
    ap.add_argument("--tiers", default=",".join(TIERS), help="comma-separated subset of lite,flash,pro")
    ap.add_argument("--workers", type=int, default=6, help="parallel calls per tier")
    ap.add_argument("--resume", type=Path, help="results file to add missing calls to")
    ap.add_argument("--style", choices=["all", "template", "natural"], default="all",
                    help="which questions to score (default all)")
    ap.add_argument("--failure-cost", type=float, default=1.00, help="$ per wrong answer (default 1.00)")
    ap.add_argument("--score", type=Path, help="print the scoreboard for a results file, no API calls")
    args = ap.parse_args()
    args.tiers = args.tiers.split(",")

    global LAYA
    if LAYA_LABELS.exists():
        LAYA = json.loads(LAYA_LABELS.read_text())["labels"]
    talks = load_talks()
    items = build()
    if args.per_kind:
        items = [it for k in ("lookup", "filter", "plan") for it in [i for i in items if i["kind"] == k][: args.per_kind]]

    if args.score:
        if schedule_fingerprint() != SNAPSHOT:
            warn_snapshot()
        saved = json.loads(args.score.read_text())
        scored = [it for it in items if it["id"] in {c["qid"] for c in saved["calls"]}]
        if args.style != "all":
            scored = [it for it in scored if it["style"] == args.style]
        show_scoreboard(saved, scored, talks, args.failure_cost)
        return

    if args.resume:
        out, saved = args.resume.resolve(), json.loads(args.resume.read_text())
    else:
        RESULTS.mkdir(exist_ok=True)
        out = RESULTS / f"eval_{len(items)}q_{datetime.now():%Y%m%d-%H%M}.json"
        saved = {"models": MODELS, "thinking": THINKING, "location": LOCATION, "calls": [], "storage": {}}

    saved = run(args, items, talks, out, saved)
    show_scoreboard(saved, items, talks, args.failure_cost)


if __name__ == "__main__":
    main()
