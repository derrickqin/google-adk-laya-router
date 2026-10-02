# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["google-adk>=2.10", "google-genai>=2.25", "laya>=0.3.22", "rich>=13"]
# ///
"""The DevFest Helper, an ADK agent whose before_model_callback asks Laya
which Gemini tier each request needs.

  question -> Laya (in-process) -> Flash-Lite | Flash | Pro -> answer -> validator
                                                                   fails? -> one tier up

Usage:
  uv run helper.py                          # the three stage questions
  uv run helper.py "When's lunch?" "..."    # your own questions
  uv run helper.py --tiers 2                # Flash/Pro only, no Flash-Lite
  uv run helper.py --router ifelse          # compare with the keyword router
  uv run helper.py --replay results/helper_<ts>.json   # offline backup
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console

import routers
from common import LOCATION, MODELS, PRICES, ROUTER_PER_SECOND, THINKING, TIERS, call_cost, validate
from eval_set import ANSWER_SCHEMA, SCHEDULE, SNAPSHOT, load_talks, schedule_fingerprint, warn_snapshot

HERE = Path(__file__).parent
RESULTS = HERE / "results"

STAGE_QUESTIONS = [
    'What time does "How to Succeed in a Hackathon" start?',  # Flash-Lite; same slot as this talk (S1 joke)
    "I'm new to AI, any beginner talks worth seeing?",  # Flash
    "I'm here till 3pm. Build my day: one talk per slot, GCP first, then AI, and nothing on Android.",  # Pro
]

console = Console()
logging.getLogger("google_genai.models").setLevel(logging.ERROR)  # ADK trips an AFC advisory we don't need on stage

# Per-question routing state, shared between the callbacks and the loop. One user, one question at a time.
TURN: dict = {}


def last_user_text(llm_request) -> str:
    for content in reversed(llm_request.contents or []):
        if content.role == "user":
            return " ".join(p.text for p in content.parts or [] if getattr(p, "text", None))
    return ""


def make_callbacks(caches: dict, tier_map: dict, router: str):
    from google.genai import types

    def before_model(callback_context, llm_request):
        """Pick the tier, then point the request at that tier's model, cache and thinking level."""
        question = last_user_text(llm_request)
        if TURN.get("force_tier"):
            tier = TURN["force_tier"]
        else:
            if router == "laya":
                label, conf, secs, _ = routers.laya(question)
            else:
                label, conf, secs = routers.ifelse(question)
            tier = tier_map[label]
            TURN.update(label=label, confidence=conf, router_secs=secs)
        TURN["tier"] = tier

        llm_request.model = MODELS[tier]
        cfg = llm_request.config
        cfg.system_instruction = None  # it lives in the cache, and Vertex refuses both
        cfg.cached_content = caches[tier]
        cfg.thinking_config = types.ThinkingConfig(thinking_level=THINKING[tier])
        cfg.response_mime_type = "application/json"
        cfg.response_json_schema = ANSWER_SCHEMA
        cfg.labels = {"demo": "routing", "tier": tier, "strategy": f"helper-{router}"}
        cfg.automatic_function_calling = types.AutomaticFunctionCallingConfig(disable=True)
        TURN["t0"] = time.perf_counter()
        return None

    def after_model(callback_context, llm_response):
        u = llm_response.usage_metadata
        if u:
            TURN["usage"] = {
                "prompt": u.prompt_token_count or 0, "cached": u.cached_content_token_count or 0,
                "output": u.candidates_token_count or 0, "thinking": u.thoughts_token_count or 0,
            }
        TURN["latency"] = time.perf_counter() - TURN["t0"]
        return None

    return before_model, after_model


async def ask(runner, question: str) -> str:
    from google.genai import types

    session = await runner.session_service.create_session(app_name=runner.app_name, user_id="stage")
    text = ""
    async for event in runner.run_async(user_id="stage", session_id=session.id,
                                        new_message=types.Content(role="user", parts=[types.Part(text=question)])):
        if event.is_final_response() and event.content and event.content.parts:
            text = "".join(p.text or "" for p in event.content.parts)
    return text


async def answer_one(runner, question: str, talks: list[dict], escalate: bool) -> dict:
    """Route, answer, validate; on a failed check, retry one tier up."""
    TURN.clear()
    attempts = []
    while True:
        text = await ask(runner, question)
        try:
            answer = json.loads(text)
        except Exception:
            answer = None
        ok, why = validate(answer, talks, TURN.get("label", "any"))
        attempts.append({"tier": TURN["tier"], "usage": TURN.get("usage", {}), "latency": TURN.get("latency", 0.0),
                         "valid": ok, "why": why, "answer": answer})
        tier = TURN["tier"]
        if ok or not escalate or tier == TIERS[-1]:
            break
        TURN["force_tier"] = TIERS[TIERS.index(tier) + 1]
    return {"question": question, "label": TURN.get("label"), "confidence": TURN.get("confidence"),
            "router_secs": TURN.get("router_secs", 0.0), "attempts": attempts}


def show(r: dict, talks: list[dict], router: str) -> float:
    """Print one question's decision and answer. Returns its cost."""
    by_id = {t["id"]: t for t in talks}
    console.print(f"\n[bold]› {r['question']}[/]")
    conf = f"{r['confidence']:.2f}" if r["confidence"] is not None else "-"
    console.print(f"  [cyan]{router}[/]: {r['label']} [dim](confidence {conf}, {r['router_secs'] * 1000:.0f} ms)[/]")
    total = r["router_secs"] * ROUTER_PER_SECOND
    for a in r["attempts"]:
        u = a["usage"] or {"prompt": 0, "cached": 0, "output": 0, "thinking": 0}
        cost = call_cost(u, PRICES["today"][a["tier"]])
        total += cost
        last = a is r["attempts"][-1]
        mark = "[green]✓ valid[/]" if a["valid"] else f"[red]✗ {a['why']}{'' if last else ' → escalate'}[/]"
        console.print(
            f"  → [bold]{MODELS[a['tier']].removeprefix('gemini-')}[/]  "
            f"{u['prompt']:,} in ({u['cached']:,} cached) · {u['output'] + u['thinking']:,} out · "
            f"${cost:.4f} · {a['latency']:.1f}s  {mark}"
        )
    ans = r["attempts"][-1]["answer"] or {}
    if ans.get("answer"):
        console.print(f"  [italic]{ans['answer']}[/]")
    ids = [i for i in ans.get("session_ids") or [] if i in by_id]
    if len(ids) > 1:
        for i in ids:
            t = by_id[i]
            console.print(f"    [dim]{t['start']}–{t['end']}[/]  {t['title'][:58]}  [dim]{t['room'].split(' (')[0]}[/]")
    return total


async def run_live(args, questions: list[str], talks: list[dict]) -> list[dict]:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        sys.exit("Set GOOGLE_CLOUD_PROJECT (and run: gcloud auth application-default login).")
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
    os.environ["GOOGLE_CLOUD_LOCATION"] = LOCATION

    from google import genai
    from google.adk.agents import LlmAgent
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    from run_eval import make_cache

    tier_map = routers.TIER_FOR if args.tiers == 3 else routers.TIER_FOR_2
    used = sorted(set(tier_map.values()) | ({"pro"} if args.escalate else set()), key=TIERS.index)
    if args.router == "laya":
        with console.status("Loading Laya…"):
            routers.laya("warm-up")

    client = genai.Client(vertexai=True, project=project, location=LOCATION)
    schedule = SCHEDULE.read_text()
    caches = {}
    try:
        with console.status("Caching the schedule for each tier…"):
            for tier in used:
                caches[tier] = make_cache(client, types, tier, schedule).name
        before, after = make_callbacks(caches, tier_map, args.router)
        agent = LlmAgent(
            name="devfest_helper",
            model=MODELS["flash"],  # overridden per request by before_model
            instruction="Answer the attendee's question.",
            before_model_callback=before,
            after_model_callback=after,
        )
        runner = InMemoryRunner(agent=agent, app_name="devfest_helper")
        results = []
        total = 0.0
        if args.pause:
            console.print("[dim]Ready.[/]")
        for n, q in enumerate(questions, 1):
            if args.pause:
                console.input(f"\n[dim]Enter to ask Q{n}…[/]")
            r = await answer_one(runner, q, talks, args.escalate)
            results.append(r)
            total += show(r, talks, args.router)
        console.print(f"\n[bold]{len(results)} questions · ${total:.4f}[/]")
        return results
    finally:
        for name in caches.values():
            client.caches.delete(name=name)


def run_remote(args, questions: list[str], talks: list[dict]) -> list[dict]:
    """Same stage flow, but every question goes to the Cloud Run service (private: identity token)."""
    import subprocess
    import urllib.request

    token = subprocess.run(["gcloud", "auth", "print-identity-token"], capture_output=True, text=True,
                           check=True).stdout.strip()
    base = args.remote.rstrip("/")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    with console.status("Warming up Cloud Run (Laya + schedule caches)…"):
        urllib.request.urlopen(urllib.request.Request(base + "/warm", data=b"{}", method="POST", headers=headers),
                               timeout=300).read()
    if args.pause:
        console.print("[dim]Ready.[/]")
    url = base + "/ask"
    results, total = [], 0.0
    for n, q in enumerate(questions, 1):
        if args.pause:
            console.input(f"\n[dim]Enter to ask Q{n}…[/]")
        req = urllib.request.Request(url, data=json.dumps({"question": q}).encode(), method="POST", headers=headers)
        t0 = time.perf_counter()
        with console.status("Asking Cloud Run…"), urllib.request.urlopen(req, timeout=180) as resp:
            r = json.loads(resp.read())
        results.append(r)
        total += show(r, talks, f"{r['router']} @ Cloud Run")
        console.print(f"  [dim]round trip {time.perf_counter() - t0:.1f}s[/]")
    console.print(f"\n[bold]{len(results)} questions · ${total:.4f}[/]")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("questions", nargs="*", help="questions to ask (default: the three stage questions)")
    ap.add_argument("--router", choices=["laya", "ifelse"], default="laya")
    ap.add_argument("--tiers", type=int, choices=[2, 3], default=3, help="3 = Flash-Lite/Flash/Pro, 2 = Flash/Pro")
    ap.add_argument("--no-escalate", dest="escalate", action="store_false", help="don't retry a tier up on a failed check")
    ap.add_argument("--pause", action="store_true", help="start up, then wait for Enter before each question (stage mode)")
    ap.add_argument("--replay", type=Path, help="print a saved run, no API calls")
    ap.add_argument("--remote", help="Cloud Run URL of server.py: ask it instead of running locally")
    args = ap.parse_args()
    talks = load_talks()

    if args.replay:
        if schedule_fingerprint() != SNAPSHOT:
            warn_snapshot()
        saved = json.loads(args.replay.read_text())
        total = sum(show(r, talks, saved["router"]) for r in saved["results"])
        console.print(f"\n[bold]{len(saved['results'])} questions · ${total:.4f}[/]  [dim](replay)[/]")
        return

    questions = args.questions or STAGE_QUESTIONS
    results = run_remote(args, questions, talks) if args.remote else asyncio.run(run_live(args, questions, talks))
    RESULTS.mkdir(exist_ok=True)
    where = "remote" if args.remote else f"{args.router}_{args.tiers}tier"
    out = RESULTS / f"helper_{where}_{datetime.now():%Y%m%d-%H%M}.json"
    out.write_text(json.dumps({"router": args.router + (" @ Cloud Run" if args.remote else ""), "tiers": args.tiers,
                               "results": results}, indent=2))
    console.print(f"[dim]Saved {out.relative_to(HERE)} (replay with --replay)[/]")


if __name__ == "__main__":
    main()
