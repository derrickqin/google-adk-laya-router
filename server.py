"""Cloud Run: the DevFest Helper as an HTTP service, Laya in-process.

  GET  /      -> status
  POST /warm  -> create the per-tier schedule caches now (the client calls this at start-up)
  POST /ask   {"question": "..."} -> routing decision, attempts (tier, usage, latency, validity), answer

Private service: call it with an identity token (helper.py --remote does this).
Built by the Dockerfile; deployed with the command in README.md.
"""

import asyncio
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

import helper
import routers
from common import LOCATION, MODELS, PRICES, ROUTER_PER_SECOND, TIERS, call_cost
from eval_set import SCHEDULE, load_talks

ROUTER = os.environ.get("ROUTER", "laya")
TIER_MAP = routers.TIER_FOR if os.environ.get("TIERS", "3") == "3" else routers.TIER_FOR_2
CACHE_TTL_S = 3600
REFRESH_AFTER_S = 3000  # recreate caches before the 1-hour TTL runs out

STATE: dict = {}
LOCK = asyncio.Lock()  # one question at a time: the callbacks share per-question state (fine for a demo)


def refresh_caches() -> None:
    """(Re)create one schedule cache per tier, in place: the callbacks hold a reference to this dict."""
    from google.genai import types

    from run_eval import make_cache

    client = STATE["client"]
    for name in STATE.get("caches", {}).values():
        try:
            client.caches.delete(name=name)
        except Exception:
            pass
    fresh = {t: make_cache(client, types, t, STATE["schedule"]).name for t in TIERS}
    STATE["caches"].clear()
    STATE["caches"].update(fresh)
    STATE["cached_at"] = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from google import genai
    from google.adk.agents import LlmAgent
    from google.adk.runners import InMemoryRunner

    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
    os.environ["GOOGLE_CLOUD_LOCATION"] = LOCATION
    t0 = time.perf_counter()
    if ROUTER == "laya":
        routers.laya("warm-up")  # weights are baked into the image; this just loads them
    STATE.update(
        client=genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"], location=LOCATION),
        schedule=SCHEDULE.read_text(), talks=load_talks(), caches={}, cached_at=0.0,
        startup_s=time.perf_counter() - t0,
    )
    # Caches are created on the first request, so an idle service stores nothing.
    before, after = helper.make_callbacks(STATE["caches"], TIER_MAP, ROUTER)
    agent = LlmAgent(name="devfest_helper", model=MODELS["flash"], instruction="Answer the attendee's question.",
                     before_model_callback=before, after_model_callback=after)
    STATE["runner"] = InMemoryRunner(agent=agent, app_name="devfest_helper")
    yield
    for name in STATE["caches"].values():
        try:
            STATE["client"].caches.delete(name=name)
        except Exception:
            pass


app = FastAPI(title="DevFest Helper", lifespan=lifespan)


class Ask(BaseModel):
    question: str


@app.get("/")
def status() -> dict:
    return {"status": "ok", "router": ROUTER, "tiers": TIER_MAP, "models": MODELS,
            "startup_s": round(STATE.get("startup_s", 0.0), 1), "caches": bool(STATE.get("caches"))}


@app.post("/warm")
async def warm() -> dict:
    """Create the schedule caches now (~15 s), so the first real question doesn't pay for it."""
    async with LOCK:
        if not STATE["caches"] or time.time() - STATE["cached_at"] > REFRESH_AFTER_S:
            refresh_caches()
    return {"caches": list(STATE["caches"]), "age_s": round(time.time() - STATE["cached_at"])}


@app.post("/ask")
async def ask(body: Ask) -> dict:
    async with LOCK:
        if not STATE["caches"] or time.time() - STATE["cached_at"] > REFRESH_AFTER_S:
            refresh_caches()
        t0 = time.perf_counter()
        r = await helper.answer_one(STATE["runner"], body.question, STATE["talks"], escalate=True)
    cost = r["router_secs"] * ROUTER_PER_SECOND + sum(
        call_cost(a["usage"], PRICES["today"][a["tier"]]) for a in r["attempts"] if a["usage"])
    return {**r, "router": ROUTER, "cost": cost, "server_s": time.perf_counter() - t0}
