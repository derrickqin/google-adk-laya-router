"""DevFest Helper for the ADK dev UI (`adk web`).

Same router as the stage helper (Laya, in-process -> Flash-Lite / Flash / Pro, schedule cached
per model), but answers in plain text, and each reply ends with a routing line, so the
routing is visible in the chat. No escalation here: the stage helper keeps that.

Run from the repo root:
  uv run --python 3.13 --with 'google-adk>=2.10' --with 'laya>=0.3.22' --with rich \
    adk web web
"""

import os
import sys
import time
from pathlib import Path

ROUTING = Path(__file__).resolve().parents[2]  # the repo root
sys.path.insert(0, str(ROUTING))

from google import genai  # noqa: E402
from google.adk.agents import LlmAgent  # noqa: E402
from google.genai import types  # noqa: E402

import routers  # noqa: E402
from common import LOCATION, MODELS, PRICES, ROUTER_PER_SECOND, THINKING, TIERS, call_cost  # noqa: E402
from eval_set import SCHEDULE  # noqa: E402
from run_eval import make_cache  # noqa: E402

os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
os.environ["GOOGLE_CLOUD_LOCATION"] = LOCATION
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
if not PROJECT:
    raise RuntimeError("Set GOOGLE_CLOUD_PROJECT (and run: gcloud auth application-default login).")

CACHES: dict = {}
CACHED_AT = 0.0
REFRESH_AFTER_S = 3000  # caches live an hour; recreate before then
TURN: dict = {}  # one question at a time in the dev UI
STYLE = "(Reply in plain text for a chat window: brief, a short list for several talks, no JSON, no session IDs.)"

routers.laya("warm-up")  # load Laya once, when adk web imports the agent


def _caches() -> dict:
    """One explicit schedule cache per tier, created on the first question."""
    global CACHED_AT
    if not CACHES or time.time() - CACHED_AT > REFRESH_AFTER_S:
        client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
        CACHES.update({t: make_cache(client, types, t, SCHEDULE.read_text()).name for t in TIERS})
        CACHED_AT = time.time()
    return CACHES


def _last_user_text(llm_request) -> str:
    for content in reversed(llm_request.contents or []):
        if content.role == "user":
            return " ".join(p.text for p in content.parts or [] if getattr(p, "text", None))
    return ""


def before_model(callback_context, llm_request):
    label, conf, secs, _ = routers.laya(_last_user_text(llm_request))
    tier = routers.TIER_FOR[label]
    TURN.clear()
    TURN.update(label=label, conf=conf, secs=secs, tier=tier)
    llm_request.model = MODELS[tier]
    cfg = llm_request.config
    cfg.system_instruction = None  # the schedule + rules live in the cache, and Vertex refuses both
    # so the chat-style instruction rides along with the question instead
    for content in reversed(llm_request.contents or []):
        if content.role == "user":
            content.parts.append(types.Part(text=STYLE))
            break
    cfg.cached_content = _caches()[tier]
    cfg.thinking_config = types.ThinkingConfig(thinking_level=THINKING[tier])
    cfg.labels = {"demo": "routing", "tier": tier, "strategy": "adk-web"}
    cfg.automatic_function_calling = types.AutomaticFunctionCallingConfig(disable=True)
    TURN["t0"] = time.perf_counter()  # after cache set-up, so the line shows the model call itself
    return None


def after_model(callback_context, llm_response):
    """Append a routing line to the final answer: tier, Laya's confidence, cost and time."""
    if llm_response.partial or not llm_response.content or not llm_response.content.parts or not TURN:
        return None
    u = llm_response.usage_metadata
    usage = {"prompt": getattr(u, "prompt_token_count", 0) or 0, "cached": getattr(u, "cached_content_token_count", 0) or 0,
             "output": getattr(u, "candidates_token_count", 0) or 0, "thinking": getattr(u, "thoughts_token_count", 0) or 0}
    cost = call_cost(usage, PRICES["today"][TURN["tier"]]) + TURN["secs"] * ROUTER_PER_SECOND
    secs = time.perf_counter() - TURN["t0"]
    line = (f"\n\n---\nlaya: {TURN['label']} ({TURN['conf']:.2f}, {TURN['secs'] * 1000:.0f} ms) → "
            f"{MODELS[TURN['tier']].removeprefix('gemini-')} · {usage['cached']:,} of {usage['prompt']:,} tokens cached · "
            f"${cost:.4f} · {secs:.1f} s")
    llm_response.content.parts.append(types.Part(text=line))
    return llm_response


root_agent = LlmAgent(
    name="devfest_helper",
    model=MODELS["flash"],  # overridden per request by before_model
    before_model_callback=before_model,
    after_model_callback=after_model,
)
