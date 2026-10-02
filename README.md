# google-adk-laya-router: a Gemini model router in a few lines, on a CPU-only container

This project shows how easy it is to use [Laya](https://pypi.org/project/laya/) as a model router. Laya is a small open model (421M parameters) that answers typed questions about a prompt ("is this a day plan?", yes/no) in one local pass. Here an [ADK](https://google.github.io/adk-docs/) agent asks it, in a `before_model_callback`, which Gemini tier each request needs: **3.5 Flash-Lite**, **3.8 Flash** or **3.1 Pro**. There's no routing service to call and no fine-tuning. The router is a pip package that runs in-process, and it needs **no GPU**, so it deploys to a CPU-only container on services like **Cloud Run**.

The idea came from **Jev**, TypeSafe's hosted routing classifier: a cheap classifier decides, then the expensive model executes. But Jev runs on TypeSafe's API, outside Google Cloud, and every routing decision sends it a summary of the request. Google Cloud users may just want to keep their data within Google Cloud. The open alternative, OpenJev, is 27B parameters, needs an H100 and has a non-commercial license (CC BY-NC). Laya is the natural option: Apache-2.0, small enough for CPU, and it speaks the same Jev protocol (`laya-serve`). So it runs inside your own project, next to the agent, and prompts never leave Google Cloud to be routed.

Then the part that matters: **evals to make sure the router works**. Every question is answered by every tier, and every strategy is scored by **$ per successful task**, not $ per token.

The app is a "DevFest Helper" that answers questions about the [GDG Melbourne DevFest 2026](https://gdg-melbourne-devfest-2026.sessionize.com/) schedule. Built for the talk [*Token-Maxxing Is Not a Strategy: Getting More Value per Token on Google Cloud*](https://gdg-melbourne-devfest-2026.sessionize.com/session/1321310) (GDG DevFest Melbourne, 3 Oct 2026). The same router for coding agents, as a LiteLLM hook: [litellm-gemini-laya-router](https://github.com/derrickqin/litellm-gemini-laya-router) (experimental).

## How little code it takes

Simplified to two tiers (the real thing is `routers.py` + `helper.py`):

```python
from google.adk.agents import LlmAgent
from laya import Router

router = Router()  # downloads the ~800 MB checkpoint once; runs on CPU, or a GPU if there is one
QUESTIONS = {"day_plan": {"type": "noul",  # a yes/no question, answered as a probability
                          "instructions": "Is the attendee asking to organise their whole day?"}}

def before_model(callback_context, llm_request):
    question = llm_request.contents[-1].parts[0].text
    p_plan = router.predict(question, QUESTIONS)["answers"]["day_plan"]["noul"]
    llm_request.model = "gemini-3.1-pro-preview" if p_plan >= 0.3 else "gemini-3.8-flash"

root_agent = LlmAgent(name="devfest_helper", model="gemini-3.8-flash", before_model_callback=before_model)
```

The real version asks two questions in the same pass, so it can choose between three tiers. In that same callback it also sets the tier's schedule cache, thinking level, JSON schema and billing labels.

## Runs on CPU: Cloud Run, no GPU

`server.py` is the same agent as an HTTP service. The `Dockerfile` installs CPU-only PyTorch and bakes the Laya weights into the image. Measured on Cloud Run on 1 Oct 2026 (details and the deploy command in [Cloud Run](#cloud-run)):

| | |
|---|---|
| Instance | 4 vCPU / 8 GiB, gen2, no GPU |
| Laya decision | **~1.0 s** (~1.8 s at 2 vCPU; ~70 ms on a MacBook GPU) |
| Round trip, Flash-Lite / Flash | ~4–5 s: the Gemini call dominates, not the router |
| Router compute | ~$0.00003 per decision (estimate: a 1 vCPU / 2 GiB share at request-based prices, see `common.py`) |
| Cold start | ~30 s to load Laya. `--min-instances 1` removes it, but bills continuously |

## Does the router work? The evals

130 questions (+5) about the schedule, generated and graded by code, each answered once by every tier. The strategies (always-X, an if/else keyword router, Laya, escalation, a cascade) are then scored from those saved answers, so re-scoring is free.

**It works.** Laya 3-tier was **100% right on the 30 reworded questions, at −37% cost per successful task vs always-Pro**. It sent only 18 of 135 questions to Flash-Lite (17 of them right) and never missed a day plan.

What the evals taught me along the way:

1. **Measure before you route.** Without the eval I'd have shipped the wrong router: the keyword router looked perfect on templated questions (100/100) and fell to 21/30 on reworded ones.
2. **Flash-Lite is the trap.** It's the cheapest tier and it fails quietly: "what's on at 12:37", most filters, most day plans. One wrong answer costs more than every token you saved, so the score charges $1 per failure (`--failure-cost`). A 3-tier if/else router came out **+988% vs always-Pro** on the reworded questions, because it sent 19 of 30 to Flash-Lite.
3. **Escalation catches nothing when wrong answers look valid.** Escalation and the cascade only step up when a no-answer-key validator fails (bad JSON, unknown session or room, a clash). Flash-Lite's wrong answers were well-formed, so they passed and were never escalated. On the reworded questions the Flash-Lite-first cascade didn't escalate once (30 of 30 calls stayed on Flash-Lite) and finished at 57%, the same as always-Flash-Lite. Adding escalation to the routers changed nothing either.

To be fair to the baseline: on an app this simple, a two-tier if/else Flash/Pro router does as well (−41%). Laya earns its keep on real traffic, where the keywords aren't there.

## Results (1 Oct 2026, `results/eval_135q_20261001.json`, today's prices, $1 per failure)

| Strategy | All 135 | Reworded 30 | vs Pro (reworded) | vs Pro (reworded, January) |
|---|---|---|---|---|
| Always Flash | 98% | 97% | +95% | +114% |
| Always Pro | 100% | 100% | — | — |
| if/else 3-tier | 91% | 83% | +988% | +990% |
| **Laya 3-tier** | **99%** | **100%** | **−37%** | **−28%** |
| if/else Flash/Pro | 99% | 100% | −41% | −28% |
| Laya Flash/Pro | 99% | 100% | −36% | −25% |

Re-score it yourself for free: `uv run run_eval.py --score results/eval_135q_20261001.json --style natural`.

## Costs

| What | Cost |
|---|---|
| `eval_set.py`, `label_laya.py`, `--score`, `--replay` | Free (local only) |
| Smoke test (`--per-kind 1`, 3 questions × 3 tiers) | ~$0.11 |
| **Full eval** (130 questions × 3 tiers) | **~$3.30** |
| Live helper, 3 stage questions | ~$0.03 |
| Cloud Run with `--min-instances 0` | Nothing while idle |
| Cloud Run with **`--min-instances 1`** | **Bills continuously** (4 vCPU / 8 GiB) until you set it back to 0 |

Gemini calls bill to your own Google Cloud project, on Vertex AI.

## Quick start

You need [uv](https://docs.astral.sh/uv/), the gcloud CLI, and a Google Cloud project with Vertex AI enabled.

```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=<your-project-id>

uv run eval_set.py --show 3               # build questions, self-test graders (free)
uv run label_laya.py                      # Laya labels, local (free; first run downloads ~800 MB)
uv run run_eval.py --per-kind 1           # smoke test: 3 questions x 3 tiers (~$0.11)
uv run run_eval.py                        # full run: 130 questions x 3 tiers (~$3.30)
uv run run_eval.py --resume <file>        # finish an interrupted run
uv run run_eval.py --score <file>         # scoreboard only, no API calls
uv run run_eval.py --score <file> --style natural --failure-cost 0.5

uv run helper.py                          # LIVE: the 3 stage questions (~$0.03)
uv run helper.py "your question" --pause  # your own questions
uv run helper.py --replay results/stage-backup.json   # replay a saved run, no API calls
```

Run from the repo root. Each script declares its own dependencies, which `uv run` installs.

### The schedule

`data/schedule.json` is a snapshot of the [Sessionize public schedule API](https://gdg-melbourne-devfest-2026.sessionize.com/api/schedule) taken on 1 Oct 2026, the one `results/` was made with. It's trimmed to sessions (with their abstracts), rooms, topics and speaker names: speakers' bios, photos, links and slide links are removed.

Questions and answer keys are generated from the schedule, so the saved `results/` only re-score correctly against this snapshot. The live schedule has changed since: talks have moved rooms and times. To try it, delete `data/schedule.json`. The next script downloads the live version, warns that it differs from the snapshot, and from then on only a fresh `run_eval.py` + `label_laya.py` gives valid scores.

The saved runs had the full schedule (with bios) in the model's context, so fresh runs on the trimmed snapshot use fewer input tokens and cost slightly less than `results/` shows.

## Files

| File | What it does |
|---|---|
| `data/schedule.json` | The 1 Oct 2026 schedule snapshot, trimmed (see above). |
| `eval_set.py` | Generates questions from `data/schedule.json` and grades answers by code. Also writes `eval_set.json` for inspection. |
| `paraphrases.json` | 30 questions reworded the way attendees type ("What GCP sessions can I hit before lunch?"). Same answer keys. Hand-checked; edit freely. |
| `make_paraphrases.py` | One-off generator for `paraphrases.json` (3.8 Flash, < 1 cent). Re-running overwrites the checked file. |
| `common.py` | Tiers, models, thinking levels, prices (today and January 2027), and the answer validator. |
| `routers.py` | `ifelse` (keywords + length) and `laya` (one Laya pass, two typed questions). Tier maps for 3 tiers and Flash/Pro. |
| `label_laya.py` | Runs Laya over the eval set once (~85 ms each on a MacBook GPU) and caches labels in `results/laya_labels.json`, so scoring needs no PyTorch. |
| `run_eval.py` | Runs every question on every tier and prints the scoreboard. |
| `helper.py` | The live DevFest Helper: an ADK agent whose `before_model_callback` runs Laya, then sets the model, the tier's schedule cache, thinking level, JSON schema and billing labels for that one request. A failed validator check retries one tier up. |
| `server.py` | The same helper as an HTTP service for Cloud Run (`POST /ask`, `POST /warm`), Laya in-process. Built by the `Dockerfile`. |
| `requirements.txt` | The service's dependencies (PyTorch CPU is installed separately in the Dockerfile). |
| `web/devfest_helper/` | The same router as an ADK agent package for the dev UI (`adk web`): plain-text answers plus a routing line. |
| `results/` | Raw results: every answer, token count and latency. Re-score any time. |

## The eval set (130 questions)

| Kind | Template | Reworded | Graded by | Intended tier |
|---|---|---|---|---|
| lookup | 40 | 10 | Exact room, start time or session ID | Flash-Lite |
| filter | 35 | 10 | Exact set of session IDs | Flash |
| plan | 25 | 10 | Checker: clashes, time window, rooms, must-see, every slot filled, best preferred talk per slot | Pro |

Plus 5 break questions (`s01`–`s05`, "When's lunch?"), so 135 in the full results file.

- One answer shape for every question (`session_ids`, `room`, `time`, `answer`), because the live helper doesn't know the kind in advance.
- "Before lunch" = ends by 12:55, "after lunch" = starts 13:40 or later, and durations come from the scheduled times. These rules are in the system prompt.
- The self-test checks that every reference answer passes and every wrong one fails.

## Scoring

- **tokens $**: every call a strategy makes, including escalations and a share of cache storage.
- **$/success**: tokens $ ÷ tasks solved.
- **+failures**: the same, plus `--failure-cost` (default $1) for each wrong answer, like the Zartis formula. "vs Pro" uses this column. Without it, a strategy that is cheap and wrong looks like the winner.
- **Escalation / cascade** step up a tier only when the **validator** fails: not JSON, empty, unknown session or room, or a plan with a clash or out of order. It has no answer key, so a plausible but wrong answer isn't escalated, which is the same limit a live system has.
- The **perfect router** knows each question's kind. It's the ceiling for routing, not a real router.

## Laya

One pass, two typed questions (`routers.LAYA_QUESTIONS`):
- `day_plan` (yes/no): "organise their whole day, with rules…?" If p ≥ 0.3 → plan → Pro.
- `task` (choice): quick fact / search / itinerary → Flash-Lite / Flash (/ Pro).

Wording was picked by measuring 3 choice wordings, 3 yes/no wordings and 2 off-topic wordings on this eval set, so it's tuned to it: a mild overfit worth admitting. Zero-shot Laya can't do an off-topic guard (it rejects 75–90% of real questions), which is a job for fine-tuning.

Only the yes/no question can send a request to Pro. Zero-shot, the choice question answers "itinerary" for most short questions ("When's lunch?"), so below the threshold that's treated as "not sure" and goes to Flash. This was found by the live helper, and it's why the 5 break questions were added.

## Live helper (stage run, 1 Oct)

| Question | Laya | Model | $ | Time |
|---|---|---|---|---|
| What time does "How to Succeed in a Hackathon" start? | lookup 0.51 | 3.5 Flash-Lite | $0.0011 | 3.5 s |
| I'm new to AI, any beginner talks worth seeing? | filter 0.41 | 3.8 Flash | $0.0028 | 5.4 s |
| I'm here till 3pm. Build my day: one talk per slot, GCP first, then AI, and nothing on Android. | plan 0.80 | 3.1 Pro | $0.0243 | 23.5 s |

The plan matched the reference planner exactly. Replay it with `uv run helper.py --replay results/stage-backup.json`.

## ADK dev UI (`adk web`)

A chat UI for asking the helper anything, locally. Same Laya router, tiers and per-model schedule caches as `helper.py`, but answers in plain text, and each reply ends with a routing line:

```
laya: plan (0.80, 98 ms) → 3.1-pro-preview · 29,168 of 29,221 tokens cached · $0.0341 · 20.1 s
```

```bash
export GOOGLE_CLOUD_PROJECT=<your-project-id>
uv run --python 3.13 --with 'google-adk>=2.10' --with 'laya>=0.3.22' --with rich adk web --port 8090 web
# open http://localhost:8090/dev-ui/?app=devfest_helper
```

- The agent package is `web/devfest_helper/` (`agent.py` exposes `root_agent`). Laya loads when `adk web` starts; the **first question** also creates the three schedule caches (~15–20 s). After that, Flash-Lite answers in ~3 s and Pro in ~20 s.
- Follow-ups work: the chat history goes to the model, and each turn is routed on its own (tested: a day plan → Pro, then "Which of those are in the Auditorium?" → Flash).
- No escalation in the UI (the stage helper keeps it). Calls are labelled `strategy=adk-web` in billing.
- The dev UI is unauthenticated and for local use only. On first open it asks about ADK usage telemetry: your choice.

## Cloud Run

One private service, `devfest-helper`. The Laya weights and the schedule are baked into the image, and it calls Gemini on `global`.

```bash
export GOOGLE_CLOUD_PROJECT=<your-project-id>
REGION=australia-southeast1   # anywhere; Gemini is still called on global

gcloud run deploy devfest-helper --source . --project $GOOGLE_CLOUD_PROJECT --region $REGION \
  --no-allow-unauthenticated --cpu 4 --memory 8Gi --execution-environment gen2 --min-instances 0 --max-instances 2 \
  --concurrency 4 --timeout 300 --cpu-boost --labels demo=routing \
  --set-env-vars GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT,GOOGLE_GENAI_USE_VERTEXAI=TRUE,GOOGLE_CLOUD_LOCATION=global,ROUTER=laya,TIERS=3,OMP_NUM_THREADS=4,MKL_NUM_THREADS=4,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1

URL=$(gcloud run services describe devfest-helper --project $GOOGLE_CLOUD_PROJECT --region $REGION --format 'value(status.url)')
uv run helper.py --remote $URL --pause        # stage flow against Cloud Run

# Before a demo: keep one instance warm (Laya loaded). This BILLS CONTINUOUSLY, so set it back to 0 afterwards.
gcloud run services update devfest-helper --project $GOOGLE_CLOUD_PROJECT --region $REGION --min-instances 1
gcloud run services update devfest-helper --project $GOOGLE_CLOUD_PROJECT --region $REGION --min-instances 0
```

- **Private:** callers need `roles/run.invoker`. `--remote` sends your `gcloud auth print-identity-token`.
- **Sizing:** 4 vCPU / 8 GiB, gen2. The 421M-parameter model needs ~2 GB of RAM; the extra CPU is for Laya's latency.
- **Cold start (measured 1 Oct):** time to load Laya and run the warm-up decision went 42 s (2 vCPU) → 38 s (4 vCPU) → 35 s (+ `HF_HUB_OFFLINE`, no update checks) → **30 s** (+ gen2). The weights are already baked into the image. What's left is reading 1.7 GB and initialising the model. To remove it, use `--min-instances 1`; to shrink it, try the ONNX INT8 export (not tested).
- **Laya latency on Cloud Run (measured 1 Oct):** ~1.8 s per decision at 2 vCPU by default, ~1.4 s with `OMP_NUM_THREADS=2` (PyTorch otherwise starts a thread per host core), **~1.0 s at 4 vCPU** (deployed). A MacBook does it in ~70 ms on its GPU (`mps`, Laya's default on a Mac) or ~300 ms on its CPU. The round trip was ~4–5 s for Flash-Lite and Flash either way, because the Gemini call dominates.
- **Caches:** created on `POST /warm` or the first question, one per tier, refreshed after 50 minutes. An idle service stores nothing new; existing caches expire after an hour.
- **Cost:** with `--min-instances 0`, nothing while idle. With `--min-instances 1` it bills continuously.

## Demo-grade, not production

- **One question at a time.** `server.py` serialises requests behind an `asyncio.Lock`, because the callbacks share per-question state. For real traffic, move that state into the ADK session.
- **Cache creation blocks the event loop.** `server.py` creates the schedule caches synchronously inside that async lock (~15 s on first use, then every 50 minutes), so `GET /` stalls too while it runs. Fine for one user; move it to a thread (`asyncio.to_thread`) for anything more.
- Templated questions give keyword routers an easy ride (if/else: 100/100 templated, 21/30 reworded). Quote the reworded numbers when comparing routers.
- A 30-question reworded set is small. Treat differences of a few questions as noise.
- Prices are hard-coded in `common.py` (checked 30 Sep 2026). Only 3.7/3.8 Flash change in January.

## License

MIT. See [LICENSE](LICENSE). The schedule data in `data/` belongs to GDG Melbourne and its speakers, and isn't covered by this license.
