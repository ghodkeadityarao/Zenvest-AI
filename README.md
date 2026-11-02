# Zenvest AI: ETF recommender

Website: https://zenvest-ai.onrender.com/

#### (This website may take 2–3 minutes to load as it is hosted on a free server.)
---

## Overview

A web app that turns a goal written in plain English into five ETF recommendations from a dataset of 2,264 funds (`justetf_2.csv`, 129 columns of returns, volatility, fees and sector and country exposure).

1. **Understands the goal.** A language model extracts a profile, through Groq (`openai/gpt-oss-20b`, with `openai/gpt-oss-120b` as a fallback) or Gemini (`gemini-3.5-flash-lite`): risk level, horizon, distribution policy, fee limit, sectors, regions, minimum fund size and currency. Whatever the model returns is validated, so an unknown or out-of-range value falls back to a safe default.
2. **Ranks the funds.** The limits the user stated (fee, distribution, volatility for the risk level, fund size, currency) come first: funds that meet all of them rank above the rest. Within that group an XGBoost ranker orders the funds by how well they fit the profile.
3. **Explains the ranking.** SHAP values show the main factors behind each fund's position, and a chat panel answers finance questions using the recommended funds as context.

## What it has

- **Web app** (Flask, `templates/index.html`): goal box with example goals, profile tags, five fund cards with fee, return, volatility, size and SHAP factors, a "limits first" panel showing how many funds meet every limit, and a chat panel. A notice states that it is educational information, not financial advice.
- **Real numbers only:** funds with a missing volatility, 1-year return or fund size (232 of 2,264) rank last, so every figure shown on a card comes from the data, not from a filled-in value. Before this, 11 of 750 recommended slots in the evaluation showed a made-up median.
- **The data date is stated** on the page, in every response and in the chat prompt: the figures are from March 2023.
- **Limits first:** if fewer than five funds meet every limit, the rest are the closest matches and the page says so.
- **Currency counts.** It was extracted before but never used in scoring.
- **Ranker trained on 80 varied profiles** (181,120 rows), not on one profile.
- **No pickle:** the ranker is stored as native XGBoost JSON and the prior as CSV in `artifacts/`, and is rebuilt in about 3 seconds if missing.
- **Protection for the model key:** 10 requests per minute per address and 1,000 per day, invalid requests do not count, and behind a proxy (`RENDER` or `TRUST_PROXY=1`) the address the proxy added to `X-Forwarded-For` is used.
- **Safer chat:** goal and message length limits, only `user` and `assistant` turns are kept from the browser, and fund names and numbers in the model's context come from the dataset by ISIN, never from text the browser sent. Errors shown to visitors are generic.
- **Two model providers:** Groq or Gemini, chosen with `LLM_PROVIDER` or by whichever key is set (`GROQ_API_KEY` or `GEMINI_API_KEY`; Groq wins if both are set). A model Groq reports as retired is skipped, and a rate limit or busy server is retried after a short wait. A failed call keeps a key-free reason that the profile check prints.
- **No rule-based fallback:** without the key or the service, the app says the AI service is unavailable.

## Results

All numbers come from `evaluate.py` (output in `evaluation_results.json` and `evaluation_per_profile.csv`) and need no API key. 150 profiles were generated at random from a seed the ranker never saw; 140 of them have at least five funds that meet every limit. For each method the top five funds were checked against the profile's own limits.

| Method | Top-5 funds meeting all limits (95% interval) | Goals at low, medium, high risk in the right order (of 24) | Preferred-sector exposure of the top 5 | Preferred-region exposure of the top 5 |
|---|---|---|---|---|
| **This system** (limits first, then ranker) | **100%** | **21** | 0.290 | 0.338 |
| Original design (ranker trained on one profile, soft limits) | 74.7% (69.7 to 79.9) | 0 | 0.120 | 0.148 |
| Ranker without limits first | 71.0% (65.7 to 76.4) | 16 | 0.517 | 0.395 |
| Profile score only | 72.0% (66.7 to 77.3) | 18 | 0.544 | 0.405 |
| Simulated-investor prior only | 29.9% (26.6 to 33.1) | 0 | 0.072 | 0.072 |
| Largest funds | 16.7% (12.6 to 21.4) | 0 | 0.091 | 0.092 |
| Random | 7.1% (5.1 to 9.4) | 2 | 0.074 | 0.076 |
| Cheapest funds | 2.0% (0.6 to 3.7) | 0 | 0.000 | 0.000 |
| Highest return | 1.7% (0.6 to 3.3) | 0 | 0.027 | 0.000 |

The average preferred-sector exposure across all funds is 0.092 (75 profiles with a sector preference) and 0.090 for regions (42 profiles).

- **The original ranker ignored the profile.** Its top five had a median volatility of 1.1%, 1.4% and 1.4% for low, medium and high risk, so the same funds came back whatever the user asked. The new system gives 1.6%, 19.7% and 26.8%.
- **Why it was wrong:** the ranker was trained on one profile, and its training target gave 40% weight to a score that is the same for every user, which pulled every request toward the same ultra-short bond funds.
- **Putting limits first costs some tilt.** The sector and region exposure is lower than with the profile score alone (0.290 against 0.544), because the limits shrink the set of candidate funds, but it is still about three times the average.
- **The simulated-investor prior was switched off.** The original system blended in a collaborative score built from 500 simulated investors. On a separate validation seed (`python evaluate.py --ablation`, `ablation_cf_prior.json`), every positive weight lowered personalisation and nothing improved:

| Prior boost | Top-5 funds meeting all limits | Risk groups ordered (of 24) | Sector exposure | Region exposure |
|---|---|---|---|---|
| 0 (used) | 100% | 21 | 0.282 | 0.244 |
| 0.02 | 100% | 18 | 0.277 | 0.231 |
| 0.05 | 100% | 10 | 0.249 | 0.189 |
| 0.1 | 100% | 10 | 0.181 | 0.079 |
| 0.25 | 100% | 0 | 0.121 | 0.056 |

The prior's code (`collaborative_filter.py`) is kept, with the boost set to 0 in `recommender.py`.

**Profile extraction by the language model** (`eval_profiler.py`, 42 labelled goals in `profiler_goals.py`, result in `profiler_results.json`). The goals are 32 investment requests and 10 other texts, and only the fields a goal states are scored. With Gemini (`gemini-3.5-flash-lite`) all 42 goals came out fully correct and no call failed:

| Field | Correct |
|---|---|
| Risk level | 28 of 28 |
| Horizon | 21 of 21 |
| Distribution policy | 16 of 16 |
| Sectors | 11 of 11 |
| Regions | 7 of 7 |
| Fee limit | 4 of 4 |
| Currency | 3 of 3 |
| Minimum fund size | 2 of 2 |
| Investment request or not | 42 of 42 |

With 42 goals, a perfect score still leaves room: the 95% interval for "fully correct" starts at about 92%. The goals are plain, mostly one-sentence requests written by one person following the prompt's own rules, 7 of them reuse sentences from the original code's quick-test list, and 2 are close to the prompt's examples. So this shows the extraction works on clear phrasing, not on vague, contradictory or unusual requests. The Groq models were not scored, because the free tier rate-limited them.

## About the data

An audit of `justetf_2.csv` (2,264 funds, one row per ISIN):

- **It is from March 2023.** 2,142 funds are quoted on 21 March 2023, and the latest quote is the same day. Funds launched since then, and any performance since then, are missing, and the median 1-year return in it is −7.2% (the 12 months to March 2023).
- **Not all are ETFs:** 2,151 ETFs, 76 ETNs (including crypto notes) and 37 ETCs (such as gold). Only 5 of the 750 recommended slots in the evaluation were not ETFs (gold ETCs).
- **Gaps:** volatility is missing for 226 funds (10%), 1-year return for 228 (10%), 3-year return for 676 (30%) and size for 19. Funds missing volatility, 1-year return or size are ranked last. Missing 3-year returns are still filled with the median when scoring the medium horizon, and that value is never shown.
- **Extreme values were kept:** 12 funds with a 3-year return above 200% (crypto notes), 18 with a 1-year return below −60% and 10 with volatility above 100%.
- **Not checked:** the figures were not compared with the source, and the source's terms of use were not reviewed.

## Limits

- **The 100% is by construction.** Funds that meet the limits are ranked first, so for any profile with at least five such funds the top five all meet them. The table is useful for the comparison with the other methods and for the personalisation columns, not as a quality score.
- **The checks use the system's own definitions.** The volatility limit for each risk level (10%, 20%, 50%) is the same mapping the ranker uses. There are no human relevance judgements and no backtest, so none of this says that the recommended funds are good investments.
- **Profiles are random, not real users.** The 150 profiles come from a generator I wrote. The first evaluation seed was discarded after I used it to diagnose the problem, and the prior was chosen on a different seed, so the final seed was not used for any choice.
- **The ranker is a stand-in for a formula.** It is trained to reproduce a hand-written profile score, so SHAP explains the ranker, not independent reasons. There is no real investor data anywhere.
- **The data is three years old** (see above), and returns shown are past returns. Some profiles are impossible: for example, only one USD fund is low-volatility, accumulating and under 0.3% in fees.
- **The chat uses the model** to decide whether a question is about finance and to answer it, so it can be wrong. It is limited to short, educational answers.
- The rate limit and daily budget are held in one process's memory, so a restart clears them and they need a single worker.
- The Flask routes were tested with a stand-in ranker and stand-in model calls. The real ranker and the Groq calls were checked separately, and no live model call was made in this work.

## Repository

| Path | Contents |
|---|---|
| `app_flask.py` | Flask app: page, `/api/recommend`, `/api/chat`, limits, validation, headers |
| `llm_profiler.py` | Groq profile extraction, profile validation, chat call |
| `recommender.py` | Profile score, limits, XGBoost ranker, SHAP explanations |
| `engine.py` | Builds and loads the artifacts |
| `data_pipeline.py` | Loads and cleans the ETF data |
| `collaborative_filter.py` | Simulated-investor prior (switched off) |
| `limits.py` | Per-address rate limit and daily budget |
| `evaluate.py`, `evaluation_results.json`, `evaluation_per_profile.csv`, `ablation_cf_prior.json` | Held-out evaluation and the prior ablation |
| `eval_profiler.py`, `profiler_goals.py` | Profile-extraction check (needs a Groq or Gemini key) |
| `artifacts/` | Ranker (JSON), its feature names and the prior (CSV) |
| `templates/index.html` | The web page |
| `tests/` | 56 tests: 22 for the data, ranker, limits, evaluation code and scorer, 19 for the app routes, limits, input handling and error messages, and 15 for the model calls (both providers, error reporting, model fallback, reasoning settings, retries). The first two groups need different Python environments (XGBoost and Flask). |
