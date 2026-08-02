# Limitations — seeded Day 1, per plan §2.6

Every spec Part 7 bullet lives here from Day 1 as an owner-tagged stub, so known
disclosures never depend on anyone remembering them. Append new compromises the
day they happen. Fill each stub with final numbers during the write-up (Aug 6–7).

## Data
- [P1] Survivorship bias: universe picked from today's S&P 500 membership, tested backwards; comparison remains fair because every method sees the identical biased universe (Zhang's argument).
- [P1] Only 40 stocks — thin for estimating correlation matrices.
- [P2] Bloomberg fundamentals may be restated, not point-in-time; per-field notes in FIELDS_USED.md. (If the fallback fired: Track B ran on sector + price-derived columns only — state prominently.)
- [P2] Sparse characteristic columns dropped, remaining gaps median-imputed (vs Zhang dropping stocks).

## Method
- [P1] Simplified rolling z-score instead of Avellaneda & Lee's OU-derived s-score.
- [P1] Betas drift; stale betas make a supposedly market-neutral spread directional.
- [P1] k-means converges to local optima; cluster membership is unstable across windows (stability figure F4 quantifies it).
- [P2] Label parameters (2.0 trigger, 50% reversion, 5-day horizon) are choices; results may be sensitive to them.
- [P2] Estimating an ~18×18 characteristic correlation matrix from 40 stocks is statistically thin.
- [P3] Track C's diagonal shrinkage (λ=1e-3) is itself a modelling choice; not tuned.
- [P3] Exit fills at the same close that generated the exit signal (entry is t+1-lagged); disclosed convention matching the spec's E0 wording.
- [P2/P3] The label measures reversion within 5 days of the trigger; the trade runs from entry (trigger+1) — a one-day mismatch, disclosed.

## Evaluation
- [P3] Small trigger sample → wide bootstrap confidence intervals; shown, not hidden.
- [P3] i.i.d. bootstrap understates uncertainty (trades cluster in time/pairs/regimes); CIs are optimistic lower bounds on width.
- [P3] Transaction-cost model is stylized (flat bps per leg); real costs depend on size and liquidity.
- [P3] Short-sale constraints and borrow costs ignored.
- [P3] Equal-weight aggregation of concurrent trades, no capital constraint; identical across strategies, therefore fair, but stylized.
- [P3] Multiple testing: 12 grid cells → expect ~0.6 cells "significant" at 5% by chance; we emphasize cross-grid patterns and the pre-registered primary comparison (Track A × E1 vs E0). (If Track C was de-scoped: restate for 8 cells.)

## Scoped but not run
- Track D (autoencoder residuals, Krause & Calliess 2024): cut per spec 8.4 — one lit-review sentence; leakage-checklist item 12 = N/A.
- 12e error-analysis deep-dive: cut in the 4-day replan; coefficient comparison and calibration commentary carry 12e.
