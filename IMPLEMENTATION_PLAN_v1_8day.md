# Implementation Plan — Predicting Temporary Stock-Price Gap Convergence

**CSC311 Summer 2026 — Option 2 Final Project**
**Team:** Alex Bachynsky, Golam Eram, Jaskaran Narula
**Deadline:** Friday, August 7, 2026, 11:00pm EST
**Plan horizon:** Day 1 = Friday July 31 → Day 8 = Friday August 7 (Days 2–3 fall on a weekend)
**Companion document:** `project_spec_v2.md` (the *what* and *why*; this plan is the *who*, *when*, and *in what order*)

---

## 0. How to use this plan

This plan turns the spec into three **independently workable workstreams** — one per person — connected by a small set of **frozen artifact contracts** (Section 2.2). The design goal is that no one ever sits idle waiting for someone else's code:

1. **Contracts first.** The interfaces between people are *data files with fixed schemas*, not function calls into each other's code. If your output matches the contract table, the next person's code consumes it — whether it was produced by the real pipeline or by a fixture.
2. **Fixtures from Day 1.** Two synthetic-data generators land on Day 1 (Section 2.4). Person C builds the entire labeling → features → models stack against planted-outcome fixtures days before real z-scores exist. Person B builds the backtest engine against fixture z-scores. The random-walk fixture is later reused *verbatim* as the Part 4 noise test — the parallelism scaffolding doubles as the leakage audit.
3. **Two real integration points, not continuous integration pain.** Day 4 (real Track A triggers + first real model fit) and Day 5 (full 2×4 grid on validation). Everything before those days is contract-checked, not integration-checked.
4. **Gates are honest.** Track C (Day 5 evening) and Track D (Day 6 morning) run only if their spec-defined gates pass. Default is NO-GO. The core 2×4 project is complete without them.
5. **The test set is touched exactly once**, Day 7 morning, screen-shared, from frozen models. The experiment runner physically guards the test split (Section 4, task B7).

**If you read only one thing before starting:** your own workstream section (3, 4, or 5), plus the artifact contract tables in 2.2.

### Contents

| § | Section |
|---|---|
| 1 | Team model and ownership map |
| 2 | Architecture, contracts, and engineering conventions (repo, artifact schemas, `config.py`, fixtures, tests, git, kickoff agenda) |
| 3 | Workstream A — Representation & Pairs (PCA → residuals → clusters → pairs → z-scores; Track C) |
| 4 | Workstream B — Data, Markets & Experiments (data, Track B, backtest engine, costs, controls, noise test) |
| 5 | Workstream C — Learning & Evaluation (triggers, features, E1–E3, splits, metrics; Track D) |
| 6 | Timeline, coordination, and risk (day-by-day, syncs, gates, report plan, risk register, cut ladder) |
| 7 | Appendix: tooling notes and practical gotchas (library pins, yfinance, Bloomberg, WSL2) |
| 8 | Appendix: spec → plan traceability matrix (every spec requirement → owner + day) |

### Reading order per person

| Person | Read first | Then | Skim |
|---|---|---|---|
| A — Representation & Pairs | §3 | §2.2, §2.5, §6.1 | §4 (engine you feed), §7 |
| B — Data, Markets & Experiments | §4 | §2.2, §2.4, §6.1 | §3 (pair-builder you call), §7.1, §7.6 |
| C — Learning & Evaluation | §5 | §2.2, §2.4, §6.1 | §4 (runner that calls your models), §7.5 |

---

## 1. Team model and ownership map

### 1.1 Roles

Real names map to roles at kickoff (Day 1) based on two constraints: **Person C** should be the strongest with scikit-learn/PyTorch (owns all four models and the gated autoencoder track); **Person B** must have Bloomberg terminal access (owns the Track B characteristics pull). The spec's Part 8 ownership notes (Track C → Person A, Track D → Person C) are preserved.

| | Person A | Person B | Person C |
|---|---|---|---|
| **Title** | Representation & Pairs | Data, Markets & Experiments | Learning & Evaluation |
| **Spec steps** | 2, 3, 4, 5, 6 | 0, 1, 4B, 9(E0), 10(strategy metrics), 11, 12a–c | 7, 8, 9(E1–E3), 10(splits + classification metrics) |
| **Core question owned** | "What does each stock do *on its own*, and who belongs with whom?" | "Is the data honest, and is the comparison fair?" | "Can a model learn which gaps close, and how do we measure that without fooling ourselves?" |
| **Gated extension** | Track C — partial-correlation distance (~5h) | — (owns both gates' *evidence*: grid + noise test) | Track D — autoencoder residuals (~12–15h) |
| **Part 4 audit items** | Checklist 1–4 (+11 if C runs) | Checklist 8–10 + noise test + manual trace | Checklist 5–7 (+12 if D runs) |

### 1.2 Module ownership (merge rights follow ownership)

| Modules | Owner |
|---|---|
| `src/factors/`, `src/clustering/`, `src/pairs/`, `src/spreads/`, `src/synth/make_synthetic.py`, `scripts/checks/`, `scripts/audit/` | A |
| `src/data/`, `src/backtest/`, `src/experiments/`, `src/contracts.py`, `Makefile` | B |
| `src/labeling/`, `src/features/`, `src/datasets/`, `src/models/`, `src/evaluation/`, `src/analysis/`, `src/synth/fixture_zscores.py` | C |
| `src/config.py`, `DECISIONS.md` | all three (change control, §2.3) |

### 1.3 The dependency spine (what actually blocks what)

Only three hand-offs on the critical path touch real data. Everything else runs on fixtures.

```
B: returns.parquet (Day 1 EOD)
   └─→ A: factors + residuals (Day 2 EOD)
          └─→ A: clusters → pairs_a → zscores_a (Day 3 EOD)
                 └─→ C: real triggers_a + first E1 fit (Day 4)  ← Integration Checkpoint 1
                        └─→ B: full 2×4 grid on validation (Day 5) ← gate evidence
B: pairs_b (Day 3 EOD) ──→ A: zscores_b (Day 4 AM) ──→ C: triggers_b (Day 4 PM)
```

Person B's Bloomberg pull (Day 2) and backtest engine (Days 2–4) sit entirely off this spine; Person C's whole stack develops off-spine on fixtures through Day 3.

### 1.4 Decisions already made (ratify at kickoff, then frozen)

These are the parameters and design choices this plan commits to, consolidated from the spec's stated options. They are written into `src/config.py` (§2.3) at kickoff; changing any of them afterwards requires all-three sign-off and a `DECISIONS.md` entry.

| # | Decision | Choice | Spec basis |
|---|---|---|---|
| K1 | Return type | Log returns | Step 1 ("either is defensible") |
| K2 | PCA component rule | Smallest m ∈ {3,4,5} with cum. variance ≥ 60% | Step 2e (a-priori rule) |
| K3 | Download window | 2015-01-01 → 2025-01-01 (test = 2023–2024); *alternative to discuss: extend to mid-2025 for a longer test window* | Step 0 code vs. text ambiguity |
| K4 | Track A cluster input | Factor-beta vectors (compressed option) | Step 4, Zhang's evidence |
| K5 | Recluster cadence | 21 trading days (A/C/D); quarterly (B) | Step 4 formation windows |
| K6 | Spread construction | Simple (no hedge ratio) | Step 6a ("less to go wrong") |
| K7 | Z-score window | 60 trading days, window-local | Step 6b |
| K8 | Label parameters | Trigger 2.0 · reversion 50% · horizon 5d | Step 7 (a-priori) |
| K9 | Execution | Enter t+1 close; exit \|z\|<0.5 or 5d; $1/leg; P&L from raw returns | Steps 9–11 |
| K10 | Cost grid | {0,5,10,15,20,30,40,50} bps/leg/transaction; headline c=10 | Step 11 |
| K11 | Splits + purge + embargo | 2015–20 / 2021–22 / 2023–24; purge 5d; embargo 10d | Step 10 |
| K12 | Primary comparison (pre-registered) | Track A × E1 vs Track A × E0 | Part 7 |
| K13 | Seeds | Global seed 311, every stochastic call seeded | Part 4 hygiene |

*Known, disclosed wrinkle on K8/K9: the label measures reversion within 5 days of the **trigger**, while the trade runs from **entry** (trigger + 1). The model predicts the event; the trade realizes it with a one-day lag. This is stated in the report's limitations, not hidden.*

---

## 2. Architecture, contracts, and engineering conventions

This section is the backbone every other section builds on: the repo layout, the authoritative artifact schemas, the frozen `src/config.py`, the fixtures that let all three people code in parallel from Day 1, the test harness, and the collaboration rules. Anything ambiguous elsewhere in the Plan resolves to what is written here.

### 2.1 Repository layout

```
pair_trading/
├── Makefile                  # all pipeline entry points (see 2.5) — owner: B
├── README.md                 # 1-page quickstart: venv, make data, make test — owner: B
├── DECISIONS.md              # append-only decision log (see 2.6) — owner: all
├── requirements.txt          # pinned deps (see 2.7; pins finalized per Section 7)
├── .gitignore                # results/ (except results/final/), venv/, __pycache__, .ipynb_checkpoints
├── src/
│   ├── config.py             # THE frozen config (2.3) — change control: all-3 + DECISIONS.md
│   ├── contracts.py          # artifact schema registry + validate_artifact() (2.2, 2.5) — owner: B
│   ├── data/                 # Step 0-1: yfinance download, cleaning, log returns — owner: B
│   ├── factors/              # Steps 2-3: rolling PCA, eigenportfolios, betas, OOS residuals — owner: A
│   ├── clustering/           # Step 4 + 4B.6: k-means, silhouette k-choice, co-membership/stability — owner: A (B calls it for track b)
│   ├── pairs/                # Step 5: the SHARED pair-builder (clusters -> pairs_{track}.csv) — owner: A; consumed by B for track b
│   ├── spreads/              # Step 6: spread + rolling z-score construction — owner: A
│   ├── labeling/             # Step 7: trigger detection, labels — owner: C
│   ├── features/             # Step 8: the 7 features, train-stat standardization — owner: C
│   ├── datasets/             # Steps 7-8 assembly: triggers_{track}.parquet builder — owner: C
│   ├── models/               # Step 9: E0/E1/E2/E3 behind one predict interface -> decisions table — owner: C
│   ├── evaluation/           # Step 10: splits/purge/embargo, metrics, calibration, bootstrap — owner: C
│   ├── analysis/             # 12e: base rate, coefficient comparison, error analysis — owner: C
│   ├── backtest/             # Step 11 + execution: engine, trade ledger, cost model — owner: B
│   ├── experiments/          # Step 12: run_grid, turnover-matched control, consensus pairs, noise test driver — owner: B
│   └── synth/                # fixtures (2.4): make_synthetic.py (owner: A), fixture_zscores.py + golden file (owner: C)
├── scripts/                  # runnable scripts: checks/ + audit/ (A), figures/ (all; one script per report figure)
├── tests/                    # pytest, mirrors src/ (2.5) — each module owner owns its tests
├── data/                     # committed artifacts (raw/, processed/, clusters/, pairs/, spreads/, datasets/, synth/)
├── results/                  # gitignored except results/final/ and results/figures/final/
├── docs/                     # leakage checklist copy, manual-trace worksheet, report figure list — owner: all
└── notebooks/                # EXPLORATION ONLY — owner: individual
```

**Notebooks policy (hard rule):** notebooks never contain pipeline logic and are never imported. Anything a notebook discovers gets promoted into a `src/` module with a test before any other person depends on it. Every artifact in `data/` and `results/` must be reproducible by a `make` target alone.

### 2.2 Artifact contracts

These tables are the authoritative interface reference; other Plan sections point here rather than redefining columns. All parquet files use a `pandas.DatetimeIndex` named `date` (trading days, strictly increasing, tz-naive) unless the schema is listed as *long* (then `date`/`window_end` is an ordinary column). All floats are `float64`; identifiers are `string`. Producers must call `contracts.validate_artifact(df, name)` before writing (see 2.5).

#### `data/raw/prices.parquet`, `data/raw/volume.parquet`, `data/raw/spy.parquet`
| column | dtype | meaning |
|---|---|---|
| *(index)* `date` | datetime64 | trading day |
| one col per ticker (40) | float64 | adjusted close (prices) / share volume (volume) |
| `SPY` (spy.parquet only) | float64 | SPY adjusted close |

Producer **B**, lands **Day 1**; refreshed never (frozen after download). Consumers: `src/data` (returns), `src/factors` (PC1-vs-SPY check), `src/features` (relative volume), `src/backtest` (raw-return P&L).

#### `data/raw/universe.csv` *(glue artifact)*
| column | dtype | meaning |
|---|---|---|
| `ticker` | string | one of the 40 tickers, post-cleaning survivors flagged |
| `sector` | string | one of the 4 sectors |
| `included` | bool | False if dropped by the >2%-missing rule (replacement noted in DECISIONS.md) |

Producer **B**, ratified at kickoff, lands **Day 1**. Consumers: everyone (canonical ticker list + ordering).

#### `data/processed/returns.parquet`
| column | dtype | meaning |
|---|---|---|
| *(index)* `date` | datetime64 | trading day |
| one col per ticker | float64 | daily **log** return `ln(P_t/P_{t-1})` |

Producer **B**, **Day 1 EOD**; never refreshed. Consumers: A (`factors`), C (nothing directly), B (`backtest` uses *raw* returns reconstructed from prices for P&L — the engine reads `prices.parquet`, not this file; see Section 4).

#### `data/processed/factors_a.parquet` / `pca_meta.parquet` / `loadings_a.parquet`
| file | column | dtype | meaning |
|---|---|---|---|
| factors_a | *(index)* `date`; `pc_1..pc_5` | float64 | daily eigenportfolio factor returns — **all 5 columns stored every day**; `pca_meta.n_components` records that day's kept m (storing all 5 avoids NaN design matrices downstream) |
| pca_meta | *(index)* `date`; `n_components` | int64 | m chosen that day |
| pca_meta | `cum_var_explained` | float64 | cumulative variance explained by the m kept components |
| loadings_a *(long)* | `date`, `ticker`, `component` | datetime64/string/int64 | which window / stock / PC |
| loadings_a | `loading` | float64 | sign-fixed eigenvector element |
| loadings_a | `beta` | float64 | OLS beta of stock on that factor, trailing window |

Producer **A**, **Day 2 EOD**; refreshed only on bugfix. Consumers: A (`clustering` — Track A cluster input is the beta vectors; `spreads` indirectly), C (`features`: `f_mkt_vol_20d` from `pc_1`), Track D comparison.

#### `data/processed/residuals_a.parquet` (and `residuals_d.parquet` if Track D runs)
Same shape/schema as `returns.parquet`; each cell is the **out-of-sample** residual return at *t* using betas estimated on the trailing window excluding *t* (betas regress on factor returns **reconstructed from the current window's eigenvectors** — see A2 — so residuals begin ~253 trading days after `DOWNLOAD_START`, i.e. early 2015). Producer **A** (**Day 2 EOD**); Track D matrix produced by C, integrated by A (Day 6). Consumers: A (`spreads`), C (`features`: `f_resid_mom_5d`).

#### `data/raw/characteristics_raw.parquet` / `data/processed/characteristics_clean.parquet` *(glue artifacts, Track B)*
| column | dtype | meaning |
|---|---|---|
| `quarter_end` | datetime64 | quarter the snapshot applies from |
| `ticker` | string | stock |
| one col per field | float64 | raw: 19 Bloomberg fields; clean: 18 cols (sentiment merged, sizes logged, z-scored per 4B.2) |

Producer **B**; raw lands **Day 2**, clean **Day 3**. Consumers: B (`4B.3` characteristics PCA, `clustering` for track b).

#### `data/clusters/labels_{track}.parquet` *(long)* and `stability_{track}.parquet` *(long)*
| file | column | dtype | meaning |
|---|---|---|---|
| labels | `window_end` | datetime64 | last day of the formation window |
| labels | `ticker` | string | stock |
| labels | `cluster_id` | int64 | k-means label (window-local; never compared across windows) |
| stability | `window_end` | datetime64 | as above |
| stability | `pair_id` | string | `AAA__BBB`, alphabetical |
| stability | `co_clustered` | bool | pair co-clustered in this window AND the previous window |

Producer **A** (tracks a/c/d, 21-trading-day cadence) and **B** (track b, quarterly), **Day 3**. Consumers: A/B (`pairs`), C (`features`: `f_cluster_stability`).

#### `data/pairs/pairs_{track}.csv`
| column | dtype | meaning |
|---|---|---|
| `pair_id` | string | `AAA__BBB`, tickers alphabetical — the universal join key |
| `stock_a`, `stock_b` | string | legs; `stock_a` < `stock_b` alphabetically |
| `group_id` | string | `{window_end:%Y%m%d}_{cluster}_{subgroup}` |
| `source` | string | `track_a` / `track_b` / `track_c` / `track_d` |
| `active_from`, `active_to` | date | window over which the pair is live (recluster to recluster) |

Producer: the shared pair-builder `src/pairs/build_pairs.py` — **canonical frozen signature (authoritative here; Sections 3/4 refer back to this)**:

```python
def build_pairs(labels_path: str,
                features_by_window: dict[pd.Timestamp, pd.DataFrame],  # per-window feature matrices -> within-cluster distances for the 5+ split
                source: str,                    # "track_a" | "track_b" | "track_c" | "track_d"
                calendar: pd.DatetimeIndex,
                out_csv: str) -> pd.DataFrame
```

Owner **A**. Signature stub (raises `NotImplementedError`) committed at kickoff so B can import and mock it from Day 2; implementation tests-green and tagged `v1-frozen` by Day 3 AM; called by B for track b Day 3 PM. Lands **Day 3 EOD**. Consumers: A (`spreads`), B (`experiments/consensus`), C (trigger attribution).

#### `data/spreads/spreads_{track}.parquet` and `zscores_{track}.parquet`
Wide: *(index)* `date` × one float64 column per `pair_id`. Consecutive active windows for the same `pair_id` tile into **runs** (Section 3, A6): within a run the spread accumulates continuously across window boundaries, and each run starts with a 60-trading-day backward-looking **burn-in** ending at `active_from` — so z is valid from a run's first active day (`SPREAD_POLICY="carry_with_burnin"`, `SPREAD_WARMUP_DAYS=60` in config). z is NaN **only outside active windows**; burn-in dates are internal and never emitted. Producer **A**: track a **Day 3 EOD**, track b **Day 4 AM**. Consumers: C (`labeling`), B (`backtest` exit rule).

#### `data/datasets/triggers_{track}.parquet` — the central contract
| column | dtype | meaning |
|---|---|---|
| `trigger_id` | string | `{pair_id}__{trigger_date:%Y%m%d}` — unique key (double underscore, matching the `AAA__BBB` pair_id style; encoded in `validate_artifact`) |
| `pair_id`, `source` | string | pair + track provenance |
| `trigger_date` | datetime64 | day |z| first crossed 2.0 from below |
| `z_trigger` | float64 | **signed** z at trigger |
| `f_abs_z` | float64 | feature: \|z_trigger\| |
| `f_spread_vol_60d` | float64 | feature: std of daily spread *changes* (Δspread) over the trailing 60d at trigger (C3's exact formula; the deviation from spec Step 8's literal "std of spread" is disclosed in methods) |
| `f_resid_mom_5d` | float64 | feature: `sign(z_trigger) ×` 5d sum of (resid_A − resid_B) — positive = still widening (C3) |
| `f_mkt_vol_20d` | float64 | feature: rolling 20d std of `pc_1` |
| `f_rel_volume_20d` | float64 | feature: mean of both legs' volume ÷ their 20d averages |
| `f_days_since_trigger` | float64 | feature: trading days since this pair's previous trigger (capped/NaN-policy in Section 5) |
| `f_cluster_stability` | float64 | feature: `co_clustered` (0/1) from latest `stability_{track}` row ≤ trigger_date |
| `label` | int8 | 1 iff \|z\| ≤ 0.5·\|z_trigger\| within next 5 trading days |
| `horizon_end_date` | datetime64 | trigger_date + 5 trading days |
| `split` | string | `train` / `val` / `test` / `purged` / `embargo` |

Producer **C**; fixture version Day 2, **real track-a version Day 4 (Integration Checkpoint 1)**. Consumers: C (`models`, `evaluation`), B (`backtest`, control matching).

#### `results/decisions_{track}_{model}.parquet`
| column | dtype | meaning |
|---|---|---|
| `trigger_id` | string | FK into triggers table |
| `enter` | bool | E0: always True; E1–E3: `p_hat > TAU`; control: random matched draw |
| `p_hat` | float64 | model probability; NaN for E0 and the control |

`{model}` ∈ `e0, e1, e2, e3, ctrl-e1, ctrl-e2, ctrl-e3` (controls name the model they are matched to). Produced by C's `models.predict_to_decisions()` invoked from B's runner (`src/experiments/run_grid.py`); controls by B. Consumers: B (`backtest`).

#### `results/trades_{track}_{model}.parquet`
| column | dtype | meaning |
|---|---|---|
| `trigger_id`, `pair_id` | string | provenance |
| `entry_date` | datetime64 | close of trigger_date + 1 trading day |
| `exit_date` | datetime64 | first of \|z\|<0.5 or 5 trading days after entry |
| `exit_reason` | string | `reverted` / `timeout` |
| `days_held` | int64 | trading days entry→exit |
| `gross_ret` | float64 | P&L on $1/leg from **raw** stock returns |
| `net_ret_{c}bps` | float64 | one column per c in the cost grid (8 columns) |

Producer **B**'s engine, from Day 2 (fixtures) / Day 5 (real). Consumers: B/C (`evaluation`, metrics JSONs).

#### `results/metrics_{track}_{model}.json`
Nested JSON: `{"classification": {auc, precision_at_tau, recall_at_tau, brier, calibration_bins}, "strategy": {hit_rate, mean_ret_per_trade, cum_ret, sharpe, max_drawdown, n_trades, per_cost: {c: {...}}}, "ci": {...bootstrap 95% for each}, "meta": {seed, config_hash, git_sha, split}}`. Producer: C's `evaluation` called from B's runner. The `config_hash` + `git_sha` fields make every result file traceable to the exact code and config that produced it.

### 2.3 `src/config.py` (full draft)

```python
"""Frozen project configuration — CSC311 pairs-trading project.

CHANGE CONTROL: every value in this file was ratified at the Day 1 kickoff
(Fri 2026-07-31). No value may change without (a) explicit agreement of all
three team members and (b) an appended entry in DECISIONS.md (date, old ->
new, reason, initials of all three). Silent edits are treated as bugs.
Values marked PLACEHOLDER are set later by their named owner via the same
process and are the only permitted additions.
"""
from __future__ import annotations
from pathlib import Path

# ----------------------------------------------------------------- identity
SEED: int = 311                 # global; numpy, sklearn, torch, control draws
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# ----------------------------------------------------------------- Step 0-1
DOWNLOAD_START: str = "2014-01-01"  # one warm-up year so the first 252d PCA window closes in early 2015
ANALYSIS_START: str = "2015-01-01"  # the sample the report describes
DATA_END: str = "2025-01-01"    # kickoff decision D3 (see 2.8): keep, or extend
MAX_MISSING_FRAC: float = 0.02  # drop ticker if >2% of trading days missing
LOG_RETURNS: bool = True        # r_t = ln(P_t / P_{t-1})
N_TICKERS: int = 40             # 4 sectors x 10; list lives in data/raw/universe.csv

# ----------------------------------------------------------------- Steps 2-3 (Track A PCA)
PCA_WINDOW: int = 252           # trailing days, EXCLUDING day t (off-by-one = leakage bug)
N_COMPONENTS_CANDIDATES: tuple[int, ...] = (3, 4, 5)
VAR_EXPLAINED_TARGET: float = 0.60  # smallest m with cum var >= target, else max candidate
EIGENVECTOR_SIGN_RULE: str = "sum_of_loadings_positive"  # applied every window
BETA_WINDOW: int = 252          # OLS of returns on factors, same trailing window, applied OOS at t

# ----------------------------------------------------------------- Steps 4/4B (clustering)
RECLUSTER_EVERY: int = 21       # trading days, tracks a/c/d, trailing 252d formation window
TRACK_B_REFRESH: str = "quarterly"
KMEANS_N_INIT: int = 10         # k-means++ init; k by max silhouette, formation window ONLY
K_RANGE: dict[str, range] = {
    "a": range(8, 14), "b": range(10, 14), "c": range(8, 14), "d": range(8, 14),
}
# Track A cluster input: each stock's factor-beta vector from the formation window
# Pair rules (Zhang): drop singletons; 2-4 -> all pairs; 5+ -> greedy NN subgroups of 2-3
MAX_WHOLE_CLUSTER: int = 4
SUBGROUP_SIZES: tuple[int, int] = (2, 3)

# ----------------------------------------------------------------- Step 6 (spread / z)
SPREAD_KIND: str = "simple"     # spread_t = cumsum over the pair's RUN of (resid_A - resid_B)
Z_WINDOW: int = 60              # trailing rolling mean/std, window-local
SPREAD_POLICY: str = "carry_with_burnin"  # runs tile across recluster windows; see Plan A6
SPREAD_WARMUP_DAYS: int = 60    # backward-looking burn-in before each run's active_from

# ----------------------------------------------------------------- Step 7 (triggers / labels)
TRIGGER_Z: float = 2.0          # onset only: prev |z| < 2.0, today >= 2.0
REVERSION_FRACTION: float = 0.5 # label 1 iff |z| <= 0.5*|z_trigger| ...
LABEL_HORIZON: int = 5          # ... within the next 5 trading days

# ----------------------------------------------------------------- Step 8 (feature windows)
SPREAD_VOL_WINDOW: int = 60
RESID_MOM_WINDOW: int = 5
MKT_VOL_WINDOW: int = 20
REL_VOLUME_WINDOW: int = 20
FEATURES: tuple[str, ...] = (
    "f_abs_z", "f_spread_vol_60d", "f_resid_mom_5d", "f_mkt_vol_20d",
    "f_rel_volume_20d", "f_days_since_trigger", "f_cluster_stability",
)

# ----------------------------------------------------------------- Step 10 (splits)
TRAIN_START, TRAIN_END = "2015-01-01", "2020-12-31"
VAL_START,   VAL_END   = "2021-01-01", "2022-12-31"
TEST_START,  TEST_END  = "2023-01-01", "2024-12-31"
PURGE_DAYS: int = 5             # drop last 5 trading days of labels before each boundary
EMBARGO_DAYS: int = 10          # further trading days dropped after each boundary

# ----------------------------------------------------------------- Step 9 (models)
TAU: float | None = None        # PLACEHOLDER — set once by Person C via the ONE
TAU_RULE: str = "see-plan-sec-5" # pre-registered validation rule; never touched on test
E3_HIDDEN: tuple[int, ...] = (16,)   # small MLP; details in Person C's section

# ----------------------------------------------------------------- Step 11 (execution / costs)
ENTRY_LAG_DAYS: int = 1         # enter at close of trigger_date + 1 trading day
EXIT_Z: float = 0.5             # exit when |z| < 0.5, or ...
MAX_HOLD_DAYS: int = 5          # ... 5 trading days after entry, whichever first
NOTIONAL_PER_LEG: float = 1.0   # long lagging / short leading leg by sign of z
COST_GRID_BPS: tuple[int, ...] = (0, 5, 10, 15, 20, 30, 40, 50)  # per leg per transaction
HEADLINE_COST_BPS: int = 10

# ----------------------------------------------------------------- Part 8 (optional tracks)
PARTIAL_CORR_SHRINKAGE: float = 1e-3   # Track C ridge; in-window only (leakage item 11)
AE_BOTTLENECK: str = "match_pca_m"     # Track D: bottleneck = that window's PCA m
AE_RETRAIN_EVERY: int = 21             # monthly retrain on trailing 252d

# ----------------------------------------------------------------- pre-registration
PRIMARY_COMPARISON: tuple[str, str] = ("track_a__e1", "track_a__e0")
```

**Task (B, Day 1, 0.5h):** commit this file verbatim after the kickoff read-aloud. Consumes: kickoff decisions. Produces: `src/config.py`. Done when: `python -c "import src.config"` passes and every teammate has confirmed the values match the freeze.

### 2.4 Fixtures — the parallelism enablers (both land Day 1)

#### `src/synth/make_synthetic.py` — pure-noise prices (also the Day 5 noise-test input)
**Task (A, Day 1, ~1.5h — A consumes it first, for A1 development, while B is busy with the real download; schemas fixed by 2.2).** Consumes: nothing (seeded RNG only). Produces: `data/synth/raw/prices.parquet`, `volume.parquet`, `spy.parquet` — **byte-for-byte the same schemas as the real raw artifacts**, which is what makes the Day 5 noise test trivial: point the whole pipeline at `data/synth/raw/` instead of `data/raw/` and change nothing else. Done when: schema contract tests pass on its outputs and two runs with the same seed are identical.

```python
def make_synthetic(
    n_tickers: int = 40,
    start: str = "2014-01-01",   # mirrors DOWNLOAD_START incl. the warm-up year
    end: str = "2025-01-01",
    seed: int = 311,
    out_dir: Path = Path("data/synth/raw"),
) -> None:
    """Random-walk prices with NO planted relationships.

    - Trading calendar: NYSE business days between start and end.
    - Tickers: SYN00..SYN39 (+ 'SPY' in spy.parquet, its own independent walk).
    - Log prices: p_t = p_{t-1} + eps, eps ~ N(0, sigma_i), sigma_i drawn once
      per ticker from U(0.008, 0.025); p_0 = ln(U(20, 500)). Returns are IID
      and cross-sectionally INDEPENDENT — any downstream 'signal' is a bug.
    - Volume: v_t = round(base_i * exp(N(0, 0.3))), base_i ~ logU(1e5, 1e7).
    - Single np.random.default_rng(seed); deterministic.
    """
```

CLI: `python -m src.synth.make_synthetic --seed 311 --out data/synth/raw` (argparse mirrors the signature). The `make noise-test` target (2.5) reruns it with the frozen seed before piping it through the full pipeline.

#### `src/synth/fixture_zscores.py` — planted-OU pairs + golden triggers
**Task (C, Day 1, ~2.5h — C consumes it throughout Days 1–3; the golden file is hand-derived from the exported z-matrix, never from the labeling code it will test).** Consumes: nothing. Produces, under `data/synth/fixture/`: `zscores_a.parquet` and `spreads_a.parquet` (date × 6 pair_ids), **`prices.parquet`**, `residuals_a.parquet` and `volume.parquet` for the 12 fake leg tickers (leg prices are seeded walks whose returns embed the planted residual differences — this is what lets B's engine compute P&L on the fixture, and B's 3-trade golden ledger is hand-computed from these prices), `factors_a.parquet` (a fake `pc_1` series for `f_mkt_vol_20d`), `pairs_a.csv` — **including one pair whose rows tile across two consecutive active windows**, exercising the run/burn-in convention and the cross-boundary trigger case — `stability_a.parquet`, and **`tests/golden/golden_triggers.csv`** (`trigger_id, pair_id, trigger_date, z_trigger, label, horizon_end_date`). The calendar is pinned (start 2019-07-01, 500 trading days, through ~mid-2021) so it **spans the train/val boundary**; a contract test asserts the fixture triggers table contains both train and val rows — otherwise C's `tune()` and τ dry-runs would silently exercise nothing. Done when: Person C's trigger/label code reproduces the golden file exactly (`make test` includes this as a contract test).

```python
def make_fixture(
    seed: int = 311,
    start: str = "2019-07-01",   # pinned so the 500-day span crosses the 2020-12-31 train/val boundary
    n_days: int = 500,
    out_dir: Path = Path("data/synth/fixture"),
) -> None:
    """Six fake pairs with KNOWN trigger/label outcomes.

    The z-score series are generated DIRECTLY (the fixture's z IS the file's
    z — no rolling recomputation), so threshold crossings are auditable by
    eye. Planted regimes, one per pair:
      P1-P3 'fast OU':  z_{t+1} = 0.55*z_t + N(0,0.4) with 4 injected
             excursions each to |z|>2  -> reverts within 5d: labels mostly 1
      P4-P5 'slow OU':  z_{t+1} = 0.97*z_t + N(0,0.3), excursions decay
             slower than the 5d horizon -> labels mostly 0
      P6    'break':    one excursion crosses 2.0 and RAMPS to 4 and stays
             -> exactly one trigger, label 0 (tests onset-only: no re-trigger
             while |z| stays above 2)
    Spreads are back-computed as z * (fixed sigma) + fixed mean; residuals as
    spread first-differences split across the two legs; volume/pc_1 are
    seeded noise so all 7 features are computable.
    """
```

**Golden-file derivation:** the golden CSV is **not** produced by the labeling code it will test. C exports the z-matrix to CSV and scans it manually (spreadsheet threshold-crossing pass: mark rows where prev |z|<2 and current |z|≥2, then check the next 5 rows against `0.5*|z_trigger|`), records every trigger and label by hand, and commits the result with a note in the file header naming who verified it. Expected count: ~15–20 triggers — small enough to hand-check in ~30 minutes, large enough to exercise both labels, both signs of z, and the onset-only rule.

### 2.5 Testing strategy

`tests/` mirrors `src/` one-to-one (`tests/test_factors.py` ↔ `src/factors/`, etc.), plus two shared helpers.

#### The future-perturbation leakage helper (`tests/leakage_utils.py`)
The single most valuable test pattern in the project, written once and reused:

```python
def assert_no_future_dependence(
    stage_fn: Callable[[pd.DataFrame], pd.DataFrame],
    base_input: pd.DataFrame,          # date-indexed input (returns, residuals, ...)
    cutoff: pd.Timestamp,              # perturb strictly AFTER this date
    rng: np.random.Generator,
    scale: float = 5.0,
) -> None:
    """Run stage_fn on base_input and on a copy whose rows after `cutoff`
    are replaced with large noise. Assert the two outputs are byte-identical
    (pd.testing.assert_frame_equal, check_exact=True) on all rows <= cutoff.
    Any difference means the stage read the future."""
```

Applied on fixture data (fast) to at least: **(1)** `factors.compute_factors` — factors/loadings/betas up to *t* unchanged when returns after *t* change (catches the PCA off-by-one); **(2)** `factors.compute_residuals` — same; **(3)** `spreads.compute_zscores` — z up to *t* unchanged (catches full-sample rolling stats); and, via C's section, the feature builder. Each is one ~10-line test calling the helper. (Owner: each stage's owner; the helper itself: A, Day 2 AM, 1h — A's stages are its first consumers.)

#### Schema contract tests (`src/contracts.py` + `tests/test_contracts.py`)
`src/contracts.py` holds one `ArtifactSchema(columns: dict[str, str], index: str | None, checks: list[Callable])` per artifact in 2.2, and `validate_artifact(df, name)` which asserts: exact column set, exact dtypes, index monotonic-increasing and unique, and per-artifact invariants (pair_id alphabetical and matching `stock_a`/`stock_b`; `split` ∈ the 5 allowed values; `label` ∈ {0,1}; every `net_ret_{c}bps` column present for the full cost grid). Producers call it before every write; `tests/test_contracts.py` runs it against the committed fixture artifacts as goldens. Done when: deliberately renaming a column in a fixture file fails the suite. (Owner: B, Day 1, 1.5h.)

#### Property tests worth having (owners as per module)
- **Sign-fix determinism** (A): for random symmetric matrices, `fix_signs(eigvecs)` output is invariant to pre-negating any eigenvector; every output column sums > 0.
- **Purge/embargo correctness** (C): no row with `split == "train"` has `horizon_end_date` past the train boundary; no `val`/`test` row's `trigger_date` falls inside an embargo window; `purged`/`embargo` rows are never consumed by fit or metrics code.
- **Onset-only triggers** (C): on the fixture, consecutive days with |z| ≥ 2 yield exactly one trigger (pair P6 covers this).
- **Decisions ⊆ triggers** (B): every `trigger_id` in a decisions file exists in the triggers table; E0 decisions have `enter` all-True and `p_hat` all-NaN.
- **Cost monotonicity** (B): for every trade, `net_ret_{c}bps` is strictly decreasing in c, and `net_ret_0bps == gross_ret`.
- **Determinism** (all): running any stage twice with `SEED=311` produces identical files.

#### Make targets (the only sanctioned pipeline entry points; owner B, Day 1–4)
| target | runs |
|---|---|
| `make data` | download + clean + `returns.parquet` + fixture generation (Steps 0–1, 2.4) |
| `make tracka` | factors → residuals → clusters/stability → pairs_a → spreads/zscores_a (Steps 2–6, track a) |
| `make trackb` | characteristics clean → char-PCA → clusters_b → pairs_b → spreads/zscores_b |
| `make dataset` | triggers_{track}.parquet for all available tracks (Steps 7–8 + splits) |
| `make grid` | full experiments: decisions → backtest → metrics for every (track, model) cell + controls; `SPLIT=trainval` default, `SPLIT=test` only for the Day 7 witnessed run |
| `make noise-test` | regenerates synthetic raw data, runs `data`→`tracka`→`dataset`→`grid` against `data/synth/raw/`, asserts no-signal criteria (Part 4; pass/fail printed) |
| `make test` | `pytest -q` — the merge gate |
| `make figures` | all report figures from committed artifacts into `results/figures/` (Agg backend) |

### 2.6 Git and collaboration conventions

- **Repo init Day 1** during kickoff: private GitHub repo, `main` protected (no direct pushes), all three added as admins. First commit = skeleton tree of 2.1 + `config.py` + `contracts.py` stubs + Makefile + the frozen `build_pairs` signature stub (raises `NotImplementedError`; B mocks against it from Day 2) + `docs/limitations.md` seeded with every spec Part 7 bullet as an owner-tagged stub (so the append-on-compromise mechanism only ever *adds* items — known disclosures never depend on memory).
- **Branches:** branch-per-person, prefixed `a/`, `b/`, `c/` (e.g. `a/rolling-pca`, `c/labeling`). Short-lived; rebase on `main` daily.
- **Module ownership = merge rights.** The owner listed in 2.1 is the only person who approves PRs touching that package. Cross-package PRs need each touched owner's approval. This replaces heavyweight review — approvals at the evening sync, in person.
- **Merge cadence:** merge to `main` at the **daily evening sync**, only after `make test` passes locally on the branch. Broken `main` is an all-stop: whoever broke it fixes or reverts before anything else merges.
- **`DECISIONS.md`:** append-only, one line per decision: `YYYY-MM-DD | decision (old -> new where applicable) | initials of all agreeing`. Required for: any `config.py` change, any contract/schema change, universe substitutions, gate outcomes (Track C/D), and the τ value when C sets it.
- **Data policy:** raw pulls **are committed** — 40 tickers × 10y of prices/volume is a few MB and the characteristics table is tiny — so all three machines reproduce byte-identical pipelines with no re-download drift. `data/` is therefore in git; a `data/raw/PROVENANCE.md` records pull timestamp and yfinance version.
- **Results policy:** `results/` is gitignored **except** `results/final/` (the Day 7 frozen test-run outputs: decisions, trades, metrics JSONs, with `git_sha` + `config_hash` embedded) and `results/figures/final/`. Nothing lands in `results/final/` before the witnessed Day 7 run.

### 2.7 Environment

Python **3.11+** in a project-local `venv` (`python3.11 -m venv venv && pip install -r requirements.txt`). `requirements.txt` packages (exact version pins are Section 7's call): `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `statsmodels`, `torch` (CPU build — `--index-url https://download.pytorch.org/whl/cpu`), `yfinance`, `matplotlib`, `pytest`, `jupyter`.

**WSL2 note:** no display server — every plotting module sets `matplotlib.use("Agg")` before any pyplot import (enforced by a two-line helper `src.evaluation.plotting.setup_mpl()` that also fixes figure DPI/size), and all figures are **saved** to `results/figures/`, never `plt.show()`n. `make figures` must run headless on all three machines.

### 2.8 Day 1 kickoff agenda (2h, all-hands, Fri Jul 31)

Scribe: Person C, capturing every ratified item into `DECISIONS.md` and `config.py` live.

| time | item |
|---|---|
| 0:00–0:10 | **D1 — Name→role mapping.** Person C = strongest sklearn/PyTorch (owns Track D if gated in); Person B = has Bloomberg terminal access; Person A = remainder. Ratify Alex/Golam/Jaskaran assignments explicitly. |
| 0:10–0:25 | **D2 — Universe.** Person B presents the proposed 40 tickers (4 sectors × 10, per Section 4). Ratify list + the substitution rule (if cleaning drops a ticker, B substitutes same-sector and logs it). Commit `universe.csv`. |
| 0:25–0:35 | **D3 — Download window.** START: `2014-01-01` (`DOWNLOAD_START`) — the warm-up year the first 252-day PCA window needs; the reported sample starts 2015. Confirm it. END: the spec's code says `end="2025-01-01"` while its text says "10 years 2015–2025." **Recommendation: keep `2025-01-01`**, giving test = 2023–2024, exactly matching the frozen split dates. Explicit alternative on the table: extend to mid-2025 for a longer test window — but that changes `TEST_END`, so it must be decided *now*, once, or never. Ratify both dates; log them. |
| 0:35–1:00 | **D4 — Config freeze.** Person B shares `src/config.py` on screen and reads **every value aloud** (Section 2.3). Objections resolved on the spot; then the file is committed and the change-control rule (all-3 + DECISIONS.md) is stated out loud. |
| 1:00–1:15 | **D5/D6 — Tooling + logistics.** Report in Overleaf (project created now, all invited); repo host GitHub private (created now, branch protection on); **daily sync time fixed** (propose 9pm EST, matching Section 6; incl. the weekend Days 2–3 — explicit availability check per person, with Day 3 (Sunday) flagged as the schedule's heaviest day: commit to the planned ~7.5–9h, not a 6h floor); Day 7 morning witnessed-test-run time fixed. |
| 1:15–1:30 | **Contracts walk-through.** 15 minutes on Section 2.2: each person names, from memory, every artifact they *produce* and every artifact they *consume*, including landing days. Any hesitation = re-read the table together. This is the cheapest integration insurance we can buy. |
| 1:30–1:50 | **Timeline + gates.** Walk the Day 1–8 anchors, Integration Checkpoint 1 (Day 4), the Track C gate (Day 5 evening) and Track D gate (Day 6 morning), and the never-cut list (purged CV, cost model, turnover-matched control, limitations). |
| 1:50–2:00 | **Start-of-work.** Confirm each person's first task: A — `make_synthetic` fixture, then the PCA engine against it; B — real download + cleaning (`make data`); C — `fixture_zscores` + hand-computed golden file, then trigger detection. Disperse. |

Done when: `DECISIONS.md` contains D1–D6, `config.py` and `universe.csv` are on `main`, and all three have the repo cloned, venv built, and `make test` green on the skeleton.

---

## 3. Workstream A - Representation & Pairs (Person A)

Person A owns the representation stack: everything from `returns.parquet` down to `zscores_{track}.parquet`. The workstream is a strict pipeline — A0 → A1 → A2 → A3 (Day 1–2), A4 → A5 → A6 (Day 3–4 AM), A7 (Day 4–5), A8 gated on Day 6. All config constants referenced below (`PCA_WINDOW`, `N_COMPONENTS` rule, `Z_WINDOW`, recluster cadence, k ranges, seed 311) live in `src/config.py` per Plan Section 2 and are imported, never re-declared.

Module layout owned by A:

```
src/
  factors/
    pca.py            # A1: rolling PCA -> factors_a, pca_meta, loadings (loadings part)
    residuals.py      # A2: rolling OLS -> residuals_a, betas (merged into loadings_a)
  clustering/
    kmeans.py         # A4: k-means wrapper + k selection
    comembership.py   # A4: co-membership + stability
    recluster.py      # A4: the rolling 21-day loop -> labels_{t}, stability_{t}
  pairs/
    build_pairs.py    # A5: SHARED pair builder (B imports this for track b)
  spreads/
    spread.py         # A6: spread + z-score -> spreads_{t}, zscores_{t}
  trackc/
    partial_corr.py   # A8: shrinkage, precision, partial-correlation distance
scripts/
  checks/
    check_scree.py check_pc1_loadings.py check_pc1_vs_spy.py check_var_over_time.py   # A3
  audit/
    audit_a_items_1_4.py   # A7
```

---

### A0 — Random-walk fixture generator (`src/synth/make_synthetic.py`)

**Goal.** The pure-noise price fixture specified in Section 2.4. A writes it because A1 consumes it within the hour (B is busy with the real download); on Day 5 the same script, same seed, becomes the noise-test input verbatim.

**Consumes:** nothing (seeded RNG; schemas fixed by the Section 2.2 contracts).
**Produces:** `data/synth/raw/prices.parquet`, `volume.parquet`, `spy.parquet`.

**Hours:** 1.5h, Day 1 (immediately after kickoff).
**Done when:** schema contract tests pass on its outputs; two same-seed runs are byte-identical; A1 develops against it the same afternoon.

---

### A1 — Rolling PCA engine (`src/factors/pca.py`)

**Goal.** Daily rolling-window PCA per spec Step 2, producing eigenportfolio factor returns and loadings for every day t from day 253 onward.

**Consumes:** `data/processed/returns.parquet` (fixture returns from `synth/make_synthetic.py` until real returns land Day 1 EOD).
**Produces:** `data/processed/factors_a.parquet`, `data/processed/pca_meta.parquet`, `data/processed/loadings_a.parquet` (loading column; `beta` column filled by A2).

**Signatures:**

```python
import numpy as np
import pandas as pd

def pca_one_window(window_returns: pd.DataFrame) -> dict:
    """One formation window (PCA_WINDOW x 40), EXCLUDING day t.
    Returns {"eigvals": np.ndarray, "eigvecs": np.ndarray (40 x m, sign-fixed),
             "weights": np.ndarray (40 x m, eigvec/sigma), "sigma": pd.Series,
             "n_components": int, "cum_var": float}."""

def choose_n_components(eigvals: np.ndarray) -> tuple[int, float]:
    """Config rule: smallest m in {3,4,5} with cumulative explained
    variance >= 0.60, else 5. Returns (m, cum_var_at_m)."""

def run_rolling_pca(returns: pd.DataFrame, out_dir: str = "data/processed") -> None:
    """Main loop over t in [PCA_WINDOW, len(returns)); writes the three parquets."""
```

**Algorithm notes (each item is a spec-flagged pitfall):**

1. Window slice is `returns.iloc[t - PCA_WINDOW : t]` — **excludes day t**. The off-by-one (`: t+1`) is the leakage bug named in the config freeze. Assert `window.index.max() < returns.index[t]` inside the loop.
2. Standardize with **window-local** `mu`/`sigma` (spec leakage trap #1). Never call `.mean()`/`.std()` on the full frame.
3. `C = np.corrcoef(Z.values, rowvar=False)` — correlation, not covariance, so volatile stocks don't dominate.
4. `np.linalg.eigh(C)`, then flip to **descending** order (`eigh` returns ascending — forgetting the flip silently gives you the noise components).
5. Sign fix every window: `if eigvecs[:, k].sum() < 0: eigvecs[:, k] *= -1`.
6. Eigenportfolio weights = `eigvecs[:, k] / sigma.values` (Avellaneda & Lee inverse-vol weighting). Factor return at t = `returns.iloc[t] @ weights[:, k]` — day-t returns with day-(t−1)-and-earlier weights, which is exactly what makes the factor return out-of-sample.
7. Compute cost: ~2,500 windows × `eigh` on a 40×40 matrix ≈ well under a minute total. **Daily estimation is fine; do not cache/approximate.** A plain Python loop is the correct engineering choice here.

**Output writing.** `factors_a.parquet`: index=date, columns `pc_1..pc_5` (all five stored every day; `pca_meta.n_components` records that day's kept m — storing all five avoids NaN design-matrix issues downstream). `pca_meta.parquet`: per date, `n_components`, `cum_var_explained`. `loadings_a.parquet` long format per the Section 2 contract (`date, ticker, component, loading, beta`), with `beta=NaN` until A2 fills it — writing the long frame is a single `melt` per day appended to a list, `pd.concat` once at the end (do not append to parquet per-day).

**Unit tests (`tests/test_pca.py`):** (a) hand-built 6×3 toy frame with a known dominant common factor → PC1 explains most variance, loadings all positive after sign fix; (b) eigenvalues descending; (c) sum of eigenvalues ≈ 40 (trace of correlation matrix); (d) window-exclusion assertion fires on a deliberately wrong slice; (e) `choose_n_components` on crafted eigenvalue vectors hits each branch of the {3,4,5,else-5} rule; (f) determinism: two runs byte-identical (no stochasticity in PCA, but locks the sign fix).

**Hours:** 5h (3.5h Day 1 against fixtures, 1.5h Day 2 rerun/validate on real returns).
**Done when:** all three parquets exist for real data, tests pass, and `factors_a` has a value for every trading day ≥ index 252.

---

### A2 — Out-of-sample residuals (`src/factors/residuals.py`)

**Goal.** Per-stock OLS betas on the trailing 252d window, applied out-of-sample at t (spec Step 3).

**Consumes:** `data/processed/returns.parquet`, `data/processed/factors_a.parquet`, `data/processed/pca_meta.parquet`.
**Produces:** `data/processed/residuals_a.parquet` (date × ticker, same shape as returns, NaN for the first 252+1 days); fills the `beta` column of `data/processed/loadings_a.parquet`.

**Signatures:**

```python
def fit_betas_one_day(factor_window: np.ndarray,   # 252 x m — reconstructed from THIS window's eigenvectors (see notes)
                      returns_window: np.ndarray,  # 252 x 40
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Single lstsq: X = [1 | factor_window] (252 x (m+1)), solve X B = returns_window
    for ALL 40 stocks at once. Returns (alphas (40,), betas (40 x m))."""

def run_rolling_residuals(returns: pd.DataFrame, factors: pd.DataFrame,
                          meta: pd.DataFrame, out_dir: str = "data/processed") -> None:
    """resid[t, i] = r[t, i] - alpha_i - beta_i @ f[t]  (betas from window ENDING t-1)."""
```

**Algorithm notes:**

- **Vectorized: one `np.linalg.lstsq` per day, not 40.** `lstsq(X, Y)` with `Y` = 252×40 solves all stocks simultaneously — the whole 2,500-day loop runs in seconds.
- Fit **with** an intercept column; residual at t subtracts both `alpha_i` and `beta_i @ f_t` so residuals are centered relative to the window fit. Record `RESIDUAL_INCLUDE_ALPHA = True` in `src/config.py` (one-line addition, note in DECISIONS.md at kickoff — it's a fill-in of the config, not a change).
- **Build the in-window factor matrix from the CURRENT window's eigenvectors, not by slicing the stored daily `factors_a` series** (this is Avellaneda & Lee's actual construction): `F = window_returns @ (eigvecs / sigma)`, a 252 × m matrix available for every day of the window. Regressing on stored daily factor returns would compound warm-ups — each stored factor return itself needs 252 prior days, so a full factor window would first exist ~504 days after data start, silently pushing the first residual into 2016 and losing a year of train triggers; it would also NaN-poison the design matrix wherever the stored m changed inside the window. With the in-window construction, residuals begin at day 253 of the data (early 2015, given the 2014 warm-up year). `factors_a` remains the artifact for `f_mkt_vol_20d` and the SPY check — it is simply not the beta-regression input. Read m per-day from `pca_meta`; never hardcode.
- Betas from the window ending t−1 applied to factor returns at t = the "train-then-apply" structure the spec calls the thing that keeps it honest. Assert the beta-fit window excludes t.
- Write per-day betas into `loadings_a.parquet` alongside the loadings (join on `date, ticker, component`).

**Unit tests:** (a) synthetic data generated as exact linear factor model + known noise → recovered betas within tolerance, residuals ≈ the injected noise; (b) shape/NaN mask of `residuals_a` matches returns; (c) residuals at t change if factor return at t changes but NOT if returns after t change (mini future-perturbation, previews A7); (d) per-window residual mean ≈ 0 in-sample but out-of-sample residual at t is computed from t's actual return.

**Hours:** 3h, Day 2.
**Done when:** `residuals_a.parquet` written on real data, `loadings_a.beta` populated, tests pass — Day 2 EOD hard deadline (C's real features and A4 both consume this).

---

### A3 — PCA sanity-check suite (`scripts/checks/`)

**Goal.** The four spec Step 2 checks, each a standalone script that writes one figure to `results/figures/` and prints a single `PASS`/`FAIL` line. These four figures go in the report.

**Consumes:** `factors_a.parquet`, `pca_meta.parquet`, `loadings_a.parquet`, `data/raw/spy.parquet` (interface with Person B — B guarantees SPY adjusted close aligned to the trading calendar, Day 1).
**Produces:** `results/figures/scree.png`, `pc1_loadings.png`, `pc1_vs_spy.png`, `var_explained_over_time.png` + printed pass/fail.

| Script | Figure | Pass criterion |
|---|---|---|
| `check_scree.py` | eigenvalue spectrum, median window + 3 sample dates | PC1 clearly dominant (λ₁/Σλ ≥ 0.25 on median window) |
| `check_pc1_loadings.py` | bar chart of PC1 loadings, median window | ≥ 38/40 loadings positive |
| `check_pc1_vs_spy.py` | scatter + rolling corr of daily PC1 factor return vs SPY log return | full-sample corr **> 0.9** (spec: "single best check") |
| `check_var_over_time.py` | top-3 cumulative variance across rolling windows, 2015–2025 | visible 2020 spike (crisis correlation); eyeball + printed max-vs-median ratio |

Each script: `python scripts/checks/check_X.py` → figure + one line, e.g. `PASS pc1_vs_spy corr=0.94`. Convert SPY prices to log returns inside the script (B ships prices, not returns, for SPY). A FAIL on any check blocks A4 — debug PCA before clustering garbage.

**Unit tests:** none beyond the scripts themselves (they *are* checks); smoke-test that each runs on fixture data without error.

**Hours:** 2h, Day 2 EOD deliverable.
**Done when:** four figures exist from real data, all four print PASS, links dropped in the team channel at Day 2 EOD.

---

### A4 — Clustering, co-membership, stability (`src/clustering/`)

**Goal.** Rolling 21-trading-day reclustering of the 40 stocks on their formation-window beta vectors; label and stability outputs for track a (machinery reused for tracks c/d).

**Consumes:** `loadings_a.parquet` (beta vectors), trading calendar from `returns.parquet`.
**Produces:** `data/clusters/labels_a.parquet`, `data/clusters/stability_a.parquet` (schemas per Section 2 contract).

**Signatures:**

```python
def fit_kmeans_select_k(X: np.ndarray, k_range: range,
                        seed: int = 311) -> tuple[np.ndarray, int, float]:
    """k-means++ init, n_init=10, for each k in k_range compute silhouette on X
    (formation data ONLY); return (labels for best k, best_k, best_silhouette)."""

def comembership(labels: np.ndarray) -> set[tuple[str, str]]:
    """Set of alphabetically-ordered ticker tuples co-clustered under labels."""

def stability_frame(prev_pairs: set | None, curr_pairs: set,
                    window_end: pd.Timestamp) -> pd.DataFrame:
    """One row per pair in curr_pairs: (window_end, pair_id, co_clustered =
    pair in prev_pairs). First window: co_clustered = False for all."""

def run_recluster_loop(betas_long: pd.DataFrame, cadence: int = 21,
                       formation: int = 252, track: str = "a") -> None:
    """Every 21 trading days from index 252+21 onward: build per-stock feature
    vectors, cluster, write labels_{track} and stability_{track}."""
```

**Algorithm notes:**

- **Cluster input** (config freeze, Zhang's compressed option): each stock's vector of factor betas from the formation window. Use the betas fitted at the **window-end date** (the A2 betas dated `window_end` were estimated on exactly the trailing 252 days — no separate re-fit needed). Feature dim = that date's `n_components`; standardize columns of the 40×m beta matrix before k-means (window-local, trivially — it's one cross-section).
- k selection by **max silhouette, k ∈ 8..13, on formation data only** — never on trading outcomes. Log `(window_end, best_k, silhouette)` to `data/clusters/kmeans_log_a.csv` for the report.
- `sklearn.cluster.KMeans(init="k-means++", n_init=10, random_state=311)`. **Empty clusters:** sklearn reseeds internally (relocates a centroid to the point with highest inertia contribution), so no custom handling is needed — but assert `len(np.unique(labels)) == k` after fit and log a warning if a cluster came out empty-then-reseeded, since with 40 points and k=13 it can happen; this is the note the spec asks for.
- **Label switching:** never compare raw `cluster_id` across windows. All cross-window logic goes through `comembership()` sets — invariant to relabelling by construction. `cluster_id` in `labels_a.parquet` is only meaningful within one `window_end`.
- Stability metric for the report: per window, `mean(co_clustered)` = fraction of currently co-clustered pairs that were also co-clustered last window. `stability_a.parquet` is also the direct upstream of C's `f_cluster_stability` feature — the contract is the per-pair boolean, not the aggregate.
- Recluster dates: every 21 trading days on the actual trading calendar (positional stride over `returns.index`, starting at index 252), NOT calendar days.
- **Clustering-input decision (spec Step 4 says "try both"):** the committed input is Option 2 — factor-beta vectors — on Zhang's decisive evidence (PC-based clustering passed the statistical-arbitrage test at p = 0.01 vs p = 0.785 for raw features). This is recorded as an explicit kickoff decision in `DECISIONS.md`, so the deviation from "try both" is deliberate, not an omission. Option 1 (the 252-length residual-series vectors) survives only as a Day 6 robustness check (~1.5h, run **only if Track C is NO-GO** and the day has slack): cluster residual series for ~6 representative windows, report co-membership overlap with the committed clustering — one report paragraph either way.

**Unit tests:** (a) planted 3-cluster synthetic betas → silhouette selects a k that recovers the planted partition (compare via co-membership sets, not labels); (b) `comembership` invariant under label permutation; (c) `stability_frame` correct on hand-built two-window example; (d) seed 311 → identical labels across two runs; (e) recluster dates are exactly every 21st trading day.

**Hours:** 4h, Day 3.
**Done when:** `labels_a` + `stability_a` written for every recluster date on real data; k-log saved; tests pass.

---

### A5 — The shared pair builder (`src/pairs/build_pairs.py`)

**Goal.** One function that turns any track's cluster labels into `pairs_{track}.csv` under Zhang's rules. **This is a shared interface: Person B imports `build_pairs` for Track B on Day 3.** The canonical signature lives in Section 2.2 (this section implements it); a stub raising `NotImplementedError` is committed at kickoff so B can import and mock from Day 2, and the implementation is tests-green and tagged `v1-frozen` by Day 3 AM.

**Consumes:** `labels_{track}.parquet` + the per-window feature matrix used for clustering (for within-cluster distances) + the recluster-date calendar.
**Produces:** `data/pairs/pairs_{track}.csv` (contract columns: `pair_id, stock_a, stock_b, group_id, source, active_from, active_to`).

**Signatures:**

```python
def split_large_cluster(members: list[str], D: pd.DataFrame) -> list[list[str]]:
    """Zhang 5+ rule via greedy nearest-neighbour. D = within-cluster Euclidean
    distance matrix over the SAME feature space used for clustering."""

def pairs_for_window(labels: pd.DataFrame, features: pd.DataFrame,
                     window_end: pd.Timestamp, source: str) -> pd.DataFrame:

def build_pairs(labels_path: str, features_by_window: dict[pd.Timestamp, pd.DataFrame],
                source: str,               # "track_a" | "track_b" | "track_c" | "track_d"
                calendar: pd.DatetimeIndex, out_csv: str) -> pd.DataFrame:
```

**Zhang rules, and the 5+ split algorithm precisely:**

1. Drop singleton clusters.
2. Clusters of 2–4: take all pairs (C(n,2)).
3. Clusters of 5+: split into subgroups of 2–3 by greedy nearest-neighbour:
   - Compute the within-cluster pairwise Euclidean distance matrix `D` on the clustering feature vectors.
   - Repeat while ≥ 2 members unassigned: find the globally minimum `D[i, j]` over unassigned i, j; form `{i, j}` as a new subgroup; mark both assigned.
   - If exactly **one** member remains unassigned, absorb it into the subgroup containing its nearest assigned neighbour, **provided that subgroup has size 2** (making a 3); if that subgroup already has 3, absorb into the next-nearest size-2 subgroup. (With subgroups built as 2s, a size-2 host always exists when the cluster size is odd.)
   - Take all pairs within each subgroup.
4. `pair_id = f"{min(a,b)}__{max(a,b)}"` — tickers alphabetical, always.
5. Active windows tied to the recluster cadence: `active_from` = first trading day strictly after `window_end`; `active_to` = the next recluster's `window_end` (inclusive), or the last data date for the final window. Consecutive selections of the same `pair_id` produce consecutive rows whose windows tile without gap — A6 relies on this to detect continuity.

**Unit tests:** (a) cluster sizes 1/2/3/4 → 0/1/3/6 pairs; (b) size-5 with a hand-built distance matrix → known subgroups {closest pair}, {next pair + leftover} and exactly 1+3=4 pairs; (c) size-6 → three 2-subgroups, 3 pairs; (d) `pair_id` alphabetical regardless of input order; (e) active windows tile the calendar exactly; (f) `source` propagates. These tests are the acceptance gate for B's import — tag the module `v1-frozen` in git once green (Day 3 AM at the latest).

**Hours:** 2.5h, split: Day 2 PM (split algorithm + unit tests against hand-built distance matrices — fixture-testable, no real-data dependency; 1.5h) and Day 3 AM (final wiring + `v1-frozen` tag; 1h). This split exists to shorten Sunday's serial chain (see the risk register on weekend load).
**Done when:** tests green, `pairs_a.csv` written on real data Day 3 EOD, B has successfully called `build_pairs(..., source="track_b")`.

---

### A6 — Spreads and z-scores (`src/spreads/spread.py`)

**Goal.** Per-pair spread (simple version per config) and rolling 60d z-score, for every track's pair file.

**Consumes:** `residuals_a.parquet` (or `residuals_d.parquet` for track d), `pairs_{track}.csv`.
**Produces:** `data/spreads/spreads_{track}.parquet`, `data/spreads/zscores_{track}.parquet` (date × pair_id).

**Signatures:**

```python
def build_spread_for_pair(residuals: pd.DataFrame, pair_rows: pd.DataFrame,
                          warmup: int = 60) -> pd.Series:
    """All active rows for one pair_id -> spread series over its active runs."""

def zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    """(spread - rolling_mean) / rolling_std, window-local trailing stats.
    min_periods=window; NaN during warmup."""

def run_spreads(track: str, residuals_path: str) -> None:
```

**The re-anchoring policy (the flagged subtlety).** With a 21-day recluster cadence and a 60-day z-window, restarting the spread at every `active_from` would leave every pair NaN-z for its first ~3 windows — most pairs would never become tradeable, and Track B's quarterly (63-day) windows would barely clear warmup. So the frozen policy is:

> **`SPREAD_POLICY = "carry_with_burnin"`, `SPREAD_WARMUP_DAYS = 60` (added to `src/config.py` at kickoff).** Group each pair_id's rows in `pairs_{track}.csv` into maximal **runs** of consecutive active windows (windows tile, so a gap = the pair dropped out and re-entered). Within a run, the spread accumulates continuously across window boundaries — same pair, same economic relationship, no re-anchor. At the **start** of each run, accumulation begins `60` trading days *before* `active_from`, using already-written residuals from those dates. This burn-in is backward-looking history available at `active_from` — **no leakage** — and it means z-scores are valid from the first active day of every run. z-scores are still only *emitted* (non-NaN in `zscores_{track}`) for dates inside active windows; burn-in dates exist only internally.

Justification, one line for the report: continuity within a run reflects that the pair relationship persists across refreshes; the historical burn-in makes new pairs immediately tradeable without touching future data. (The rejected alternative — hard restart + overlap requirement — is dominated: it either forbids new pairs or silently shrinks the tradeable set.)

Other notes: z-score uses **trailing** rolling mean/std (`.rolling(60, min_periods=60)` on the shifted-inclusive series — stats at t may include t's spread value per spec 6b's plain rolling z; the trigger comparison in C's Step 7 uses these z values as-is). Spread per config: `spread_t = Σ (resid_A − resid_B)` over the run from burn-in start; no hedge ratio.

**Unit tests:** (a) planted OU fixture from `synth/fixture_zscores.py` → z-scores match the golden file within tolerance (this is the *shared* golden fixture C also tests against — exactness here is what makes Integration Checkpoint 1 boring); (b) a pair active in windows 1–3, gone in 4, back in 5 → two runs, second run's spread re-anchors with burn-in, no value bleeds across the gap; (c) z is NaN nowhere inside active windows after burn-in; (d) columns of outputs == unique pair_ids.

**Hours:** 2.5h Day 3 (track a, EOD deliverable) + 1h Day 4 AM (track b, as soon as B's `pairs_b.csv` lands — run is one function call; the hour is for eyeballing distributions).
**Done when:** `zscores_a.parquet` Day 3 EOD (C's real trigger build depends on it for Checkpoint 1); `zscores_b.parquet` Day 4 by noon.

---

### A7 — Leakage-audit items 1–4 (`scripts/audit/audit_a_items_1_4.py`)

**Goal.** Mechanically verify the four Part 4 checklist items A owns, using the future-perturbation helper defined in Plan Section 2 (`tests/leakage_utils.py`: `perturb_after(df, date, seed)` replaces all rows after `date` with seeded noise; `assert_no_future_dependence(...)` wraps the comparison).

**Consumes:** all A-pipeline modules + real data. **Produces:** printed PASS/FAIL per item + `results/audit/audit_items_1_4.md` (four short written notes, one paragraph each, pasted into the report's checklist section).

| Item | Test |
|---|---|
| 1. PCA on trailing windows only | Run `run_rolling_pca` on real returns and on `perturb_after(returns, T)`; assert `factors_a`, `loadings_a` **bit-identical** for all dates ≤ T. Repeat for two choices of T (mid-2018, mid-2022). |
| 2. Standardization window-local | Same perturbation test catches full-sample stats automatically (full-sample mu/sigma would change pre-T output); additionally grep-audit `src/factors/` for any `.mean()`/`.std()` call not on a window slice, and cite the code lines in the note. |
| 3. Betas estimated before application | Perturbation test on `residuals_a` (identical ≤ T); plus the explicit in-loop assertion from A2 that the beta window excludes t. |
| 4. Clustering on formation data only | Perturbation test on `labels_a`/`stability_a` for recluster dates ≤ T; plus a written note that k selection reads silhouette, never trading outcomes, with the code path cited. |

Runtime is a few full-pipeline re-runs — minutes, not hours. Failures here are release-blocking for the Day 5 grid run.

**Hours:** 3h total — 2h Day 4 (script + first pass), 1h Day 5 (final pass as part of the team leakage-checklist review, plus polishing the four written notes).
**Done when:** four PASS lines from the Day 5 run, notes committed.

---

### A8 — Track C: partial-correlation distance (gated; Day 6 only)

Runs **only** if the Day 5 evening sync gate passes (spec 8.0: full 2×4 end-to-end on real data, leakage audit + noise test passed). Owner A; ~5h per spec 8.1. Everything downstream of pairs is untouched — that is the point.

**Consumes:** the in-window correlation matrices already computed in A1 (refactor `pca_one_window` to optionally return `C`), A4/A5/A6 machinery. **Produces:** `labels_c.parquet`, `stability_c.parquet`, `pairs_c.csv` (source `track_c`), `spreads_c.parquet`, `zscores_c.parquet`, overlap figures.

Subtasks:

| | Task | Hours |
|---|---|---|
| A8.1 | `src/trackc/partial_corr.py`: `partial_corr_distance(C: np.ndarray, lam: float = 1e-3) -> np.ndarray` — diagonal shrinkage `(1-lam)*C + lam*I`, invert to precision `P`, `pcorr = -P/outer(sqrt(diag(P)))` with unit diagonal, `dist = sqrt(2*(1-pcorr))`. Unit tests: pcorr symmetric, diag 1, known 3-variable analytic case; near-singular C inverts stably after shrinkage. Record `lam` in config + report; not tuned. | 1.5 |
| A8.2 | Wire into A4's recluster loop at the same 21-day cadence, formation-window matrices only (leakage item 11). k-means needs vectors, not distances: fit on each stock's 40-dim row of the distance matrix (standard workaround; state it in the report), but **select k via `silhouette_score(D, labels, metric="precomputed")` on the actual distance matrix** — D is the true dissimilarity (mirrored in Section 7.3). Same k-range 8–13, seed 311. Write `labels_c`, `stability_c`. | 1.5 |
| A8.3 | Pairs via A5 (`source="track_c"`) → spreads/z via A6 against `residuals_a` (Track C changes only the *distance metric for grouping*; the tradeable object is still the Track A residual spread). One function call each + spot checks. | 0.5 |
| A8.4 | Overlap analysis: per window, Jaccard overlap of Track C vs Track A (and vs B) co-membership sets + pair lists; one figure `results/figures/trackc_overlap.png`. Feeds B's consensus-pairs extension (spec 8.3) — hand B the per-window pair sets, interface = `pairs_c.csv` itself. | 1.0 |
| A8.5 | Noise-test rerun including Track C (Part 4 requirement: any extension that runs must pass), audit note for checklist item 11. | 0.5 |

**Done when:** A8.1–A8.3 complete and `zscores_c.parquet` delivered to C/B **by Day 6 noon** (front-load the 5h into Day 6 AM + early PM) — so C can assemble `triggers_c`, fit/tune the track-c models, and select τ_c (~2h, Day 6 PM) before the C13 freeze, which on a Track C go moves to late Day 6 evening (after the sync). If the noon hand-off slips, Track C is reported **on validation only** and excluded from the frozen test run, stated plainly in the report. Noise test PASS (incl. the track-c row); item-11 note written. If the gate fails, A8 collapses to one literature-review sentence (spec 8.4) and Day 6 hours roll into figures + the Option-1 robustness check (A4 note).

---

### Person A day-by-day hour budget

| Day | Tasks | Hours |
|---|---|---|
| 1 (Fri) | Kickoff (2h) + A0 fixture generator (1.5h) + A1 build vs fixture prices (3h) + rerun on real returns at EOD (0.5h) | 7 |
| 2 (Sat) | A1 finish/validate real (1h) + perturbation helper `tests/leakage_utils.py` (1h) + A2 (3h) + A3 checks (2h) + A5 split algorithm + tests (1.5h) — factors, residuals, 4 PASS figures by EOD | 8.5 |
| 3 (Sun) | A4 (4h) + A5 final wiring + `v1-frozen` tag (1h) + A6 track a (2.5h) — `zscores_a` EOD | 7.5 |
| 4 (Mon) | A6 track b (1h, AM) + Integration Checkpoint 1 (1h) + A7 script (2h) + buffer/debug for C's real triggers (2h) | 6 |
| 5 (Tue) | Grid-run support + noise-test triage on A-stack (2h) + leakage review + A7 final + notes (2h) + Track C gate sync (1h) | 5 |
| 6 (Wed) | A8 Track C (5h, front-loaded: `zscores_c` by noon) *or*, on NO-GO, Option-1 robustness check (1.5h) + figures; A-owned figures polish (2h) | 7 |
| 7 (Thu) | Witness test run (1h) + finalize A figures/tables (3h) + report: methods Steps 2–6 draft (3h) | 7 |
| 8 (Fri) | Writing, limitations items owned by A (betas drift, z-score simplification, shrinkage), read-through | 6 |

---

### Workstream A definition of done

- [ ] `factors_a`, `pca_meta`, `loadings_a` (with betas), `residuals_a` written from real data; A1/A2 tests green (Day 2 EOD)
- [ ] Four sanity checks PASS on real data; four figures in `results/figures/` (Day 2 EOD)
- [ ] `labels_a`, `stability_a`, `pairs_a.csv`, `spreads_a`, `zscores_a` written; pair-builder frozen and successfully imported by B (Day 3 EOD)
- [ ] `zscores_b` delivered by Day 4 noon; golden-fixture z-score test passes (shared with C)
- [ ] Leakage items 1–4: perturbation tests PASS at two cut dates; four written audit notes committed (Day 5)
- [ ] A-stack passes the noise test (no structure found in random-walk input)
- [ ] Track C: gate decision recorded in DECISIONS.md; if run — full artifact set + overlap figure + noise-test PASS + item-11 note; if skipped — lit-review sentence
- [ ] Every stochastic call seeded 311; two consecutive full A-pipeline runs byte-identical
- [ ] All config values imported from `src/config.py`; the two A-added constants (`RESIDUAL_INCLUDE_ALPHA`, `SPREAD_POLICY`/`SPREAD_WARMUP_DAYS`) ratified at kickoff and logged in DECISIONS.md

---

## 4. Workstream B — Data, Markets & Experiments (Person B)

Person B owns the two ends of the pipeline — the data that feeds everyone (Steps 0–1), and the experiment machinery that consumes everyone's outputs (backtest engine, E0, runner, costs, controls, noise test) — plus Track B end-to-end. The critical path property: **nothing in B5–B12 waits on real data.** The engine, runner, and cost logic are built and unit-tested against `synth/fixture_zscores.py` output on Days 2–3 and swap to real artifacts at Integration Checkpoint 1 (Day 4).

Module map (all under the repo root):

```
src/data/universe.py          # B1
src/data/prices.py            # B2
src/data/characteristics.py   # B3 (loader) + B4 (pipeline)
src/backtest/engine.py        # B5
src/models/e0_baseline.py     # B6
src/experiments/run_grid.py   # B7
src/experiments/cost_sweep.py # B8
src/experiments/turnover_control.py  # B9
src/experiments/consensus.py  # B10
src/experiments/noise_test.py # B11  (+ Makefile target: make noise-test)
docs/manual_trace.md          # B12
tests/test_prices.py  tests/test_engine.py  tests/test_costs.py
tests/test_characteristics.py tests/test_runner_guard.py
```

---

### B1 — Universe selection (Day 1, kickoff item)

**Goal.** Fix the 40-ticker universe: 4 sectors × 10 large-cap S&P 500 names, every ticker continuously listed 2014-01-01 through 2025-01-01 (one year of pre-2015 history is needed for the first 252-day PCA window). Sectors chosen for beta diversity: Information Technology (high beta), Financials (rate-sensitive), Energy (commodity-driven), Consumer Staples (defensive) — this guarantees PC2/PC3 have real sector structure to find.

**Module.** `src/data/universe.py`

```python
SECTORS: dict[str, list[str]]   # the table below, as a literal
TICKERS: list[str]              # sorted flat list of 40
def get_universe() -> pd.DataFrame:
    """Returns DataFrame[ticker, sector]. Single source of truth for every module."""
```

**Proposed list** (ratified at kickoff; auto-validated by B2's cleaning checks — any ticker failing the 2%-missing or calendar checks is replaced from the same sector at kickoff +1 day, logged in `DECISIONS.md`):

| Information Technology | Financials | Energy | Consumer Staples |
|---|---|---|---|
| AAPL | JPM | XOM | PG |
| MSFT | BAC | CVX | KO |
| NVDA | WFC | COP | PEP |
| ORCL | C | SLB | WMT |
| CSCO | GS | EOG | COST |
| INTC | MS | OXY | MDLZ |
| IBM | USB | VLO | CL |
| TXN | PNC | MPC | KMB |
| ADBE | AXP | PSX | GIS |
| QCOM | BLK | HAL | SYY |

All 40 were listed and in the S&P 500 well before 2014 (MPC spun off 2011, PSX 2012 — both safely pre-window). Deliberately excluded: post-2015 IPOs/spinoffs (ABNB, COIN, CARR, OTIS, KVUE, GEV, CEG), dual-class duplicates (one GOOGL-style listing max — none included), and anything acquired/delisted mid-sample. Survivorship bias is acknowledged per spec Step 0 and Plan Section 2; it is disclosed, not fixed.

- **Consumes:** nothing. **Produces:** `src/data/universe.py` (code is the artifact).
- **Definition of done:** `get_universe()` returns 40 rows, 4 sectors × 10; ratified in kickoff minutes.
- **Hours:** 1h (list is pre-drafted above; kickoff ratifies).

### B2 — Prices, volume, SPY, returns (Day 1)

**Goal.** Download, clean, and freeze the price data everyone builds on. Real `returns.parquet` must land by Day 1 EOD — this is Person A's Day 2 input.

**Module.** `src/data/prices.py`

```python
def download_prices(tickers: list[str], start: str = "2014-01-01",
                    end: str = "2025-01-01", max_retries: int = 3,
                    cache_dir: str = "data/raw/cache/") -> tuple[pd.DataFrame, pd.DataFrame]:
    """yfinance pull, auto_adjust=True asserted explicitly (do not trust the default
    silently — assert on the call kwargs). Per-ticker retry with exponential backoff;
    raw per-ticker CSVs cached to cache_dir and committed, so the pull is reproducible
    even if yfinance data shifts later. Returns (prices, volume), date x ticker."""

def clean_prices(prices: pd.DataFrame, volume: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Runs every spec Step 0 check as an explicit assert + printed report.
    Returns (prices, volume, report_df)."""

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """np.log(p / p.shift(1)), first row dropped. LOG_RETURNS per config."""
```

Cleaning checks — each one an `assert` (hard checks) or a printed WARN table (review checks), all echoed into a report printed at run time and pasted into `DECISIONS.md`:

1. **≤2% missing rule:** per ticker, missing-day fraction vs the union calendar; any ticker >2% → hard fail (triggers the B1 replacement protocol).
2. **Forward-fill isolated single-day gaps only:** a gap is fillable iff exactly 1 NaN with valid values on both sides; multi-day gaps are *not* filled — they fail check 1 or get flagged.
3. **Common trading calendar:** intersect all tickers + SPY to NYSE trading days; assert identical index across all output frames.
4. **|return| > 50% flags:** list every (date, ticker, return) exceeding ±50% into a manual-review table; each row needs a one-line verified-event note (e.g., known split mishandled) or the ticker is escalated. Expected count on adjusted large-caps: zero.
5. **~252 rows/year:** assert `250 <= rows_per_calendar_year <= 254` for every full year.

- **Consumes:** `src/data/universe.py`. **Produces:** `data/raw/prices.parquet`, `data/raw/volume.parquet`, `data/raw/spy.parquet`, `data/processed/returns.parquet` (all per the Section 2 contracts), committed cache CSVs.
- **Tests:** `tests/test_prices.py` — synthetic frame with a planted 1-day gap (filled), a 3-day gap (not filled), a fake +80% return (flagged); returns of a hand-priced 3-day series equal hand-computed logs.
- **Definition of done:** all four parquets exist, cleaning report shows 40/40 tickers passing, Person A confirms `returns.parquet` loads with expected shape (~2,760 × 40 including the 2014 warm-up year).
- **Hours:** 4h. **Day:** 1 (EOD hard deadline).

### B3 — Bloomberg Track B pull (Day 2, one terminal session)

**Goal.** One sitting at the Bloomberg terminal: quarterly observations of the 19 spec-4B.1 fields, 40 tickers, 2015Q1–2024Q4 (~30k cells). **The session falls on a Saturday — confirm weekend terminal access before Day 1**; if the terminal is weekday-only, the pull moves to Day 1 (before or after kickoff).

**Session plan (write this on paper before sitting down):**

1. **First 20 minutes — verify mnemonics with `FLDS`** on one ticker (AAPL). Every mnemonic below is best-effort and flagged **verify-on-terminal**; record the final mnemonics used in `data/raw/characteristics/FIELDS_USED.md`:

| Spec field | Candidate mnemonic | Spec field | Candidate mnemonic |
|---|---|---|---|
| P/E | `PE_RATIO` | Dividend/Share | `EQY_DVD_SH` (or `EQY_DVD_SH_12M`) |
| Price/Book | `PX_TO_BOOK_RATIO` | Volatility 60d | `VOLATILITY_60D` |
| Price/Sales | `PX_TO_SALES_RATIO` | RSI | `RSI_14D` |
| Price/EBITDA | `PX_TO_EBITDA` (verify vs EV variants) | Close | `PX_LAST` |
| Market Cap | `CUR_MKT_CAP` | Ask | `PX_ASK` |
| Shares Out | `EQY_SH_OUT` | Bid | `PX_BID` |
| Sales Growth | `SALES_GROWTH` | Analyst Rating | `TOT_ANALYST_REC` (composite: `BEST_ANALYST_RATING` — verify) |
| Cash Flow Growth | `CASH_FLOW_GROWTH` (verify) | Buy Recs | `TOT_BUY_REC` |
| FCF Growth | `FREE_CASH_FLOW_GROWTH` (verify) | Sell Recs | `TOT_SELL_REC` |
| Normalized ROE | `RETURN_COM_EQY` / `NORMALIZED_ROE` (verify) | | |

2. **Pull via Excel BDH** (quarterly periodicity, `Days=A`, `Fill=P`): one sheet per field, dates × tickers, `=BDH(ticker, field, "12/31/2014", "12/31/2024", "Per=CQ")`. Export each sheet to CSV immediately — do not leave the terminal with data only in a live spreadsheet.
3. **Point-in-time caveat (spec 4B.1):** where `FLDS` shows an as-reported variant (e.g., `IS_EPS` family vs adjusted), prefer it and note it in `FIELDS_USED.md`; where only restated values exist, pull them and add the field to the limitations disclosure list. Do not burn session time hunting — 15 minutes max, then disclose.
4. **Before leaving:** spot-check 5 random cells on the terminal screen vs the CSVs; confirm ≥ 90% cell coverage per field (fields below that are candidates for B4's column-drop).

**Post-session:** `src/data/characteristics.py::load_raw()` melts the per-field CSVs into the long schema and validates it:

```python
def load_raw(raw_dir: str = "data/raw/characteristics/") -> pd.DataFrame:
    """Returns long DataFrame[date, ticker, field, value]; asserts 19 fields,
    40 tickers, 40 quarterly dates; prints per-field coverage table."""
```

**HARD FALLBACK RULE (schedule protection).** If a validated pull has not landed by the **Day 2 evening sync**, Track B degrades — it does not slip. The decision fires *before* B4 consumes the data on Day 3 morning, not after (a Day-3-EOD trigger would fire only after B4's six hours and the `pairs_b` deadline were already gone): rebuild a reduced characteristic set from free sources: **sector** (one-hot, from `universe.py`), **market cap** (shares outstanding from free filings snapshots × our price — approximate), and **trailing price-derived fields computed from our own `prices.parquet`/`volume.parquet`**: 60d volatility, RSI-14, close price, 20d dollar volume, 12m momentum. Historical *fundamentals* (P/E, ROE, growth) from free sources are unreliable point-in-time and are **not** substituted. This is ~8 usable columns instead of 18; the factorial design still runs with a weaker Track B, and the degradation gets a prominent limitations paragraph plus a `DECISIONS.md` entry. The fallback path reuses B4's pipeline unchanged (it just sees fewer columns).

- **Consumes:** terminal access; `universe.py`. **Produces:** `data/raw/characteristics/*.csv` (one per field) + `FIELDS_USED.md`.
- **Definition of done:** `load_raw()` passes validation, coverage table printed, point-in-time notes recorded — or the fallback rule is invoked and logged.
- **Hours:** 3h session + 1h post-processing. **Day:** 2.

### B4 — Track B pipeline: characteristics → clusters → pairs (Day 3)

**Goal.** Zhang's pipeline (spec 4B.2–4B.6) from raw characteristics to `pairs_b.csv`, per quarterly snapshot.

**Module.** `src/data/characteristics.py` (continued)

```python
def clean_snapshot(long_df: pd.DataFrame, quarter_end: pd.Timestamp) -> pd.DataFrame:
    """One quarter -> 40 x ~18 cleaned matrix. Zhang's three steps, exactly:
    1) sentiment = (buy - sell) / (buy + sell)   [replaces the two rec-count cols; 19 -> 18]
    2) log-transform size columns: market_cap, shares_out -> np.log(x)
    3) per-column z-score: (x - col.mean()) / col.std()   [within this snapshot]
    Our policy first: drop any column with < 90% coverage this quarter; median-impute
    remaining gaps (median over the 40 stocks, this quarter); then drop any column with
    zero/near-zero variance this quarter (post-imputation) BEFORE z-scoring — a constant
    column divides by ~0 and NaN-poisons the correlation matrix. Log all three actions."""

def pca_characteristics(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """corr matrix (~18x18) via np.corrcoef — assert np.isfinite(C).all() before eigh —
    then np.linalg.eigh, descending sort, sign fix (sum of loadings > 0). Keep smallest
    m with cumvar >= 0.60, capped at 5 (thin-data caveat, spec 4B.3). Returns
    (eigvals, eigvecs, scores) where scores = Route B projection: 40 stocks x m scores.
    Route A (spec 4B.5's PCA-as-detector) is deliberately skipped, citing Zhang's
    decisive comparison (p = 0.01 passed vs p = 0.785 failed); logged in DECISIONS.md."""

def name_components(eigvecs: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """The 'read the components' deliverable (spec 4B.4): top-5 |loading| characteristics
    per PC for PC1-PC3, written to results/tables/track_b_components.md with a
    proposed name column (e.g. PC1 = 'expensiveness') for the report."""

def cluster_track_b(scores: pd.DataFrame, quarter_end: pd.Timestamp,
                    seed: int = 311) -> pd.DataFrame:
    """k-means (k-means++, n_init=10), k in 10..13 by max silhouette on this snapshot
    only. Returns long DataFrame[window_end, ticker, cluster_id]."""
```

Pairs come from **Person A's shared pair-builder** — the canonical frozen signature lives in Section 2.2 (stub committed at kickoff, tests-green by Day 3 AM); B passes the quarterly labels plus per-quarter score matrices as `features_by_window` (they supply the within-cluster distances for the 5+ split). B4 calls it with `source="track_b"` and **quarterly active windows** (each quarter's clusters govern `active_from` = first trading day after quarter end, `active_to` = last day of the following quarter). Stability: `stability_b.parquet` built by the same co-membership convention as Track A — pair co-clustered in consecutive quarterly snapshots.

- **Consumes:** `data/raw/characteristics/*.csv` (or fallback set), `src/pairs/build_pairs.py` (Person A, interface frozen Day 2). **Produces:** `data/clusters/labels_b.parquet`, `data/clusters/stability_b.parquet`, `data/pairs/pairs_b.csv`, `results/tables/track_b_components.md`.
- **Tests:** `tests/test_characteristics.py` — sentiment formula on hand values (15 buys/5 sells → +0.5); z-scored columns have mean≈0, std≈1; a planted low-coverage column gets dropped; PCA on a rank-2 synthetic characteristic matrix recovers 2 dominant components.
- **Definition of done:** `pairs_b.csv` validates against the Section 2 schema; component-naming table exists; Person A confirms `pairs_b.csv` ingests into the spread builder (A produces `zscores_b` Day 4 AM).
- **Hours:** 6h. **Day:** 3 (EOD).

### B5 — Backtest engine + trade ledger (Days 2–4, fixtures first)

**Goal.** The single engine every strategy flows through (Key Architectural Rule, Section 2). Built and golden-tested on `synth/fixture_zscores.py` output Days 2–3; pointed at real artifacts Day 4.

**Module.** `src/backtest/engine.py`

```python
def run_backtest(
    zscores: pd.DataFrame,        # date x pair_id  (zscores_{track}.parquet)
    prices: pd.DataFrame,         # date x ticker   (data/raw/prices.parquet)
    triggers: pd.DataFrame,       # triggers_{track}.parquet (trigger_id, pair_id, trigger_date, z_trigger, ...)
    decisions: pd.DataFrame,      # trigger_id, enter, p_hat
    cost_grid_bps: tuple[int, ...] = (0, 5, 10, 15, 20, 30, 40, 50),
) -> pd.DataFrame:                # the trades ledger, one row per entered trigger

def daily_strategy_returns(trades: pd.DataFrame, cost_bps: int = 10) -> pd.Series:
    """Equal-weight average of concurrent open trades' daily P&L. See aggregation policy."""
```

**Mechanics encoded (all per the config freeze — restated here only where the implementation detail is non-obvious):**

- **Entry:** at the close of `trigger_date + 1` trading day (t+1 rule). First P&L day is therefore `trigger_date + 2`'s return. If `entry_date` would fall beyond the data, the trigger is dropped and counted.
- **Direction:** from `sign(z_trigger)`: `z > 0` → spread (A−B) is high → **short stock_a, long stock_b**; `z < 0` → the reverse. $1 notional per leg.
- **Exit:** first trading day with `|z| < 0.5` (exit at that day's close, `exit_reason="reverted"`), else at close 5 trading days after entry (`exit_reason="timeout"`). `days_held` = trading days entry→exit. **Disclosed asymmetry:** entry is lagged one day (t+1), but the exit fills at the same close that generated the exit signal — this matches the spec's own E0 wording (Step 9) and is stated explicitly in the report's execution paragraph and the B12 trace template so leakage-checklist item 8 reads airtight.
- **P&L from RAW stock returns**, not residuals: the engine derives **simple** returns internally from `prices` (`p/p.shift(1) - 1`; log returns would misprice the short leg). Per-day trade P&L in dollars: `pnl_t = r_long_t − r_short_t`; `gross_ret = Σ pnl_t` over `(entry_date, exit_date]` — accumulated per-day so `daily_strategy_returns` can reuse the same numbers.
- **Costs at transaction time:** `c` bps per leg per transaction → 2 legs at entry + 2 legs at exit = **4 applications**. With fixed $1/leg notional: `net_ret_{c}bps = gross_ret − 4 × c × 1e-4`, but *booked* as −2c bps on the entry day and −2c bps on the exit day in the daily P&L stream (spec Step 11: at the point of each transaction, never a lump at the end). One `net_ret_{c}bps` column per grid value, precomputed — B8's sweep then reads columns, no reruns.
- **Aggregation policy (disclosed in the report):** strategy daily return = **equal-weight mean of all concurrently open trades' daily P&L** (in per-$1-per-leg units); days with no open trades contribute 0; **no capital constraint** — every accepted trigger is taken regardless of how many trades are already open. Stylized, stated, identical across all strategies, therefore fair.

**Tests** (`tests/test_engine.py`, `tests/test_costs.py`) — written Day 2 against fixtures, must stay green forever:

1. **3-trade golden test:** hand-computed fixture (one reverted, one timeout, one z<0 direction flip) — assert `entry_date`, `exit_date`, `exit_reason`, `days_held`, `gross_ret` to 1e-10 against the hand math committed as a CSV next to the test. P&L is derived from the fixture's 12-ticker `prices.parquet` (Section 2.4 — the fixture ships leg prices precisely so this test can run).
2. **Shift test (leakage):** perturb day-t return of a traded stock with the trade triggered at t; assert ledger unchanged. Then perturb day t+2 (first P&L day); assert `gross_ret` moves by exactly the perturbation. Proves day-t signal never touches day-t return.
3. **Cost arithmetic test:** for the golden trades, assert `net_ret_10bps == gross_ret − 0.004` exactly, and that the daily P&L stream books −2c bps on entry and exit days.
4. **Concurrency test:** two overlapping trades → `daily_strategy_returns` equals hand-computed equal-weight mean.

- **Consumes:** fixtures (Days 2–3), then `zscores_a`, `triggers_a`, `prices.parquet`, decisions (Day 4). **Produces:** `results/trades_{track}_{model}.parquet` per the contract (writing routed through B7).
- **Definition of done:** all four tests green on fixtures (Day 3 EOD); real Track A ledger produced at Integration Checkpoint 1 (Day 4).
- **Hours:** 8h total (Day 2: 4h core loop + golden test; Day 3: 2h costs/aggregation + remaining tests; Day 4: 2h real-data integration).

### B6 — E0 baseline decisions (Day 4)

**Goal.** The fixed rule as a decisions table — through the SAME engine, so E0 is never special-cased.

**Module.** `src/models/e0_baseline.py`

```python
def e0_decisions(triggers: pd.DataFrame) -> pd.DataFrame:
    """Every trigger: enter=True, p_hat=NaN. Columns: trigger_id, enter, p_hat."""
```

Three lines of logic; its value is architectural. E1–E3 produce the identical schema via Person C's `EntryModel` API, and the turnover control (B9) produces it too — the engine cannot tell strategies apart.

- **Consumes:** `triggers_{track}.parquet`. **Produces:** `results/decisions_{track}_e0.parquet`.
- **Definition of done:** Track A × E0 runs end-to-end on real train+val data Day 4; trade count and base rate reviewed at Checkpoint 1.
- **Hours:** 1h. **Day:** 4.

### B7 — Experiment runner + results aggregation (Days 4–5)

**Goal.** One command per grid cell; test set physically guarded.

**Module.** `src/experiments/run_grid.py`

```python
def run_cell(track: str, model: str, split: str) -> dict:
    """Loads triggers_{track}, filters to split rows, obtains decisions:
    e0 -> e0_decisions(); e1/e2/e3 -> Person C's EntryModel API
    (interface: EntryModel.load(model, track).predict_proba(features) -> p_hat,
    with tau applied from C's frozen tau_{track}_{model}; fit/tuning happen in
    C's code on train/val only). Runs run_backtest, computes metrics via C's
    metrics module, writes decisions/trades/metrics artifacts. Returns metrics dict."""

# CLI: python -m src.experiments.run_grid --tracks a,b --models e0,e1,e2,e3 --split val
```

**Test-set guard (hard, not procedural):** `--split test` additionally requires `--i-am-sure` AND a date check `date.today() >= date(2026, 8, 6)` (Day 7) AND the presence of `results/FREEZE.md` (written at the Day 6 evening config freeze, listing frozen taus/hyperparameters). Any missing condition → `sys.exit` with an explanatory message. Guarded in `tests/test_runner_guard.py` (monkeypatched date). Additionally, `run_cell` asserts it never reads rows with `split` ∈ {`purged`, `embargo`} into any fit or metric.

**Aggregation:** `make_grid_table()` → `results/tables/grid_2x4.md` + `.csv`: rows = tracks, columns = models, each cell showing AUC [CI], net return @10bps [CI], trade count — the report's main table, **with row and column marginal means** so spec 12a's three questions (row effect, column effect, interaction) are read directly off it; B writes the three answers explicitly in the Day 8 results section. Extends to 3×4/4×4 automatically if Track C/D rows exist.

- **Consumes:** triggers, C's `EntryModel` + metrics interfaces, B5, B6. **Produces:** `results/decisions_*.parquet`, `results/trades_*.parquet`, `results/metrics_{track}_{model}.json`, `results/tables/grid_2x4.*`.
- **Definition of done:** Day 4 skeleton runs Track A × {E0, E1} on train; Day 5 the full 2×4 grid runs on train+val with one command; guard test green.
- **Hours:** 6h (Day 4: 3h skeleton + guard; Day 5: 3h full grid + aggregation table).

### B8 — Cost sweep + signature figure (Day 5)

**Goal.** Spec Step 11's sweep and breakeven analysis, from the ledger's precomputed `net_ret_{c}bps` columns — zero backtest reruns.

**Module.** `src/experiments/cost_sweep.py`

```python
def breakeven_bps(trades: pd.DataFrame) -> float:
    """Mean net return per trade is linear in c (slope -4e-4), so breakeven is exact:
    c* = mean(gross_ret) / 4e-4; also verified by interpolating across the grid columns."""

def cost_sweep_figure(trades_by_strategy: dict[str, pd.DataFrame],
                      out: str = "results/figures/cost_sweep.png") -> None:
    """Net cumulative return vs c, one line per strategy, breakeven c* annotated
    on each line, vertical reference at headline c=10. Val split only until Day 7."""
```

- **Consumes:** `results/trades_{track}_{model}.parquet`. **Produces:** `results/tables/breakevens.csv` (Day 5), `results/figures/cost_sweep.png` (Day 6 AM).
- **Tests:** golden trades → breakeven matches hand computation; interpolated and analytic breakevens agree to 1e-6.
- **Definition of done:** breakeven table cited at the Day 5 evening gate sync; sweep figure renders for all 8 strategies on validation by Day 6 noon.
- **Hours:** 2.5h split: 0.5h Day 5 PM (the breakeven table — one analytic formula over precomputed columns) + 2h Day 6 AM (the figure). The figure is deliberately off Day 5, the schedule's tightest day.

### B9 — Turnover-matched control (Day 6, spec 12b)

**Goal.** The project's headline experiment: is the filter's edge signal, or just fewer trades?

**Module.** `src/experiments/turnover_control.py`

```python
def matched_control_decisions(model_decisions: pd.DataFrame, e0_triggers: pd.DataFrame,
                              seed: int) -> pd.DataFrame:
    """Per calendar quarter q: n_q = # triggers the model accepted in q; sample n_q
    uniformly at random (rng seeded from 311 + seed) from ALL of q's E0 triggers.
    Returns a standard decisions table (enter bool, p_hat=NaN)."""

def run_control(track: str, model: str, n_seeds: int = 1000) -> dict:
    """1000 seeded draws -> 1000 engine runs -> distribution of net return @10bps
    (and AUC-equivalent hit rate). Reports the model's percentile within the control
    distribution; writes results/control_{track}_{model}.json + histogram figure with
    the model's actual value as a vertical line."""
```

**Why per-quarter matching (goes in the report verbatim):** matching only the *total* trade count would let the control spread trades uniformly over time while the model concentrates them — but market conditions, base rates, and costs cluster in time (2020 vs 2017 are different regimes). Per-quarter matching forces the control to share the model's **temporal footprint**, so the only remaining difference is *which* triggers within each quarter were chosen — which is exactly the signal question. 1000 runs are cheap: the engine on a few hundred triggers is milliseconds, and all `net_ret` columns are precomputed per trigger, so each control run is effectively a groupby over sampled rows.

- **Consumes:** `results/decisions_{track}_{model}.parquet` (each filtered model on val; test version regenerated Day 7 from frozen configs), `triggers_{track}`, B5. **Produces:** `results/control_{track}_{model}.json`, `results/figures/control_hist_{track}_{model}.png`.
- **Tests:** with a degenerate model accepting everything, control ≡ E0 per quarter; per-quarter counts of every draw exactly match the model's; seed 311 reproduces bit-identical distributions.
- **Definition of done:** percentile reported for Track A × E1 (primary comparison) with histogram; other filtered cells as time allows.
- **Hours:** 4h. **Day:** 6.

### B10 — Consensus pairs (Day 6, spec 12c)

**Goal.** Set operations on `pairs_a` / `pairs_b` at **pair-quarter granularity** (a pair-quarter is "consensus" iff the pair is active in both tracks during that quarter — pair lists change over time, so plain pair-level intersection would be wrong).

**Module.** `src/experiments/consensus.py`

```python
def bucket_pair_quarters(pairs_a: pd.DataFrame, pairs_b: pd.DataFrame) -> pd.DataFrame:
    """Expand each pairs file to (pair_id, quarter) rows from active_from/active_to;
    label each row consensus / a_only / b_only."""

def consensus_report(buckets: pd.DataFrame, triggers: dict[str, pd.DataFrame],
                     trades: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per bucket: n pair-quarters, n triggers, reversion rate (label base rate),
    E1 AUC restricted to the bucket, mean net return @10bps. One table
    (results/tables/consensus.csv) + one grouped-bar figure with bootstrap CIs."""
```

- **Consumes:** `data/pairs/pairs_a.csv`, `pairs_b.csv`, `triggers_{a,b}`, trades ledgers. **Produces:** `results/tables/consensus.csv`, `results/figures/consensus.png`.
- **Definition of done:** three buckets populated (report the overlap size even if consensus is tiny — "little overlap" is itself a spec-anticipated finding), table + figure done on val (refreshed with test numbers after Day 7).
- **Hours:** 4h. **Day:** 6.

### B11 — Noise test (Day 5, spec Part 4)

**Goal.** Run the FULL pipeline — PCA → residuals → clusters → pairs → z-scores → triggers → all models → engine → metrics — on `synth/make_synthetic.py` random-walk prices (the Day-1 fixture generator, reused verbatim). If the pipeline finds signal in pure noise, we have a leakage bug.

**Module.** `src/experiments/noise_test.py`, invoked as `make noise-test` (writes everything under `results/noise/`, seeded 311, never mixed with real artifacts).

**Concrete pass criteria** (evaluated automatically, printed as PASS/FAIL per line). One subtlety first: on random walks the label is *mechanically* related to the features without any leakage — P(|z| halves within 5 days) genuinely falls with |z_trigger| (a 3.5σ excursion must travel farther than a 2.1σ one) and varies with the level-vol/increment-vol ratio, and both quantities are features. So "AUC above 0.5 on noise" is **not** by itself evidence of a bug, and the binding criteria are:

1. **Returns (binding):** every strategy's mean net return per trade at c=10 is ≤ 0, or its 95% CI covers 0. (Gross may be mildly positive — z-score rules harvest some mean-reversion even from random walks via the rolling-window construction; *net-of-cost with a CI excluding zero on noise* is the red flag.)
2. **AUC vs the mechanical baseline (binding):** each model's AUC is not significantly above the AUC of an `f_abs_z`-only logistic model on the same synthetic triggers (bootstrap the AUC *difference*; its CI must cover 0). The raw model-AUC-vs-0.5 CI is printed as an **advisory** line only — it is expected to exclude 0.5 for the mechanical reason above and does not trigger the stop-the-line protocol by itself.
3. **Trigger count sanity:** triggers exist (> 50 — the machinery demonstrably fires) and the base rate is strictly inside (0, 1).

**Model coverage & timing:** the Day 5 run covers E0/E1/E2 (E3 is provisional on Day 5 by design); the criteria are re-run with E3 included on Day 6 before the freeze — a cheap rerun. The Track C gate is evaluated on the Day 5 E0/E1/E2 pass, stated explicitly at the sync.

**Failure = stop-the-line:** immediate all-hands message; nobody merges anything downstream of the suspected stage until the cause is found; the leakage checklist (spec Part 4) is re-walked item by item as a team; resolution logged in `DECISIONS.md`. The Day 5 evening Track C gate cannot pass while the noise test is red.

- **Consumes:** `synth/make_synthetic.py`, the entire pipeline (A's and C's modules included — this is also an integration test). **Produces:** `results/noise/metrics_*.json`, `results/noise/PASS_FAIL.md` (goes in the report's leakage section).
- **Definition of done:** `make noise-test` runs green end-to-end before the Day 5 evening sync.
- **Hours:** 3h (the pipeline already runs; this is orchestration + criteria). **Day:** 5.

### B12 — Manual trace (Day 5, spec Part 4)

**Goal.** Leakage-checklist item 8: hand-verify the t/t+1 alignment on paper.

**Procedure.** Pick 3 trigger dates from the real Track A × E0 ledger — one calm-regime (2017), one crisis (March–April 2020), one 2022 — one trade each. For each, on paper: the z-score path around the trigger, confirmation the crossing is an onset (prev day |z| < 2.0), entry at close of t+1, the two legs' raw prices and daily simple returns for each holding day, hand-summed gross P&L, cost deduction at c=10, exit day and reason. Then compare every number against the ledger row; any mismatch is a bug, full stop.

**Output:** `docs/manual_trace.md` from this template (committed Day 4 so the structure is agreed before use):

```markdown
## Trade N — {pair_id}, triggered {date} ({regime})
| item | hand value | ledger value | match |
|---|---|---|---|
| z_trigger / onset check | | | |
| entry_date (t+1) / direction | | | |
| daily leg returns (per day) | | | |
| gross_ret / net_ret_10bps | | | |
| exit_date / exit_reason / days_held | | | |
Verdict: MATCH / MISMATCH (issue #)
```

- **Consumes:** `results/trades_a_e0.parquet`, `zscores_a`, `prices.parquet`. **Produces:** `docs/manual_trace.md` (cited in the report's leakage audit). The template's convention line reads: *decision from day-t signal, entry at the t+1 close, first P&L day t+2; exit signal and exit fill share the same close (disclosed convention, B5).*
- **Definition of done:** 3/3 trades verdict MATCH, or a bug filed and fixed and the trace redone. This is a **hard Day 5 gate criterion** — it may not slip.
- **Hours:** 2h. **Day:** 5.

### B13 — The 12d question: does the characteristic advantage survive factor neutralization? (Day 7 PM)

**Goal.** Spec 12d's named analysis — the direct engagement with Han et al. (2023). The grid already produces the raw material (the Track A vs Track B row effect, measured on factor-neutralized residual spreads); this task reads it through the 12d lens and writes the discussion paragraph: if characteristics still help once common factors are removed, they carry economic information beyond factor loadings; if the advantage vanishes, Han's result may partly reflect characteristics proxying for factor exposure. Either answer is a real finding (spec 12d), and the second is a legitimate qualification of a 2023 *EJOR* paper.

- **Consumes:** the test-set grid table (T2 with marginal means) + consensus output. **Produces:** the 12d paragraph in the report's discussion; Han et al. (2023) added to the lit-review citation checklist.
- **Definition of done:** the paragraph exists in Overleaf with the row-effect numbers cited; Han et al. appears in the bibliography.
- **Hours:** 1.5h. **Day:** 7 PM.

---

### Person B — day-by-day hour budget

| Day | Date | Tasks | Hours |
|---|---|---|---|
| 1 | Fri Jul 31 | Kickoff (2h) · B1 (0.5h) · B2 (4h) · `config.py` commit + `contracts.py` v1 + `io_utils` (2h) → returns.parquet EOD | 8.5 |
| 2 | Sat Aug 1 | B3 Bloomberg session + post-processing (4h) · B5 engine core + golden test on fixtures (4h) | 8 |
| 3 | Sun Aug 2 | B4 Track B pipeline → pairs_b EOD (6h) · B5 costs/aggregation + tests (2h) | 8 |
| 4 | Mon Aug 3 | Checkpoint 1 (1h) · B5 real-data integration (2h) · B6 (1h) · B7 skeleton + test guard (3h) · trace template (0.5h) | 7.5 |
| 5 | Tue Aug 4 | B7 full 2×4 grid on train+val, AM (3h) · B11 noise test, PM (3h) · B12 manual trace (2h) · B8 breakeven table (0.5h) · evening sync | 8.5 |
| 6 | Wed Aug 5 | B8 sweep figure (2h) · B9 turnover control (4h) · B10 consensus (3.5h) — consensus is the designated slip-to-Day-7-AM candidate | 9.5 |
| 7 | Thu Aug 6 | Witnessed test run via `--split test --i-am-sure` (1h) · regenerate B8–B10 outputs on test + results freeze (3h) · B13 12d paragraph (1.5h) · figure/table finalization (1.5h) | 7 |
| 8 | Fri Aug 7 | Report: data, costs, control, consensus, leakage-audit sections (5h) · full-team read-through + submit (1h) | 6 |

Buffer: Day 4 is deliberately light (7.5h) to absorb Day 2–3 slippage (the Bloomberg fallback decision fires at the Day 2 evening sync). On Day 5 nothing may slip except the B8 sweep *figure* (already scheduled Day 6 AM) — the noise test and the manual trace are hard criteria of the Day 5 evening gate.

### Workstream B — definition of done

- [ ] `prices/volume/spy/returns` parquets committed; cleaning report 40/40 PASS, pasted in `DECISIONS.md`
- [ ] Bloomberg characteristics CSVs + `FIELDS_USED.md` with verified mnemonics and point-in-time notes — **or** fallback invoked, logged, and disclosed in limitations
- [ ] `labels_b`, `stability_b`, `pairs_b` validate against Section 2 schemas; Track B component-naming table exists
- [ ] Engine test suite green: golden 3-trade, shift test, cost arithmetic, concurrency
- [ ] Every strategy (E0, E1, E2, E3, control) flows through `run_backtest` via a decisions table — zero special cases
- [ ] `--split test` guard proven by test; test set executed exactly once, Day 7, witnessed, from `results/FREEZE.md` configs
- [ ] Cost-sweep figure + breakeven table for all 8 strategies
- [ ] Turnover-control percentile + histogram for Track A × E1 (primary), 1000 seeded runs, reproducible under seed 311
- [ ] Consensus table + figure at pair-quarter granularity, three buckets
- [ ] `make noise-test` green with all three pass criteria; `PASS_FAIL.md` in repo
- [ ] `docs/manual_trace.md`: 3/3 MATCH
- [ ] B13: the 12d factor-neutralization paragraph drafted with the row-effect numbers and Han et al. (2023) cited
- [ ] Every stochastic component in B code seeded from 311; two consecutive full runs produce bit-identical `results/` artifacts

---

## 5. Workstream C — Learning & Evaluation (Person C)

Person C owns everything from "a z-score series exists" to "a frozen, tuned model emits `p_hat` for a trigger": trigger detection and labeling (Step 7), the 7 features (Step 8), E1/E2/E3 (Step 9), splits/purging/embargo, metrics, calibration, and bootstrap CIs (Step 10), plus 12e analyses and gated Track D (spec 8.2). **Days 1–3 run entirely on fixtures** (`synth/fixture_zscores.py` output + its hand-computed golden triggers file); real data is touched for the first time at Integration Checkpoint 1 on Day 4. Every module below must run identically on fixture and real inputs — the track/source of the input is a parameter, never hardcoded.

Files Person C owns:

```
src/synth/fixture_zscores.py      # C0 (Day 1: planted-OU fixture + hand-computed golden triggers, per Section 2.4)
src/labeling/triggers.py          # C1
src/features/build_features.py    # C3
src/datasets/assemble.py          # C4
src/evaluation/splits.py          # C5
src/evaluation/metrics.py         # C10
src/evaluation/calibration.py     # C10
src/evaluation/bootstrap.py       # C10
src/models/common.py              # C6 (EntryModel API — consumed by B's runner)
src/models/e1_logistic.py         # C6
src/models/e2_gda.py              # C7
src/models/e3_mlp.py              # C8
src/models/freeze.py              # C13
src/analysis/base_rate.py         # C2
notebooks/c11_error_analysis.ipynb
tests/test_triggers.py  tests/test_splits.py  tests/test_features.py  tests/test_gda_vs_lda.py
```

---

### C0 — Planted-OU fixture + golden triggers (Day 1, ~2.5h)

**Consumes:** nothing (seeded RNG; schemas per Section 2.2/2.4).
**Produces:** `data/synth/fixture/*` (z-scores, spreads, **prices**, residuals, volume, `pc_1` series, pairs — including one pair tiling across two consecutive active windows for the run-semantics test — and stability for 6 fake pairs; calendar pinned to 2019-07-01 + 500 trading days so triggers span the train/val boundary) and the hand-computed `tests/golden/golden_triggers.csv` — derived by exporting the z-matrix and scanning it in a spreadsheet, never by running the labeling code it will test (Section 2.4).
**Done when:** fixture artifacts pass the schema contract tests and the golden file is committed with a header note naming who hand-verified it.

---

### C1 — Trigger detection and labeling (Day 1 PM – Day 2, ~6h)

**Consumes:** `data/spreads/zscores_{track}.parquet` (fixture version Day 1), `data/pairs/pairs_{track}.csv` (fixture), config constants.
**Produces:** the trigger/label core of `triggers_{track}` (columns `trigger_id, pair_id, source, trigger_date, z_trigger, label, horizon_end_date`) — features and split added by C3/C5.
**Done when:** `pytest tests/test_triggers.py` passes the golden-file test exactly (every expected trigger found, no extras, labels match).

```python
# src/labeling/triggers.py
def detect_triggers(
    zscores: pd.DataFrame,            # date x pair_id
    pairs: pd.DataFrame,              # pairs_{track}.csv incl. active_from/active_to
    z_entry: float = config.TRIGGER_Z,        # 2.0
    horizon: int = config.LABEL_HORIZON,      # 5 trading days
    reversion_frac: float = config.REVERSION_FRACTION,  # 0.5
) -> pd.DataFrame: ...
```

**Onset semantics (exact).** Consecutive tiling rows for the same `pair_id` are first merged into maximal **runs** — the identical construction A6 uses (windows tile, so a gap means the pair genuinely dropped out and re-entered). Active-window boundaries *within* a run are invisible to the trigger logic. A trigger then fires for pair *p* at date *t* iff:
1. both *t−1* and *t* lie inside the same **run** with non-NaN z on both days — so only a run's true first day can never trigger, not every 21-day refresh boundary (applying the test per pairs-row instead of per run would silently discard roughly a quarter of legitimate triggers);
2. `|z_{t-1}| < 2.0` and `|z_t| >= 2.0` (onset only, per spec 7a);
3. the pair is **re-armed** (below).

**Overlap policy (decide now, write in the docstring):** after a trigger at *t*, the pair is dis-armed. It re-arms only when **both** (a) the prior trigger's 5-day horizon has closed (`date > horizon_end_date`) **and** (b) `|z|` has printed a value `< 2.0` on some day at-or-after horizon close. A new trigger then requires a fresh below-to-above crossing. *Justification:* spec 7a exists precisely to prevent hundreds of near-duplicate overlapping examples; without condition (b), a spread that stays pinned above 2.0 through the horizon would immediately re-trigger on a trivial 1.99→2.01 wiggle, recreating the duplicate problem the onset rule was built to kill. One trigger = one independent widening episode.

**Label:** `label = 1` iff `min(|z_s|) <= reversion_frac * |z_trigger|` for *s* in the 5 trading days after *t* (i.e. *t+1…t+5*, pair-active days); `horizon_end_date = t+5` in trading days. `z_trigger` is stored **signed**. Triggers whose horizon would extend past the pair's **run end** (the `active_to` of the run's final row) or past the last data date are dropped with a logged count (they have undefined labels; report the count in the appendix, expected to be small). The drop applies only at true run ends — never at internal window boundaries.

`trigger_id = f"{pair_id}__{trigger_date:%Y%m%d}"` (unique because at most one trigger per pair per day).

**Test:** golden-file comparison against `tests/golden/golden_triggers.csv` (hand-derived per Section 2.4 — never produced by this code), plus unit cases for: crossing on a run's first day (no trigger), a trigger straddling an internal window boundary (fires — the fixture's tiling pair covers this), NaN gap at *t−1* (no trigger), pinned-above-2 re-arm suppression, exact-equality boundary (`|z_t| == 2.0` triggers; `|z| == 0.5*|z_trigger|` labels 1).

---

### C5 — Splits, purge, embargo (Day 2, ~3h)

**Consumes:** `trigger_date`, `horizon_end_date` columns + the trading calendar (index of `returns.parquet`; fixture calendar Days 1–3). Boundaries from `src/config.py` (train ≤ 2020-12-31, val 2021–2022, test 2023–2024).
**Produces:** the `split` column with values `train | val | test | purged | embargo`.
**Done when:** deterministic unit tests on synthetic dates prove both mechanisms (see below).

```python
# src/evaluation/splits.py
def assign_split(
    trigger_dates: pd.Series,
    horizon_end_dates: pd.Series,
    calendar: pd.DatetimeIndex,
) -> pd.Series: ...  # categorical: train/val/test/purged/embargo
```

Rules, applied in order:
1. Base assignment by `trigger_date` vs the chronological boundaries.
2. **Purge:** any observation whose `horizon_end_date` falls on or after the first trading day of the *next* split → `purged`. With a 5-day horizon this is exactly "the last 5 trading days of labels before each boundary" from the config freeze — but implement it via `horizon_end_date` comparison, not day-counting, so it stays correct if the horizon ever changes on validation (C2 escalation path).
3. **Embargo:** any observation whose `trigger_date` falls within the first 10 trading days of val or test → `embargo`.

Rows marked `purged`/`embargo` are **kept in the parquet, never silently dropped** — B's runner and C's model code filter on `split`, and the report can state exact counts per category (a leakage-audit line item, Part 4 items 5–6).

**Tests:** synthetic 30-day calendar with a boundary in the middle; assert (i) a trigger 3 days before the boundary with a 5-day horizon → `purged`; (ii) a trigger 6 days before → `train`; (iii) triggers on embargo days 1–10 of the later split → `embargo`, day 11 → `val`; (iv) idempotence and no row loss.

---

### C3 — The seven features (Day 2 PM – Day 3, ~5h)

**Consumes:** trigger rows (C1); `zscores_{track}`, `spreads_{track}`, `residuals_a` (or `residuals_d`), `factors_a`, `data/raw/volume.parquet`, `data/clusters/stability_{track}.parquet`, `pairs_{track}.csv`. Fixture versions of all of these exist Day 1.
**Produces:** the `f_*` columns of `triggers_{track}`, **raw (unstandardized)** — standardization happens inside model `fit` (below).
**Done when:** `tests/test_features.py` passes hand-computed values on the fixture for every feature, and an assertion sweep confirms no feature uses any value dated after `trigger_date`.

```python
# src/features/build_features.py
def build_features(
    triggers: pd.DataFrame, *,
    zscores: pd.DataFrame, spreads: pd.DataFrame,
    residuals: pd.DataFrame,          # residuals_a for tracks a/b/c; residuals_d for track d
    factors: pd.DataFrame,            # ALWAYS factors_a — pc_1 is the market factor for all tracks
    volume: pd.DataFrame,
    stability: pd.DataFrame, pairs: pd.DataFrame,
) -> pd.DataFrame: ...                # triggers + 7 f_ columns
```

Exact formulas — all quantities as of trigger date *t* or earlier (leakage rule, Step 8):

| Column | Formula |
|---|---|
| `f_abs_z` | `abs(z_t)` — i.e. `abs(z_trigger)` |
| `f_spread_vol_60d` | std of daily spread **changes** `Δspread_s = spread_s − spread_{s−1}` over the trailing 60 trading days ending at *t* (min 30 obs, else NaN→median-impute from train rows, logged) |
| `f_resid_mom_5d` | `sign(z_trigger) * Σ_{s=t-4..t} (resid_A_s − resid_B_s)`. **Sign convention: positive = the gap is still widening in the direction of the trigger**; negative = already narrowing. This makes the feature direction-invariant so one coefficient serves both long and short triggers. |
| `f_mkt_vol_20d` | std of `pc_1` column of `factors_a` over trailing 20 trading days ending at *t*. Used for **all** tracks — Track B/C/D have no own return-factor model, and "market volatility" is a property of the market, not the track. |
| `f_rel_volume_20d` | `0.5 * [ vol_A_t / SMA20(vol_A)_t + vol_B_t / SMA20(vol_B)_t ]`, SMA over trailing 20 days including *t* |
| `f_days_since_trigger` | trading days since this pair's previous trigger (per the C1 trigger stream), capped at **126**; pairs with no prior trigger get 126 |
| `f_cluster_stability` | the pair's `co_clustered` bool (as 0/1) from `stability_{track}` at the most recent `window_end <= trigger_date`; if the pair has no prior-window record (newly formed), **0** — "no evidence of persistence" is the conservative default, stated in the report |

**Standardization contract:** `StandardScaler` is **fit on train-split rows only** inside `EntryModel.fit`, applied unchanged to val/test at `predict_proba` time, and serialized with the model (C13). The parquet always holds raw features so B's runner and the error analysis see interpretable values.

**Methods-section note:** `f_spread_vol_60d` uses the std of daily spread *changes*, a deliberate sharpening of spec Step 8's literal "rolling std of spread" — the spread is a cumulative sum, so its level-std mostly measures drift, not jumpiness. Disclosed in one sentence in the report.

---

### C4 — Dataset assembly (Day 3 fixture, Day 4 real, ~2h + 2h)

**Consumes:** C1 triggers + C3 features + C5 splits, `pairs_{track}.csv` (for `source`).
**Produces:** `data/datasets/triggers_{track}.parquet`, exactly the Section-2 contract schema (all 7 `f_` columns, `label`, `horizon_end_date`, `split`).
**Done when:** schema-validation test passes (column names/dtypes exact, `trigger_id` unique, no NaN labels, split categories complete); fixture version Day 3 EOD; **real `triggers_a.parquet` is the Day 4 Integration Checkpoint 1 deliverable**, built live with A and B watching row counts.

```python
# src/datasets/assemble.py
def assemble_dataset(track: Literal["a","b","c","d"]) -> pd.DataFrame: ...
# CLI: python -m src.datasets.assemble --track a
```

One function, one track argument — building `triggers_b` (Day 4 PM, after A ships `zscores_b`) and later `triggers_c`/`triggers_d` is a re-run, not new code.

---

### C2 — Base-rate analysis (Day 4, ~2h)

**Consumes:** real `triggers_a.parquet` (and `triggers_b` when it lands).
**Produces:** `results/figures/base_rate_by_year.png` (the 12e base-rate-decay figure: reversion rate per calendar year 2016–2024 with trigger counts as bar annotations), plus a small table printed at the Day 4 sync: overall rate, by year, by track, by regime (calm vs stressed, split on median `f_mkt_vol_20d`).
**Done when:** the table is reviewed by all three at the Day 4 checkpoint and the decision row below is executed.

Decision table (spec 7c), pre-agreed:

| Observed train base rate | Action |
|---|---|
| 15% – 85% | Proceed; `class_weight='balanced'` (already in config) handles moderate skew |
| < 15% or > 85% | **Escalate at Day 4 sync.** Label parameters (reversion fraction, horizon) may be revisited **on train+validation only**, never test, with a dated DECISIONS.md entry recording old→new values and the observed rate that forced it |
| > ~90% | Also flag as a possible bug (label logic or z-score construction) before touching parameters |

---

### C6 — Model API + E1 logistic regression (Day 3 fixture, Day 4 real, ~4h)

**Consumes:** `triggers_{track}.parquet`. **Produces:** `src/models/common.py` (the interface B's runner imports — this is the C↔B contract), `src/models/e1_logistic.py`, and the E1 coefficient table `results/tables/e1_coefficients_{track}.csv`.
**Done when:** E1 fits on fixture data with sane output (fixture AUC well above 0.5, since fixtures have planted OU signal); first real fit on `triggers_a` train rows happens Day 4 at Checkpoint 1; B's runner can round-trip `fit → tune → predict_proba` without touching model internals.

```python
# src/models/common.py
class EntryModel(abc.ABC):
    name: str                                   # "e1" | "e2" | "e3"
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "EntryModel": ...
        # fits StandardScaler on X_train, then the model; returns self
    def tune(self, X_val: pd.DataFrame, y_val: pd.Series) -> dict: ...
        # hyperparameter selection by validation AUC; refits on train at best config;
        # returns {"best_params": ..., "val_auc_grid": ...}
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...   # shape (n,), P(label=1)
    def get_params_report(self) -> dict: ...    # everything the report table needs
    def save(self, path: pathlib.Path) -> None: ...
    @classmethod
    def load(cls, path: pathlib.Path) -> "EntryModel": ...

FEATURES: list[str] = ["f_abs_z", "f_spread_vol_60d", "f_resid_mom_5d", "f_mkt_vol_20d",
                       "f_rel_volume_20d", "f_days_since_trigger", "f_cluster_stability"]
```

B's runner consumes only this API plus τ (C9) to produce `results/decisions_{track}_{model}.parquet` (`enter = p_hat > tau`); interface named, moving on.

**E1:** `sklearn.linear_model.LogisticRegression(penalty="l2", class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=config.SEED)`; `C ∈ {0.01, 0.1, 1, 10}` chosen by validation AUC in `tune`. **Coefficient deliverable:** table of standardized coefficients (features are scaled, so magnitudes are comparable) with feature names, per track — this is a headline report table ("market volatility carries a negative weight…" per Step 9).

---

### C7 — E2 Gaussian Discriminant Analysis (Day 3 PM – Day 4, ~3h)

**Consumes/Produces:** same dataset; `src/models/e2_gda.py` behind the same `EntryModel` API.
**Done when:** `tests/test_gda_vs_lda.py` passes: our numpy predict_proba matches `sklearn LinearDiscriminantAnalysis.predict_proba` to `1e-6` on fixture data.

**Decision, stated now:** primary E2 = **GDA with shared covariance** (mathematically LDA), implemented **directly in numpy** — class priors `φ`, class means `μ_0, μ_1`, pooled covariance `Σ`, posterior via Bayes' rule with the Gaussian class-conditionals. It is ~30 lines and is the course-aligned choice (this is exactly the GDA derivation from lecture); the sklearn cross-check makes it safe. `sklearn QuadraticDiscriminantAnalysis` (per-class covariances, small `reg_param` for stability on 7 features) runs as a one-line **robustness row** in the report, not a grid cell. GDA has no hyperparameters, so `tune` is a no-op returning val AUC; class imbalance is handled naturally through the priors (report both empirical-prior and balanced-prior posteriors if the base rate is skewed; empirical is primary). Report framing: discriminative-vs-generative — does the Gaussian assumption on these 7 features help or hurt vs E1?

---

### C8 — E3 small MLP (Day 5 – Day 6, ~5h)

**Consumes/Produces:** same dataset; `src/models/e3_mlp.py` behind the same API.
**Done when:** tuned E3 reproduces byte-identical `p_hat` across two runs (seeding verified) and its grid row runs through B's engine on validation.

Architecture and training, fixed a priori:
- `7 → h → 1`, `h ∈ {8, 16}`, ReLU hidden, sigmoid output (implemented as `BCEWithLogitsLoss` on the raw logit, with `pos_weight = n_neg/n_pos` — the `class_weight='balanced'` analog).
- Adam, lr `1e-3`, full-batch (a few hundred–low thousands of triggers; batching is pointless), weight decay grid `{1e-4, 1e-3, 1e-2}` → 6 configs total.
- Early stopping on **validation AUC**, patience 25 epochs, `max_epochs = 500`; keep the best-val-AUC checkpoint.
- Seeding: `torch.manual_seed(config.SEED)`, `numpy` and `random` likewise, deterministic algorithms on; **CPU only** — a 7→16→1 net on ~1–2k rows trains in **under 10 seconds per config, under 2 minutes for the full grid**; anyone waiting longer has a bug.

**Expected result (spec Step 9): E3 does not beat E1.** The deliverable is not "a neural net" but the **bias-variance comparison** — the E0→E1→E3 ladder on real noisy data, reported with overlapping CIs if that's what we get. Keep it small on purpose; do not add layers to "fix" a null result.

---

### C9 — τ selection: the one pre-registered rule (Day 5, ~1.5h)

**Consumes:** validation `p_hat` from each fitted model + B's engine output at c=10bps.
**Produces:** `results/frozen/taus.json` (`{track: {model: tau}}`) and a dated DECISIONS.md entry.
**Done when:** the rule below is written into DECISIONS.md **before** any test-set run exists; E1/E2 τ values recorded Day 5 evening, E3's (and any extension track's) recorded Day 6 before the C13 freeze.

**The rule (pre-registered, exactly one):** for each (track, model), τ is chosen from the grid `{0.40, 0.45, …, 0.80}` to **maximize validation net P&L at c = 10 bps, subject to ≥ 25 accepted validation trades**; ties (including P&L ties within $1e-6) break toward the **higher** τ (prefer trading less when indifferent). If no τ on the grid meets the 25-trade floor, choose the τ that **maximizes accepted validation trades** (ties toward higher τ); if even that maximum is 0 trades, the cell is reported as **degenerate** with τ = 0.5 and a DECISIONS.md note — the rule has no undefined branch. τ is chosen **per model per track**, on validation only, and is never revisited after the **C13 freeze** — the Day 7 test run loads it from `taus.json` read-only. Timing: E1/E2 τ are selected Day 5; **E3's τ is selected on Day 6 immediately after C8 finalization** (freezing a τ tuned against a provisional model would undermine the pre-registration claim), before the freeze.

Mechanics: C computes decisions tables for each candidate τ; B's engine (interface: `decisions → trades → net P&L`) scores them; C picks and records. E0 has no τ (`p_hat = NaN`, `enter = True` always).

---

### C10 — Metrics, calibration, bootstrap (Days 3–6, ~6h total)

**Consumes:** `triggers_{track}` + `decisions_*` + `results/trades_{track}_{model}.parquet` (B's engine output).
**Produces:** the classification half of `results/metrics_{track}_{model}.json`; `results/figures/calibration_{track}_{model}.png`; bootstrap CI machinery used by both C's and B's metrics.
**Done when:** metrics run on fixture outputs Day 3; calibration + bootstrap wired Day 5–6; every reported number in the JSON carries a CI.

```python
# src/evaluation/metrics.py
def classification_metrics(y_true: np.ndarray, p_hat: np.ndarray, tau: float) -> dict:
    ...  # {"auc": ..., "precision_at_tau": ..., "recall_at_tau": ..., "brier": ..., "n": ...}

# src/evaluation/calibration.py
def reliability_diagram(y_true, p_hat, n_bins: int = 10, out_path: Path) -> pd.DataFrame:
    ...  # 10 QUANTILE bins (equal-count, not equal-width — our p_hat range is narrow);
         # plot predicted vs observed rate with per-bin counts annotated; returns the bin table

# src/evaluation/bootstrap.py
def bootstrap_ci(values_or_frame, stat_fn: Callable, n_boot: int = 1000,
                 seed: int = config.SEED) -> tuple[float, float, float]:
    ...  # (point, lo, hi) — percentile CIs at 2.5/97.5
```

Bootstrap protocol: **classification metrics resample TRIGGERS i.i.d.** (rows of the trigger table); **strategy metrics resample TRADES i.i.d.** (rows of B's trade ledger — B calls the same `bootstrap_ci`). 1000 resamples, seed 311, percentile intervals. **Independence caveat, in the `bootstrap.py` docstring verbatim and as a limitations bullet:** trades overlap in time and cluster by pair and regime, so i.i.d. resampling understates true uncertainty; a block bootstrap would be more faithful but is out of scope — CIs are therefore *optimistic lower bounds on width*. AUC is the primary classification metric (spec 10d); accuracy is never reported. Sanity anchor: expected AUC 0.52–0.58; **anything ≥ 0.65 is treated as a bug report** (spec 10e) and triggers a leakage re-check before anyone celebrates.

All figures land in `results/figures/` with the naming pattern `{figure}_{track}_{model}.png`.

---

### C11 — Analysis extras (Day 6–7, ~4h; spec 12e)

**Consumes:** fitted E1 models per track, trigger tables, trade ledgers, `stability_{track}`.
**Produces / Done when:**
- **Coefficient comparison (~1h):** one table, tracks as columns, standardized E1 coefficients as rows + one paragraph: does the model learn the same physics under both selection methods (e.g. is `f_mkt_vol_20d` negative in both)? Sign agreement matters more than magnitude. → `results/tables/e1_coefficient_comparison.csv`, done when the paragraph is in the report draft.
- **Error analysis (~2.5h):** `notebooks/c11_error_analysis.ipynb` — the **top-20 highest-`p_hat` false positives** on validation (test after freeze only). Cross-tabulate against: year (crisis clustering — 2021–22 drawdown?), sector (from the universe table), `f_cluster_stability` (unstable pairs?), `f_mkt_vol_20d` quartile. Exported as a short markdown note for the report; done when three concrete observations are written down (even "no pattern visible" counts).
- **Calibration commentary (~0.5h):** one paragraph reading the reliability diagrams — over/under-confidence, and where E3's calibration falls apart relative to E1 if it does.

---

### C12 — Track D: autoencoder residuals (Day 6, gated; 12–15h — spec 8.2)

**Gate (spec 8.0, Day 6 morning sync):** Track C resolved (built or deliberately skipped), Day 7 results-freeze still realistic, **and C's E-models done** (C6–C8 green, C10 wired). **The honest go/no-go: if the Day 6 morning status board shows ANY red item — in anyone's workstream, not just C's — Track D is cut**, gets its one pre-written sentence in the lit review (spec 8.4), and Day 6 proceeds as planned above. Note the arithmetic plainly at the sync: 12–15h against a ~9h day means a "go" converts C's entire Day 6 into Track D and pushes C11 to Day 7 — that is only acceptable if C13's freeze can still happen Day 6 EOD (frozen set then includes the Track D grid row's models). Spelled all the way out: a legitimate GO additionally requires **C8 to have finished Day 5 EOD** (ahead of the plan's own default, which finalizes E3 Day 6 AM) and **~3h of Person A's Day 6 PM free** for the A4–A6 integration run — which collides with Track C if both gates pass. **In practice, Track D runs only if Days 1–5 landed with zero slippage AND Track C was skipped or trivially fast; the honest default is NO-GO**, the 8.4 sentence is pre-written, and if a GO later runs out of road, the committed fallback is to ship only the representation-stage comparison (Baldi–Hornik check + `residuals_d` + cluster/pair overlap vs Track A) on validation, clearly labeled, with no frozen test-run grid row.

Subtasks if it runs:
1. **(i) Linear AE ≡ PCA check FIRST (~3h).** PyTorch linear autoencoder (no activations), 40 → m → 40, m matched to that window's PCA component count; train on one 252-day window of standardized returns; verify reconstruction error matches Step-2 PCA's rank-m reconstruction to near-equality (Baldi–Hornik). **If they don't match, stop — the bug is in the AE or in A's PCA, and finding which is the deliverable.** This check goes in the report either way.
2. **(ii) Nonlinear AE → residuals (~6h).** 40 → hidden → bottleneck (3–5, matched to PCA m) → hidden → 40, one hidden layer each side, MSE, weight decay, early stopping on a held-out slice of the training window, inputs standardized window-locally (leakage item 12). Retrain **monthly** on trailing 252d, apply forward, residual = return − reconstruction, **strictly out-of-sample** (never scores a day it trained on). → `data/processed/residuals_d.parquet` (same shape contract as `residuals_a`).
3. **(iii) Integration (~3h).** Hand `residuals_d` to Person A, who runs the A4–A6 machinery with `track d` tags → `pairs_d`, `zscores_d`, `stability_d`; C re-runs `assemble_dataset("d")` → `triggers_d.parquet` (using `residuals_d` for `f_resid_mom_5d`, `factors_a` for `f_mkt_vol_20d` as always); grid row runs through B's runner. Track D must pass the noise test before its results are reported (Part 4).

Expected result: **does not beat PCA** (spec 8.2) — the deliverable is the linear-vs-nonlinear representation comparison mirroring E1-vs-E3.

---

### C13 — Model freeze protocol (Day 6 EOD, ~1.5h)

**Consumes:** every tuned model, its fitted scaler, and `taus.json`.
**Produces:** `results/frozen/` containing, per (track, model): `e1_{track}.joblib` / `e2_{track}.joblib` (model + scaler bundled in one object via `EntryModel.save`), `e3_{track}.pt` (state_dict + scaler params + architecture config in one torch archive), plus `taus.json`; a `sha256sum results/frozen/* >> DECISIONS.md` block with timestamp and all three names.
**Done when:** a clean-process smoke test (`EntryModel.load` each file, `predict_proba` on 5 validation rows, byte-identical to pre-freeze output) passes, and the hashes are in DECISIONS.md. On a Track C go, the freeze moves to **late Day 6 evening** (after the sync) so the track-c models and τ_c make the frozen set — see A8; C's ~2h of track-c assembly/fits/τ_c displaces the C11 error-analysis start to Day 7. **The Day 7 witnessed test run loads ONLY these artifacts** — it calls `load`, never `fit` or `tune`; any code path in the test-run script that could reach `fit` raises. If the frozen hash doesn't match at Day 7 load time, the run aborts.

---

### Person C — day-by-day hour budget

| Day | Hours | Contents |
|---|---|---|
| 1 (Fri) | ~7 | Kickoff 2h; C0 fixture + golden triggers file 2.5h; C1 trigger detection v1 on fixture z-scores 2.5h |
| 2 (Sat) | ~8 | C1 golden test green 2h; C5 splits + tests 3h; C3 features (first 4) 3h |
| 3 (Sun) | ~9 | C3 finish + tests 2h; C4 fixture assembly 1.5h; C6 API + E1 on fixtures 3h; C10 metrics v1 1.5h; C7 GDA start 1h |
| 4 (Mon) | ~8 | **Checkpoint 1:** C4 real `triggers_a` 2h; C2 base rate + decision table 2h; C6 first real E1 fit + coefficients 1.5h; C7 GDA finish + 1e-6 verification 2.5h; `triggers_b` assembly when `zscores_b` lands (re-run, ~0h marginal) |
| 5 (Tue) | ~8.5 | C8 MLP build + tune 4h; C9 τ selection + DECISIONS.md 1.5h; C10 calibration 2h; grid-run support + leakage checklist review + **Track C gate sync** 1h |
| 6 (Wed) | ~8.5 | **Track D gate sync** (go → C12 consumes the day, C11→Day 7); no-go path: C8 finalize + determinism check 1.5h, C10 bootstrap final 2h, C11 coefficient comparison + error analysis start 3h, **C13 freeze + hashes 1.5h**, figures 0.5h |
| 7 (Thu) | ~7 | Witnessed test run (present, loading frozen artifacts only) 1.5h; C10/C2 final figures 1.5h; C11 finish 2h; drafting Steps 7–10 + 12e report sections 2h |
| 8 (Fri) | ~6 | Writing: models/evaluation/limitations sections, CI phrasing audit (every number has an interval), full-team read-through, submit |

---

### Workstream C — definition of done

- [ ] Golden-file trigger test green on fixtures; onset + re-arm policy documented in the `detect_triggers` docstring
- [ ] Split tests prove purge and embargo on synthetic dates; purged/embargo rows retained and countable
- [ ] All 7 features match hand-computed fixture values; no feature reads past `trigger_date`; scaler fit on train rows only and serialized with each model
- [ ] `triggers_a` (Day 4) and `triggers_b` (Day 4 PM) conform exactly to the Section-2 contract schema
- [ ] Base-rate table reviewed at Day 4 sync; decision-table row executed and (if triggered) logged in DECISIONS.md
- [ ] E1, E2, E3 all behind the `EntryModel` API and runnable by B's runner without modification; GDA matches sklearn LDA to 1e-6; E3 byte-reproducible under seed 311
- [ ] τ rule pre-registered in DECISIONS.md Day 5, before any test-set contact; `taus.json` complete for every grid cell
- [ ] Every reported metric carries a 1000-resample percentile CI (triggers for classification, trades for strategy); independence caveat in docstring + limitations
- [ ] Calibration diagrams (10 quantile bins, counts shown) for every model×track in `results/figures/`
- [ ] Base-rate-decay figure, coefficient-comparison table, error-analysis note delivered
- [ ] Any AUC ≥ 0.65 investigated as a bug before being reported
- [ ] `results/frozen/` complete Day 6 EOD, sha256 in DECISIONS.md, load-only smoke test green; Day 7 test run touched no `fit`/`tune` path
- [ ] Track D either shipped with the Baldi–Hornik check documented, or cut cleanly with its lit-review sentence — no half-built state in the repo at freeze

---

## 6. Timeline, coordination, and risk

This section is the operating schedule for the plan in Sections 3–5. Task IDs (A0…A8, B1…B12, C0…C13) refer to the per-person task lists in those sections; every reference below also names the task. All times EST. Artifact names are the canonical contracts from Section 2; a **bold** entry means that artifact lands (is merged to `main`) that block.

### 6.1 Day-by-day plan

Conventions: AM ≈ before 2pm, PM ≈ after 2pm through the 9pm sync. "Consumes/Produces" for each task is specified in Sections 3–5; here we list only the landings. Days 2–3 are Saturday/Sunday — **each person's weekend availability is confirmed verbally at the Day 1 kickoff and recorded in `DECISIONS.md`, against the actual plan, not a bare floor**: Day 2 runs ≈ 7–8.5h and Day 3 (Sunday) ≈ 7.5–9h per person, and Day 3 is the schedule's critical day (`zscores_a` and `pairs_b` both land EOD). If anyone cannot commit to those numbers, the coverage rules in the risk register (6.4, risk R8) apply immediately, not reactively.

#### Day 1 — Friday, July 31

| | Person A | Person B | Person C |
|---|---|---|---|
| **AM** | 2h all-hands kickoff (all three): ratify universe (40 tickers, 4 sectors), map real names→roles by skill, freeze `src/config.py`, walk through repo skeleton + artifact contracts (Section 2), fix daily sync time, create Overleaf project + assign lit-review owner, action item: re-read course handout for required deliverables. Then A0: writes **`synth/make_synthetic.py`** (random-walk fixture prices — later reused verbatim as the Day 5 noise-test input). | Kickoff. Then B1 (universe ratified) + B2 (`yfinance` download): **`data/raw/prices.parquet`, `data/raw/volume.parquet`, `data/raw/spy.parquet`**. | Kickoff. Then C0: writes **`synth/fixture_zscores.py`** (planted-OU fake pairs + fake volume/factor/stability series) and begins the hand-computed golden triggers file. |
| **PM** | A1 (rolling PCA engine, Step 2: window loop, in-window standardization, `eigh`, sign fix, component-count rule) developed and unit-tested against fixture prices from `synth/make_synthetic.py`. Definition of done: PCA runs end-to-end on synthetic prices, all window-boundary unit tests green. | B2 cleaning checks (missing-day threshold, forward-fill, calendar align, return-outlier scan) + log returns: **`data/processed/returns.parquet`** by EOD. Commits `src/contracts.py` v1 + `src/io_utils.py` alongside `src/config.py`. | Golden triggers file finished and committed. C1 begun (trigger detection, Step 7a) against fixture z-scores. Done when the golden-file test asserts exact expected trigger dates. |
| Hours | A: 7 (2 kickoff + 5) | B: 8 (2 + 6) | C: 7 (2 + 5) |

EOD state (anchor): repo skeleton + both fixture generators + real `returns.parquet` on `main`.

#### Day 2 — Saturday, August 1 (weekend)

| | Person A | Person B | Person C |
|---|---|---|---|
| **AM** | A1 rerun/validated on real `returns.parquet`; A2 (per-stock OLS betas + out-of-sample residuals, Step 3) begun. Writes the shared future-perturbation leakage helper (`tests/leakage_utils.py`). | B3 (Bloomberg pull, 4B.1): full 40 tickers × 19 fields × ~40 quarters in one terminal session; raw CSVs committed under `data/raw/characteristics/`. Done when row counts and field coverage logged. **Bloomberg access itself was verified before Day 1** (see risk R1). | C1 golden-file test green. C5 (chronological splits + purge-5d + embargo-10d, Step 10a–c), tested on synthetic dates spanning both boundaries. |
| **PM** | A2 finished (residuals) + A3 sanity checks (scree, PC1 loadings ≥ 0, PC1-vs-SPY overlay, variance-over-time incl. 2020 spike): **`data/processed/factors_a.parquet`, `pca_meta.parquet`, `loadings_a.parquet`, `residuals_a.parquet`** by EOD. A5 split algorithm + unit tests (fixture-only, 1.5h — shortens Sunday's serial chain). Done when PC1-vs-SPY correlation is high and the four sanity figures are posted to the sync channel. | B5 (backtest engine v1, Step 9/11 execution rules: t+1 entry, exit on \|z\|<0.5 or 5 days, $1/leg, raw-return P&L, per-transaction costs) developed against `synth/fixture_zscores.py` output + a hand-written fixture decisions table. Done when a hand-computed 3-trade fixture ledger matches engine output to the cent. | C3 (the 7 features, Step 8) begun on fixtures: `build_features(...)` emitting the first `f_*` columns, with the look-ahead unit-test pattern in place (shifting future data must not change features). |
| Hours | A: 8 | B: 8 | C: 7 |

#### Day 3 — Sunday, August 2 (weekend)

| | Person A | Person B | Person C |
|---|---|---|---|
| **AM** | A4 (Track A k-means on formation-window factor betas, silhouette-chosen k∈8–13, co-membership stability): **`data/clusters/labels_a.parquet`, `stability_a.parquet`**. | B4 (Track B cleaning, 4B.2: sentiment merge, log size, z-score, drop sparse columns + median-impute; then characteristics PCA, 4B.3–4B.5 Route B). | C3 finished + C4 (fixture dataset assembly → the `triggers` contract schema) + C6 (EntryModel API + E1 logistic: L2, `class_weight='balanced'`, seed 311) on the fixture triggers table. |
| **PM** | A5 final wiring + `v1-frozen` tag (algorithm + tests landed Day 2; Zhang rules — drop singletons, 2–4 whole, 5+ greedy split; consumed by B for Track B): **`data/pairs/pairs_a.csv`**. A6 (spread + z-score, Step 6 simple version, run/burn-in policy, Z_WINDOW=60): **`data/spreads/spreads_a.parquet`, `zscores_a.parquet`** by EOD. | B4 done (k-means on component scores, k∈10–13) → calls A's pair-builder: **`data/clusters/labels_b.parquet`, `stability_b.parquet`, `data/pairs/pairs_b.csv`** by EOD. Characteristics-PC interpretation table (4B.4) drafted. | C10 (classification metrics module v1: AUC, precision/recall at τ) + C7 (E2 GDA begun) on fixtures. Done when E1 produces a valid `decisions` table from fixture triggers and the metrics JSON schema validates. |
| Hours | A: 8 | B: 8 | C: 7 |

#### Day 4 — Monday, August 3 — **Integration Checkpoint 1**

| | Person A | Person B | Person C |
|---|---|---|---|
| **AM** | A6 extended to Track B: **`data/spreads/spreads_b.parquet`, `zscores_b.parquet`** (Day 4 AM per contract). | B5 real-data integration (cost columns across the full c-grid land in the ledger) + B6 (E0 decisions through the engine). B assists checkpoint (below). | C4 assembly (C1 triggers + C3 features + C5 splits) on **real** `zscores_a.parquet`: **`data/datasets/triggers_a.parquet`** with real labels, features, and split column. |
| **PM** | **Checkpoint 1 (all three, ~1h, screen-shared):** entry criteria met → real triggers built; base rate reviewed together; first real E1 fit on train. Then A7 (leakage-audit items 1–4: window-slice tests, standardization audit, beta-timing audit) + schedule buffer / pairing with C if checkpoint slipped. | B7 (runner `src/experiments/run_grid.py` skeleton: for each (track, model) → C's model API → `results/decisions_{track}_{model}.parquet` → engine → trades + metrics; E0 = enter-always decisions through the same path; the guarded `--split test` path). Commits the manual-trace template (B12 prep). | C2 base-rate analysis (7c) presented at checkpoint, incl. the base-rate-by-year draft figure; first real E1 fit on train split (C6). **`data/datasets/triggers_b.parquet`** by EOD once `zscores_b` lands. |
| Hours | A: 7 | B: 8 | C: 8 |

#### Day 5 — Tuesday, August 4 — grid on train+validation; **Track C gate at evening sync**

| | Person A | Person B | Person C |
|---|---|---|---|
| **AM** | A7 continued: leakage checklist items owned by A verified with written evidence (checklist lives in `docs/leakage_checklist.md`). Support grid debugging. | B7 grid runner completed and the **full 2×4 grid runs on train+validation** in the AM — its outputs must exist before the noise test and the evening gate. | E1/E2 tuned on validation via `EntryModel.tune` (C6/C7); C9 τ selection for E1/E2 recorded in DECISIONS.md; C8 (E3 small MLP) first training runs. |
| **PM** | Smell test (spec 10e) applied to every cell as a team over the morning's outputs: **`results/decisions_*`, `trades_*`, `metrics_{a,b}_{e0,e1,e2,e3}.json` (train+val)**. | B11 (noise test, on the now-complete pipeline; binding criteria per Section 4 = net-of-cost + AUC-vs-mechanical-baseline; covers E0/E1/E2, with the E3 row rerun Day 6 pre-freeze). B12 (manual trace — hard gate criterion, may not slip). B8 breakeven table (0.5h) for the sync. | C8 provisional numbers into the grid (E3 finalized Day 6; its τ selected then, per C9). |
| **Sync** | **9pm sync = grid-on-validation checkpoint + Track C gate** (see 6.2). | | |
| Hours | A: 7 | B: 8 | C: 8 |

#### Day 6 — Wednesday, August 5 — **Track D gate at morning sync**; figures begin

| | Person A | Person B | Person C |
|---|---|---|---|
| **AM** | **Morning sync = Track D gate.** If Track C GO: A8 (partial-correlation distance, spec 8.1: shrinkage λ=1e-3, precision→partial-corr→distance, same k-means/pair-builder; ~5h) → `labels_c`, `stability_c`, `pairs_c`, `spreads_c`, `zscores_c`. If NO-GO: A starts methods figures (scree, PC1-vs-SPY, variance-over-time, cluster-stability). | B8 cost-sweep figure final (AM). B9 (turnover-matched control: per-quarter trigger counts matched to E1, random draws seeded, through the same engine; 1000 seeded draws → histogram). | C8 done (E3 finalized, early stopping, weight decay). C10 calibration/reliability diagrams. If Track D GO (default NO-GO): C12 autoencoder work begins with hard stop Day 6 EOD for integration hand-off to A. |
| **PM** | Track C triggers/decisions/metrics on train+val if running; else figure production. Lit-review paragraphs for A&L/Zhang methods. | B10 (consensus pairs, 12c: intersection buckets, reversion/AUC/net-performance per bucket — designated slip-to-Day-7-AM candidate). E3 row of the noise-test criteria rerun before the freeze. | C10 bootstrap CIs on all metrics (seed 311). C11 begun (E1 coefficient table + cross-track comparison; error analysis on confident false positives). **C13: model freeze** — frozen model objects + hyperparameters + τ serialized to `results/frozen/`, sha256 in DECISIONS.md. |
| Hours | A: 8 | B: 8 | C: 9 |

Report: skeleton + intro/lit-review drafting continues in Overleaf (owner per 6.3) — this depends on no results and must not slip.

#### Day 7 — Thursday, August 6 — **test-set run (morning); results freeze (evening)**

| | Person A | Person B | Person C |
|---|---|---|---|
| **AM** | **The single, witnessed test-set run** (all three present, screen-shared; protocol in 6.2): B drives `run_grid.py --split test` with C13 frozen configs; A and C witness. **`results/metrics_*_*.json` (test), `results/trades_*` (test)**. | Drives the run. Then final grid results table with CIs + marginal means; B13 (the 12d factor-neutralization paragraph, Han et al. 2023) in the PM. | Witnesses. Then final calibration + bootstrap on test outputs; coefficient/error-analysis writeups. |
| **PM** | All figures final and committed (`results/figures/final/`, source scripts in `scripts/figures/`). Methods sections (PCA/residuals/clustering/spread) drafted. | Data/backtest/costs/experiment-design sections drafted. Turnover-control histogram + consensus figure final. | Labels/features/models/evaluation sections drafted. |
| **Sync** | **9pm sync = results freeze** (definition in 6.2). | | |
| Hours | A: 8 | B: 8 | C: 8 |

#### Day 8 — Friday, August 7 — writing and submission (deadline 11pm EST)

| | Person A | Person B | Person C |
|---|---|---|---|
| **AM** | Methods polish; limitations assembled from `limitations.md`; leakage checklist formatted into report. | Results + discussion writing; grid table + all numbers cross-checked against frozen JSONs. | Results + discussion writing; abstract + contribution statement; handout-required deliverables checklist. |
| **PM** | **~3pm: full-team read-aloud pass** of the entire report (all three, ~2h). Fix list executed immediately after. **~7pm: PDF frozen. Code packaged per handout format. Submitted by 9pm EST** (2h buffer before the 11pm deadline). | | |
| Hours | A: 6 | B: 6 | C: 7 |

### 6.2 Sync points and gates

#### Daily evening sync — 9:00pm EST, 30 minutes, every day

Fixed 4-item agenda, same order every night, no other business until these are done:

1. **Artifact status vs plan** — each person states which contracted artifacts landed vs the table in 6.1, by name.
2. **Blockers** — anything preventing tomorrow's committed landings.
3. **Contract-change requests** — any proposed change to `src/config.py` or an artifact schema requires all-3 agreement *at this meeting* and a line in `DECISIONS.md` (date, change, reason, who agreed). No silent contract changes, ever.
4. **Next-day commitments** — each person names tomorrow's landings out loud.

Merges to `main` happen at (or immediately after) this sync: feature branches are reviewed by one other person and merged with tests green. Between syncs, `main` is always in a state where the fixtures-based test suite passes.

#### Integration Checkpoint 1 — Day 4 (Mon Aug 3), PM

- **Entry criteria:** `zscores_a.parquet` landed (A); C's labeling/feature golden tests green on `synth/fixture_zscores.py` fixtures.
- **Exit criteria:** (i) real `data/datasets/triggers_a.parquet` exists and validates against the contract schema; (ii) the base rate has been computed and reviewed *together* and falls inside C2's decision-table range (15–85%; outside that, the Section 5 escalation protocol is invoked before proceeding); (iii) E1 trains end-to-end on the train split and emits a valid decisions table.
- **If exit criteria are not met by Day 4 EOD:** risk R7 response activates (A pauses new feature work and pairs with C; Track C and D are auto-cut).

#### Grid-on-validation checkpoint — Day 5 (Tue Aug 4), evening sync

Passes only if all of the following hold:

- All 8 `results/metrics_{a,b}_{e0,e1,e2,e3}.json` files exist for train+validation.
- The spec 10e smell test is applied to **every** cell, together: AUC expected in the 0.52–0.58 band (materially above ⇒ treat as a bug report, not a result); Sharpe above 2 ⇒ suspect a bug; above 3 ⇒ almost certainly leakage. Any smelly cell gets a named investigator and blocks the Track C gate.
- The **noise test** passed on its binding criteria (net-of-cost ≤ 0 within CI on noise; no model AUC significantly above the mechanical `f_abs_z`-only baseline — raw AUC-vs-0.5 is advisory, see Section 4 B11). Day 5 coverage is E0/E1/E2; the E3 row of the criteria reruns Day 6 before the freeze.
- The **manual trace** passed (three dates hand-traced: decision from day-*t* signal, entry at the *t+1* close, first P&L day *t+2*, exit fill at the exit-signal close — the disclosed convention in Section 4 B5).
- The 12-item leakage checklist (Part 4) walked through as a team, each item marked with its evidence.

#### Track C gate (Day 5 evening sync) and Track D gate (Day 6 morning sync)

Quoting spec 8.0 verbatim:

> | Extension | Gate | Decision day |
> |---|---|---|
> | **Track C — partial-correlation distance (Rotondi)** | The full 2×4 pipeline runs end-to-end on real data with the leakage audit and noise test passed | **Day 5 evening sync** |
> | **Track D — autoencoder residuals (Krause)** | Track C (or a deliberate decision to skip it) is resolved, results-freeze on Day 7 is still realistic, AND Person C's E-models are done | **Day 6 morning sync** |

**The default decision is NO-GO.** A gate passes only if *every* criterion is unambiguously green, agreed by all three; any "probably fine" is a NO-GO. If only one extension can run, it is **Track C** (spec priority: "Track C first" — roughly one-third the cost, adds a full grid row). A NO-GO costs nothing: per spec 8.4, the skipped track gets one honest sentence in the literature review.

Two operational riders (from Sections 3 and 5): on a Track C **go**, A front-loads so `zscores_c` lands by Day 6 **noon**, C spends ~2h Day 6 PM on `triggers_c` + model fits + τ_c, and the C13 freeze moves to late Day 6 evening — if the noon hand-off slips, Track C is reported on validation only and stays out of the frozen test run. For Track D, a legitimate go additionally requires C8 (E3) finished by Day 5 EOD and ~3h of A's Day 6 PM free for integration — see C12's arithmetic; the honest reading is that Track D runs only on a zero-slippage week, and its committed fallback is the representation-stage comparison on validation only.

#### Results freeze — Day 7 (Thu Aug 6), evening sync

**Frozen at this sync:** all `results/metrics_*.json` (train, val, and test), all `results/trades_*.parquet` ledgers, all figures in `results/figures/final/` and their source scripts (`scripts/figures/`), the grid results table, and `DECISIONS.md`. The freeze is a signed git tag (`results-freeze`).

**Post-freeze rule:** any code change that could alter a frozen number requires (i) a documented, reproducible bug written into `DECISIONS.md`, (ii) sign-off from all three, and (iii) a rerun of the noise test before the corrected numbers replace the frozen ones. Cosmetic report edits are unrestricted; number-changing edits are not.

#### Test-set protocol — Day 7 (Thu Aug 6), morning

- The test run happens **exactly once**, with all three present on a screen-share, from the C13-frozen models and τ, via the runner's guarded path (`run_grid.py --split test` requires an explicit `--i-am-sure` flag and refuses to run twice — it checks for existing test outputs).
- The results are what they are. Spec, quoted: *"You are not being marked on how good the results are."* Nobody proposes a "quick re-check" after seeing test numbers.
- If a genuine bug is discovered after the run, the fix and the **single** rerun are documented in the report's limitations section, with both the before and after noted. Honesty over cosmetics.

### 6.3 Report plan

**Overleaf project created at Day 1 kickoff.** Default owner Person C; overridden at kickoff if another member is clearly the strongest writer (kickoff decision, recorded in `DECISIONS.md`).

**Section ownership mirrors code ownership:**

| Report section | Owner |
|---|---|
| Abstract, intro, lit review (incl. spec 2.5–2.7 citations and the honest novelty correction) | Kickoff-chosen owner; drafted Days 5–6 — depends on no results |
| Data, universe, survivorship-bias argument | B |
| Methods: rolling PCA, residuals, clustering, stability, spread/z-score | A |
| Methods: Track B characteristics pipeline; backtest engine; cost model; experiment design incl. turnover-matched control + consensus pairs | B |
| Labels, features, models E0–E3, τ rule, splits/purge/embargo, metrics, calibration, bootstrap | C |
| Limitations | Assembled by A from `docs/limitations.md`, **seeded on Day 1 with every spec Part 7 bullet as an owner-tagged stub** (short-sale constraints/borrow costs → B; stale betas + z-score simplification → A; label-parameter sensitivity + i.i.d.-bootstrap caveat → C; the multiple-testing statement with grid-size-dependent false-positive arithmetic → C, written Day 8 alongside the contribution statement). Everyone appends new compromises all week; the seed list means known disclosures never depend on anyone remembering them |
| Leakage checklist (Part 4) reproduced in full | A formats; each item signed by its verifying owner |
| Extensions (if run): ≤1 paragraph each + grid rows (spec 8.3) | A (Track C), C (Track D) |

**Figure/table inventory** (source scripts live in `scripts/figures/`, one script per figure, regenerable from frozen artifacts):

| ID | Description | Source script | Owner | Due |
|---|---|---|---|---|
| F1 | Scree plot (representative window) | `fig_scree.py` | A | Day 6 |
| F2 | PC1 factor return vs SPY overlay | `fig_pc1_spy.py` | A | Day 6 |
| F3 | Top-3 cumulative variance over time (2020 spike) | `fig_var_time.py` | A | Day 6 |
| F4 | Cluster co-membership stability over time, Track A vs B (vs C/D if run) | `fig_stability.py` | A | Day 6 |
| T1 | Characteristics-PC interpretation table ("expensiveness" etc., 4B.4) | `tab_char_pcs.py` | B | Day 6 |
| F5 | Base rate by year 2016–2024 (decay check, 12e; triggers start ~2016 after warmups) | `fig_base_rate.py` | C | Day 6 |
| F6 | Calibration / reliability diagram (E1, val + test) | `fig_calibration.py` | C | Day 7 |
| F7 | Cost-sensitivity sweep, one line per strategy, breakevens annotated | `fig_cost_sweep.py` | B | Day 6 |
| T2 | 2×4 (–4×4) grid results table with bootstrap CIs + row/column marginal means (answers spec 12a's three questions) | `tab_grid.py` | B | Day 7 |
| F8 | Turnover-matched control histogram (1000 seeded draws vs E1) | `fig_turnover_ctrl.py` | B | Day 7 |
| F9 | Consensus-bucket comparison (consensus / A-only / B-only) | `fig_consensus.py` | B | Day 7 |
| T3 | E1 coefficient table + cross-track comparison | `tab_coefs.py` | C | Day 7 |
| T4 | Leakage checklist (Part 4, all 10–12 items) | manual | A | Day 8 AM |

**Writing timeline:** skeleton + lit review Days 5–6 (parallel with coding — no results dependency); methods sections Days 6–7; results + discussion Day 7 evening → Day 8; full-team read-aloud pass Day 8 ~3pm; PDF frozen and **submitted by 9pm EST Day 8** (2h buffer).

**Submission checklist (owner: C, drafted Day 1):** required sections per handout; individual contribution statements; code submission format (repo zip vs link — **kickoff action item: re-read the course handout for required sections and deliverables**); reproducibility note (seed 311, `requirements.txt`, one-command rerun); figures referenced in text; citations complete incl. Ekinci/ICBDEIM honesty citations; PDF compiles from a clean Overleaf clone.

### 6.4 Risk register

| # | Risk | L | Impact | Early-warning signal | Mitigation | Owner |
|---|---|---|---|---|---|---|
| R1 | Bloomberg access fails or delayed | M | Track B slips or dies | Terminal booking not confirmed by Day 1 kickoff; pull not done by Day 2 noon | Fallback per B3 (Section 4): reduced characteristic set (sector one-hots + price-derived fields computed from our own price data — free-source historical fundamentals are not point-in-time reliable and are excluded), disclosed in limitations; Track B row survives with fewer columns. Terminal session booked before Day 1. | B |
| R2 | yfinance breakage / rate limits | M | Day 1 data slip blocks everything | Download errors or gappy columns Day 1 AM | Raw pulls cached and **committed** on first success (`data/raw/` in git); alternate source: stooq via `pandas-datareader`; fixture prices keep A and C fully unblocked regardless | B |
| R3 | Too few triggers for training | M | Models can't fit; CIs useless | Expected order-of-magnitude: ~80 pairs × 10y ⇒ roughly 500–1500 triggers; alarm if train rows < 300 at Checkpoint 1 | Widen trigger to 1.75 — **on validation evidence only, never test**, decided at a sync, logged in `DECISIONS.md`; secondary: extend horizon to 7d under the same rule | C |
| R4 | Extreme base rate (<15% or >85%) | L | Metrics/calibration degenerate | Base-rate number at Checkpoint 1 review | C2 protocol (Section 5): verify label code against golden file first (extreme rate = suspected bug), then `class_weight='balanced'`, AUC as primary metric, threshold/horizon adjustment on validation only | C |
| R5 | Noise test fails Day 5 (binding criteria) | L | Every result untrustworthy | Net-of-cost profit on noise with CI excluding 0, or model AUC significantly above the mechanical `f_abs_z`-only baseline (raw AUC > 0.5 alone is *expected* on noise — see B11) | **Stop-the-line:** all-hands until root-caused; both extension gates auto-fail; Day 6 buffer absorbed; leakage checklist re-walked item by item | B (runs), all (fix) |
| R6 | Results look too good | M | Fake headline result | Any cell: AUC > 0.60 or Sharpe > 2 (spec 10e) | Treat as a bug report until proven otherwise, per the smell-test protocol in 6.2; named investigator; cell excluded from report until cleared | All |
| R7 | Integration slips at Checkpoint 1 (Day 4) | M | Grid day (Day 5) lost | `triggers_a` not building by Day 4 noon | Person A pauses new features and pairs with C on the triggers pipeline; Track C and Track D auto-cut; Day 5 AM becomes the new checkpoint | A + C |
| R8 | A team member loses a day (illness, emergency, weekend conflict) | M | Their lane stalls | Missed sync or missed committed landing | Ownership map = coverage map: **A covers B's runner, B covers A's spreads/z-scores.** C is single-point-of-failure on models — mitigated by keeping E1 deliberately trivial (sklearn, <100 lines) and merged early (Day 3), so anyone can run it; E2/E3 are cuttable per 6.5 | All |
| R9 | Track D overrun | M | Eats Day 6–7, endangers freeze | AE not producing `residuals_d.parquet` integrated by A's clustering by Day 6 EOD | **Hard stop rule: if not integrated by Day 6 EOD it is cut**, no debate, and cited per spec 8.4 as scoped-but-not-run | C |
| R10 | Report crunch Day 8 | H | Rushed writing hurts the 50%-of-grade evaluation story | Skeleton/lit review not in Overleaf by Day 6 sync | Lit review + skeleton done Days 5–6 (no results dependency); methods drafted Day 7 while results freeze; Day 8 is polish, not first-drafting | Report owner |
| R11 | Overleaf outage / LaTeX compile failure near deadline | L | Can't produce PDF | Compile errors accumulating; Overleaf status page | PDF snapshot committed to the repo at every evening sync from Day 6; local `latexmk` fallback tested Day 6; submit target 9pm leaves 2h | Report owner |
| R12 | k-means degenerate clusters (all singletons / one giant cluster) | L | Pair list collapses | `pairs_a.csv` has < 30 or > 200 pairs on Day 3 | Silhouette-driven k already bounded (8–13); pair-builder logs cluster-size histogram; fallback: fix k mid-range (k=10) with `DECISIONS.md` entry | A |

### 6.5 Scope-cut ladder

If we fall behind, cuts happen **in this order**, decided at the evening sync and logged in `DECISIONS.md`. Cut early and cleanly rather than late and messily; per spec 8.4, cut items become one honest sentence in the report.

1. **Track D** (autoencoder residuals) — gated anyway; default NO-GO.
2. **Track C** (partial-correlation distance) — gated anyway; default NO-GO.
3. **E3** (small MLP) — drop the column; the bias-variance discussion cites the expectation instead of demonstrating it.
4. **Consensus extras and error-analysis depth (12c/12e)** — reduce consensus to the bare bucket table; drop the false-positive deep dive; keep the E1 coefficient table (1h, high value).
5. **E2** (GDA) — drop the generative-vs-discriminative column.
6. **Option-1 clustering robustness check** (the Day 6 A4-note check: residual-series clustering on ~6 windows vs the committed beta-vector input) — skip it; the kickoff DECISIONS.md entry + Zhang citation already justify the committed choice.
7. **Reduced bootstrap draws** (e.g. 1000 → 250) — CIs get slightly rougher; still reported.

**The never-cut list (spec Part 8.0, restated):** **purged CV (with embargo), the transaction-cost model, the turnover-matched control, and the honest limitations section.** These survive even if the grid shrinks to Track A × {E0, E1} — because that minimal grid, evaluated honestly, is still the pre-registered primary comparison and a complete, gradeable project.

---

## 7. Appendix: tooling notes and practical gotchas

Everything in this appendix was checked against live package indexes and changelogs on **Day 1 (2026-07-31)**. Pins marked **verify locally** could not be fully confirmed from documentation — run the one-line check given before trusting them. Nothing here changes any contract or config; it exists so nobody burns an evening on API drift.

### 7.1 yfinance

Current state (verified Jul 2026): yfinance went **1.x in late 2025**; the latest release is **1.5.2 (2026-07-23)**. The 1.0 jump had **no breaking API changes** vs late 0.2.x — it added a `yf.config` class, better exceptions, and an **optional retry mechanism for transient network errors**. `curl_cffi` became an optional dependency in 1.4.0 (falls back to `requests`).

- [ ] **Pin `yfinance==1.5.2`.** Do not float this package — it breaks more often than everything else in the stack combined, because it scrapes Yahoo. If 1.5.2 misbehaves on Day 1, the fallback move is to try the immediately previous release (1.5.1), not an old 0.2.x.
- [ ] **`auto_adjust=True` is the default** (has been since 0.2.x, unchanged through 1.x). We rely on it, but pass it **explicitly** anyway so the code documents the assumption:

```python
df = yf.download(
    tickers,                    # list of 40 strings
    start="2014-01-01", end="2025-01-01",   # 2014 = warm-up year (config DOWNLOAD_START)
    auto_adjust=True,           # explicit, even though it is the default
    threads=True,               # flip to False if you see partial/empty frames
    progress=False,
)
```

- [ ] **MultiIndex layout.** Multi-ticker `download()` returns columns as a 2-level MultiIndex with **level 0 = price field, level 1 = ticker** (default `group_by="column"`). Since ~0.2.51 even a *single*-ticker download returns the MultiIndex (`multi_level_index=True` default) — so the same idiom works for the SPY pull. The exact extraction for the Step 0 artifacts:

```python
prices: pd.DataFrame = df["Close"]          # -> date x ticker, plain str columns
volume: pd.DataFrame = df["Volume"]         # -> date x ticker
# For SPY downloaded separately:
spy = yf.download("SPY", ...)["Close"]      # still a DataFrame with one column "SPY"
```

  Do **not** use `df.xs(...)` gymnastics or positional level numbers; `df["Close"]` selects on level 0 by label and is stable across versions. After extraction, assert `prices.columns.dtype == object` and `isinstance(prices.index, pd.DatetimeIndex)`.
- [ ] **Rate limiting is real.** Yahoo throttles and sometimes serves empty frames or `YFRateLimitError`. 1.x has built-in rate-limit detection plus the optional retry mechanism — enable it via `yf.config` (exact knob name: check `yf.config` attributes in the REPL — **verify locally**, the config class is new in 1.0 and still evolving). Mitigations, in order:
  1. **One batched `yf.download` call for all 40 tickers** — never a loop of 40 single-ticker calls.
  2. **Write `data/raw/prices.parquet` / `volume.parquet` / `spy.parquet` immediately** after the first successful pull and commit them. Every later run reads the parquet; nobody re-hits Yahoo all week. This is also what makes the Section 7.10 repro story possible.
  3. If frames come back partially empty: retry with `threads=False`, then with a 60 s sleep and exponential backoff (3 attempts is plenty).
  4. Have `curl_cffi` installed (it impersonates a browser TLS fingerprint and dodges some blocking); yfinance uses it automatically when present.
- [ ] **Known recent breakage class:** 1.5.x releases have been chasing `curl_cffi>=0.16` compatibility and fundamentals-endpoint timeouts. We only need `download()` for prices/volume, which is the most stable code path. Avoid `Ticker.info` and fundamentals endpoints entirely — that is Bloomberg's job (7.6).
- [ ] Definition of done for the pull: `prices.shape ≈ (2516, 40)`, no column all-NaN, `prices.loc["2020-03-16"]` shows the COVID crash (big negative day) — a 30-second eyeball that adjustment and alignment are sane.

### 7.2 pandas / numpy / pyarrow

Verified current: **pandas 3.0.5** (3.0.0 landed 2026-01-21), **numpy 2.5.1** (2.5.0 **drops Python 3.11**), **pyarrow 25.0.0**.

- [ ] **Use Python 3.12** (see 7.9). That makes numpy 2.5.x legal; on 3.11 you would be stuck on numpy 2.4.x and fighting resolver conflicts.
- [ ] **pandas 3.0 behavior changes to be aware of** (we pin 3.0.5, not 2.x, because yfinance/sklearn are tested against it by now): Copy-on-Write is always on — **chained assignment silently does nothing**; always write `df.loc[mask, col] = ...`. The default string dtype is the dedicated `str` dtype, not `object` — comparisons with ticker strings are unaffected, but `df.columns.dtype` checks should use `df.columns.map(type)` sparingly and just trust label equality.
- [ ] **Parquet round-trip rules** (bugs here corrupt the interface contracts, so they're worth 10 minutes of tests on Day 1):
  - A `DatetimeIndex` **is** preserved through `to_parquet`/`read_parquet` with the pyarrow engine, including its name. Give every date index the name `"date"` before writing and assert it after reading.
  - **All column names must be `str`.** `pyarrow` refuses non-string column labels. Danger spots: `pd.DataFrame(np.ndarray)` gives integer columns (the PCA loadings path), and any `groupby(...).unstack()` can produce tuple columns. Blanket fix in every writer: `df.columns = df.columns.map(str)`.
  - Timezone-aware indexes round-trip but cause merge misery. Normalize once at ingest: `prices.index = prices.index.tz_localize(None)` (yfinance sometimes returns tz-aware).
  - `bool` columns (`co_clustered`, `enter`) round-trip cleanly; do not store them as 0/1 ints or the schema check in `tests/test_contracts.py` gets ambiguous.
- [ ] One shared helper, used by every producer, ends all debate: `src/io_utils.py` with `write_parquet(df: pd.DataFrame, path: Path) -> None` (asserts str columns + named DatetimeIndex where applicable) and `read_parquet(path: Path) -> pd.DataFrame`. ~1 h, Person B, Day 1, alongside the repo skeleton.

### 7.3 scikit-learn

Verified current: **scikit-learn 1.9.0** (2026-06-02; supports Python 3.11–3.14; new hard dependency on `narwhals`).

- [ ] **Pin `scikit-learn==1.9.0`.**
- [ ] **KMeans `n_init`:** the default has been `"auto"` (= 1 for k-means++) since 1.4. `"auto"` with k-means++ does a single init — *not* what the config freeze says. Always pass explicitly, exactly as frozen:

```python
KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=311)
```

- [ ] **Silhouette:** `silhouette_score(X, labels)` where `X` is the *same* formation-window feature matrix given to `fit` (betas for Track A, PC scores for Track B). For Track C's precomputed distance matrix use `silhouette_score(D, labels, metric="precomputed")`. Guard: silhouette is undefined for a clustering with any single distinct label — wrap in a check before comparing across the k range.
- [ ] **LogisticRegression for E1** on 7 dense features / a few hundred rows: `LogisticRegression(penalty="l2", C=C, class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=311)`. `lbfgs` is correct for tiny dense data; never `saga` (needs scaling-sensitive tuning) or `liblinear` (different regularization path). `max_iter=2000` (per C6) preempts the convergence warning that otherwise pollutes every log. C-grid is Person C's section; nothing here constrains it.
- [ ] **GDA (E2):** per Section 5 (C7), E2 is implemented as ~30 lines of numpy GDA for course alignment; `LinearDiscriminantAnalysis.predict_proba` is the 1e-6 verification reference, and `QuadraticDiscriminantAnalysis` (per-class covariance; `reg_param≈1e-3` if covariances are near-singular — 7 features is small enough that this mostly won't bite) supplies the robustness row.
- [ ] **StandardScaler discipline:** exactly one `scaler.fit(X_train)`; then `transform` on val and test. The fitted scaler is part of the frozen model bundle Person C hands to the Day 7 test run — persist it with the model, never re-fit downstream. A `Pipeline([("scaler", StandardScaler()), ("clf", ...)])` makes this impossible to get wrong and is the recommended shape.

### 7.4 Rolling OLS: numpy, not statsmodels

The Step 3 betas require, per trading day (~2,300 days), a regression of a 252×40 return block on a 252×m factor block. That is **one multi-target least-squares solve per day**, not 40 separate regressions:

```python
def window_betas(F: np.ndarray, R: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """F: (252, m+1) design matrix — a ones column FIRST (RESIDUAL_INCLUDE_ALPHA=True,
    per A2), then the m in-window factor returns reconstructed from THIS window's
    eigenvectors (A2's construction — not sliced from the stored factors_a series).
    R: (252, 40) stock returns. Returns (m+1, 40): alphas in row 0, betas below."""
    A = F.T @ F + ridge * np.eye(F.shape[1])
    return np.linalg.solve(A, F.T @ R)
```

- [ ] Total cost: ~2,300 solves of an m×m system (m ≤ 5) — **well under one second for the whole history**. Do not reach for `statsmodels.RollingOLS`, loops over stocks, or anything clever.
- [ ] **Conditioning:** eigenportfolio factors are near-orthogonal by construction, so `A` is well-conditioned; but keep the `ridge` argument (default 0, flip to `1e-8` if `np.linalg.LinAlgError` or wild betas appear) and log whenever it is non-zero. `np.linalg.lstsq(F, R, rcond=None)` is the drop-in alternative if you prefer not to form `F.T @ F`.
- [ ] **statsmodels (`==0.14.6` — latest stable line; check `pip index versions statsmodels` on Day 1, **verify locally** that 0.14.6 imports cleanly against pandas 3.0)** is used in exactly one place: if the report wants a pretty OLS summary table (e.g., a single illustrative beta regression). It never sits in the pipeline hot path.

### 7.5 PyTorch (CPU only)

Verified current: **torch 2.13.0** (2026-07-08). We need CPU wheels only.

- [ ] Install from the CPU index so nobody downloads 2+ GB of CUDA:

```
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
```

- [ ] **Determinism recipe** (put in `src/seeding.py`, called by every entry point):

```python
def seed_everything(seed: int = 311) -> None:
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
```

  Caveats: `use_deterministic_algorithms(True)` can raise on ops without deterministic CPU kernels — nothing in a tiny MLP/AE (Linear, ReLU/tanh, MSE, BCE, Adam) triggers this, but if it ever raises, downgrade that one call to `warn_only=True` and note it in DECISIONS.md. CUDA-specific env vars (`CUBLAS_WORKSPACE_CONFIG`) are irrelevant on CPU. DataLoader workers: use `num_workers=0` — data this small needs no workers and it removes a whole seeding category.
- [ ] **Compute reality check, so nobody budgets time for "training":** E3 is 7 features → 8–16 hidden units → 1 output on a few hundred rows: full-batch or minibatch, a few hundred epochs with early stopping = **~1–5 seconds per fit**; a 20-point hyperparameter sweep is under two minutes. Track D's AE is 40→(3–5)→40 on 252 rows, monthly retrains 2015–2024 ≈ **120 fits**; at ~2–5 s per fit that is **≤ 10 minutes for the entire track**, seconds for the linear-AE sanity check. Compute is never the bottleneck; the 12–15 h Track D estimate is all plumbing and validation.

### 7.6 Bloomberg terminal export (Track B, one session, Day 2)

Realistic options for a student with one terminal session, ranked:

1. **Excel Add-in (`BDH`/`BDP`)** — the right tool. Zero setup on a terminal PC, and the output is already tabular.
2. Terminal `EXCEL` template builder / drag-and-drop export — fine as a fallback for individual fields.
3. **`blpapi` (Python API)** — overkill for one session: requires a Desktop API entitlement check, a local install, and debugging time we don't have. Skip.

The recipe:

- [ ] **Before the session**, Person B lists the 19 target fields from spec 4B.1 and, at the terminal, runs **`FLDS`** on one ticker to verify each mnemonic actually exists and returns quarterly history (e.g., `PE_RATIO`, `PX_TO_BOOK_RATIO`, `CUR_MKT_CAP`, `TOT_ANALYST_REC`, ... — **verify mnemonics at the terminal**; do not trust any list found online, field names drift).
- [ ] **One workbook, one sheet per field.** Each sheet: tickers across columns, one `BDH` array per ticker:
  `=BDH("AAPL US Equity","PE_RATIO","12/31/2014","12/31/2024","Per=Q","Days=A","Fill=P")`
  (`Per=Q` quarterly; `Fill=P` carries the previous value across empty periods). 40 tickers × ~40 quarters per sheet is far below any practical `BDH` limit; the daily data-usage cap is generous enough for ~30k cells but **do the pull once, not iteratively**.
- [ ] Let all sheets finish calculating (watch the status bar — `BDH` fills asynchronously), then **File → Save As → CSV, one file per sheet**, into `data/raw/bloomberg/{field}.csv`. Commit the CSVs immediately; the terminal session is the only unrepeatable step in the whole project.
- [ ] **Time-box: 2 hours at the terminal.** Any field still fighting back at the deadline (wrong periodicity, entitlement error, mnemonic not found) gets dropped on the spot and the B3 fallback rule (Plan Section on Track B / spec 4B.2: drop sparse columns, median-impute) absorbs it. 12 good fields beat 19 fields and a second terminal trip.
- [ ] Consumes: ticker list from `data/raw/prices.parquet` columns. Produces: `data/raw/bloomberg/*.csv`. Done when: every CSV loads in pandas with a parseable quarterly date column and ≥ 80% non-empty cells.

### 7.7 Matplotlib on WSL2

- [ ] WSL2 has no reliable display; pipeline code must **never call `plt.show()`**. Force the file-only backend once, project-wide: `MPLBACKEND=Agg` exported in the Makefile (belt) and `matplotlib.use("Agg")` at the top of `src/plotstyle.py` before any `pyplot` import (suspenders).
- [ ] All figure code goes `fig, ax = plt.subplots(...)` → `fig.savefig(path, dpi=200, bbox_inches="tight")` → `plt.close(fig)` (the `close` matters in loops — Agg leaks figures otherwise).
- [ ] One consistent style, one place — `src/plotstyle.py`, imported by every figure script:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def apply_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (8, 4.5), "figure.dpi": 110, "savefig.dpi": 200,
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "lines.linewidth": 1.6,
        "legend.frameon": False,
    })
```

  Person A creates it Day 2 with the first sanity-check figures; done when every figure in `results/figures/` visibly shares the style.

### 7.8 pytest and the golden-file pattern

Verified current: **pytest 9.1.1**; pin `pytest==9.1.1`.

- [ ] **Goldens are CSVs, not parquet**, stored under `tests/golden/` — they must be human-diffable in a PR (`fixture_zscores.py`'s expected-triggers file is the canonical example). Pattern: compute → `pd.testing.assert_frame_equal(result.reset_index(), pd.read_csv(golden, parse_dates=["trigger_date"]), check_exact=False, atol=1e-9)`. Regenerating a golden requires a reviewer — treat golden diffs like contract changes.
- [ ] **Future-perturbation tests** (the "poke tomorrow, assert today unchanged" leakage tests) must run on a **300-day slice** of the synthetic data, not the full history — a full rolling-PCA pass per test makes the suite minutes-slow and people stop running it. Target: whole suite `< 30 s` so `pytest -q` runs before every push. Mark anything slower `@pytest.mark.slow` and exclude it by default via `addopts = -m "not slow"` in `pyproject.toml`.
- [ ] Seed inside every test via `seed_everything(311)` (7.5) — never rely on import-order side effects.

### 7.9 requirements.txt

Recommended interpreter: **Python 3.12** on WSL2 Ubuntu (`sudo apt install python3.12-venv` if needed; 24.04 ships 3.12 as system python). Rationale: numpy 2.5 dropped 3.11; 3.13/3.14 are legal for sklearn but the wider ecosystem (yfinance's transitive deps, curl_cffi wheels) is best-tested on 3.12.

```text
# --- core numerics (Py 3.12) ---
numpy==2.5.1            # current stable; 2.5 line requires Py>=3.12
pandas==3.0.5           # current 3.0.x patch; CoW + str dtype defaults (see 7.2)
pyarrow==25.0.0         # parquet engine for every artifact contract
scipy==1.16.0           # sklearn/statsmodels dep; verify locally that resolver agrees
# --- data ---
yfinance==1.5.2         # scraper: pin hard, never float (see 7.1)
curl_cffi==0.13.0       # optional yfinance transport, dodges Yahoo blocking; verify locally
# --- ML ---
scikit-learn==1.9.0     # KMeans/LogReg/LDA-QDA/metrics; pulls in narwhals
statsmodels==0.14.6     # report-facing OLS summaries only; verify pandas-3 import
# --- torch: install separately with the CPU index url (see 7.5) ---
# pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
# --- plotting / testing ---
matplotlib==3.10.3      # Agg-only usage; if pip offers a newer 3.10.x/3.11.x, take it and re-pin — verify locally
pytest==9.1.1           # golden-file + leakage test suite
```

- [ ] Day 1 kickoff includes one command by each person: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` plus the torch line, then `pytest -q`. If pip's resolver rejects any pin (most likely `scipy`/`curl_cffi`/`matplotlib` — the three flagged **verify locally**), the person who hits it fixes the pin, commits, and posts in the channel. The file is frozen after Day 1 like the config.
- [ ] `pip freeze > requirements.lock.txt` after the first successful install, committed — that lockfile, not the spec file, is what the report's reproducibility statement cites.

### 7.10 Repro discipline: Makefile + RUNBOOK

Principle: **after Day 1, no target ever contacts Yahoo or Bloomberg.** The committed raw pulls (`data/raw/*.parquet`, `data/raw/bloomberg/*.csv`) are the roots of the DAG; everything downstream is a deterministic function of them plus `src/config.py` and seed 311.

```make
.PHONY: all data tracka trackb dataset grid noise-test test-run figures test

data:            ## returns.parquet + both fixtures from committed raw pulls (NO network)
	python -m src.data.prices --from-cache
	python -m src.synth.make_synthetic
	python -m src.synth.fixture_zscores
tracka: data     ## Steps 2-6, track a
	python -m src.factors.pca && python -m src.factors.residuals
	python -m src.clustering.recluster --track a
	python -m src.pairs.build_pairs --track a
	python -m src.spreads.spread --track a
trackb: data     ## 4B pipeline; reuses the shared pair-builder and spread modules
	python -m src.data.characteristics
	python -m src.clustering.recluster --track b
	python -m src.pairs.build_pairs --track b
	python -m src.spreads.spread --track b
dataset: tracka trackb
	python -m src.datasets.assemble --tracks a,b
grid: dataset    ## E0/E1/E2/E3 x tracks on train+val only
	python -m src.experiments.run_grid --tracks a,b --models e0,e1,e2,e3 --split trainval
noise-test:      ## full pipeline on synth/make_synthetic.py output; PASS/FAIL per Part 4
	python -m src.experiments.noise_test
test-run:        ## Day 7 ONLY, witnessed; guarded (--i-am-sure + FREEZE.md + date >= Aug 6) and refuses to run twice (results/TEST_RUN_DONE marker)
	python -m src.experiments.run_grid --tracks a,b --models e0,e1,e2,e3 --split test --i-am-sure
figures: grid
	python -m scripts.figures.make_all
test:
	pytest -q
all: data tracka trackb dataset grid figures
```

- [ ] `make test-run` writes `results/TEST_RUN_DONE` on completion and aborts if it exists — mechanical enforcement of "test touched exactly once."
- [ ] **`RUNBOOK.md`** (repo root, one page, Person B owns, done Day 4 and re-verified Day 7): fresh-clone-to-full-grid in order —

```text
1. git clone <repo> && cd pair_trading
2. python3.12 -m venv .venv && source .venv/bin/activate
3. pip install -r requirements.txt
4. pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
5. pytest -q                      # all green before anything else
6. make all                       # raw parquet -> full train+val grid, ~minutes
7. make noise-test                # must find nothing
8. make test-run                  # Day 7, once, all three watching
```

- [ ] Definition of done: a teammate who has never touched the repo runs steps 1–7 on a clean WSL2 machine on Day 5 and gets bit-identical `results/metrics_*` (same seed, same lockfile). That dry-run *is* the reproducibility claim in the report.

Version sources checked 2026-07-31: [yfinance PyPI](https://pypi.org/project/yfinance/) / [changelog](https://github.com/ranaroussi/yfinance/blob/main/CHANGELOG.rst), [pandas 3.0 release notes](https://pandas.pydata.org/docs/whatsnew/v3.0.0.html), [NumPy news](https://numpy.org/news/), [scikit-learn release history](https://scikit-learn.org/stable/whats_new.html), [PyTorch releases](https://github.com/pytorch/pytorch/releases), [pyarrow PyPI](https://pypi.org/project/pyarrow/), [statsmodels releases](https://github.com/statsmodels/statsmodels/releases), [pytest changelog](https://docs.pytest.org/en/stable/changelog.html).

---

## 8. Appendix: spec → plan traceability matrix

Every requirement in `project_spec_v2.md`, mapped to the plan location that implements it, its owner, and its scheduled day. Use this at the Day 5 and Day 7 syncs as the completeness checklist: any row whose artifact does not exist by its listed day is a named, visible gap — not a silent one.

| Spec item | Plan location | Owner | Day |
|---|---|---|---|
| **Part 3 / Step 0** — 40 stocks, 4 sectors × 10 | §4 B1 universe module + §2.8 kickoff D2 | B | 1 |
| Step 0 — yfinance download, `auto_adjust=True` confirmed | §4 B2 (asserted explicitly) + §7.1 | B | 1 |
| Step 0 — drop ticker >2% missing days | §4 B2 cleaning check 1 (+ B1 replacement protocol) | B | 1 |
| Step 0 — forward-fill isolated single-day gaps | §4 B2 cleaning check 2 | B | 1 |
| Step 0 — align to common trading calendar | §4 B2 cleaning check 3 | B | 1 |
| Step 0 — verify no ±50% return without real event | §4 B2 cleaning check 4 (manual-review table) | B | 1 |
| Step 0 — verify ~252 rows/year | §4 B2 cleaning check 5 | B | 1 |
| Step 0 — survivorship-bias acknowledgment (Zhang argument) | §4 B1 + §6.3 report section "Data, universe, survivorship-bias" | B | 1, 8 |
| **Step 1** — returns, choice stated (log vs simple) | §1.4 K1 (log) + §4 B2 `compute_returns` | B | 1 |
| **Step 2a** — rolling 252d window excluding day t | §3 A1 note 1 (+ in-loop assertion) | A | 1–2 |
| Step 2b — window-local standardization | §3 A1 note 2 | A | 1–2 |
| Step 2c — correlation (not covariance) matrix | §3 A1 note 3 | A | 1–2 |
| Step 2d — `eigh`, descending sort | §3 A1 note 4 | A | 1–2 |
| Step 2e — a-priori component rule, never tuned on test | §1.4 K2 + A1 `choose_n_components` | A | 1 (frozen), 1–2 |
| Step 2f — sign-fix every window | §3 A1 note 5 + sign-fix property test (§2.5) | A | 1–2 |
| Step 2g — eigenportfolios, inverse-vol weights | §3 A1 note 6 | A | 1–2 |
| Step 2 sanity checks — scree, PC1 loadings, PC1-vs-SPY, variance-over-time (2020 spike) | §3 A3 (4 scripts, PASS/FAIL) + figures F1–F3 | A | 2 (figures 6) |
| **Step 3** — OLS betas trailing window, applied OOS | §3 A2 | A | 2 |
| Step 3 — stale-beta limitation disclosed | §3 A day-budget Day 8 ("betas drift") + limitations.md | A | 8 |
| **Step 4** — cluster input, primary option stated | §1.4 K4 (factor-beta vectors, Zhang evidence) + §3 A4 | A | 3 |
| Step 4 — "try both" second option (252-length residual series) | §3 A4 note: committed skip recorded in DECISIONS.md (Zhang p=0.01 vs 0.785) + optional Day-6 robustness check (~1.5h, only if Track C NO-GO) | A | 1 (decision), 6 (optional) |
| Step 4 — k-means++, n_init=10, empty-cluster handling | §3 A4 + §7.3 (explicit `n_init=10` warning) | A | 3 |
| Step 4 — choose k by silhouette/elbow, formation window only | §3 A4 (silhouette, k∈8–13, logged) | A | 3 |
| Step 4 — label-switching → co-membership matrix | §3 A4 `comembership()` (never compare raw labels) | A | 3 |
| Step 4 — stability analysis (fraction co-clustered w→w+1) | §3 A4 + figure F4 (A vs B stability) | A | 3, 6 |
| **Step 4B.1** — 19 Bloomberg fields, quarterly pull | §4 B3 (session plan, mnemonics, fallback rule) + §7.6 | B | 2 |
| 4B.1 — point-in-time warning (as-reported preferred, else disclosed) | §4 B3 step 3 + `FIELDS_USED.md` + limitations | B | 2, 8 |
| 4B.2 — sentiment merge (19→18) | §4 B4 `clean_snapshot` step 1 (+ unit test 15/5→+0.5) | B | 3 |
| 4B.2 — log-transform size columns | §4 B4 `clean_snapshot` step 2 | B | 3 |
| 4B.2 — z-score every column | §4 B4 `clean_snapshot` step 3 | B | 3 |
| 4B.2 — our addition: drop sparse columns + median-impute, disclosed | §4 B4 (<90% coverage drop, logged) + limitations | B | 3, 8 |
| 4B.3 — PCA on characteristics, fewer components (thin-data caveat) | §4 B4 `pca_characteristics` (cap 5) | B | 3 |
| 4B.4 — read and name the components | §4 B4 `name_components` + table T1 | B | 3, 6 |
| 4B.5 — Route B (projection) primary | §4 B4 (scores = Route B projection) | B | 3 |
| 4B.5 — Route A "if time allows" | §4 B4 `pca_characteristics` docstring: deliberately skipped citing Zhang's decisive comparison; logged in DECISIONS.md | B | 1 (decision) |
| 4B.6 — k=10–13 + Zhang group rules (skip 1, take 2–4, split 5+) | §4 B4 `cluster_track_b` + §3 A5 shared builder | B (A's builder) | 3 |
| **Step 5** — pairs within clusters, output schema w/ `source` tag | §3 A5 + §2.2 pairs contract | A | 3 |
| **Step 6a** — spread construction (option chosen, h formation-only n/a) | §1.4 K6 (simple) + §3 A6 (+ re-anchoring policy) | A | 3 |
| Step 6b — rolling z-score, trailing window-local stats | §1.4 K7 (60d) + §3 A6 `zscore` | A | 3 |
| Step 6 — OU s-score simplification stated explicitly | §3 A day-budget Day 8 ("z-score simplification") + limitations | A | 8 |
| **Step 7a** — onset-only trigger at \|z\| crossing 2.0 | §5 C1 (exact onset semantics + re-arm policy + P6 fixture test) | C | 1–2 |
| Step 7b — label (50% reversion within 5d), parameters stated a priori | §1.4 K8 + §5 C1 | C | 1 (frozen), 1–2 |
| Step 7c — base-rate check before trusting anything | §5 C2 + pre-agreed decision table + Checkpoint 1 review | C | 4 |
| Step 7d — overlap property → motivates purging | §5 C5 (purge via `horizon_end_date`) | C | 2 |
| **Step 8** — feature 1: \|z\| at trigger | §5 C3 `f_abs_z` | C | 2–3 |
| Step 8 — feature 2: spread volatility 60d | §5 C3 `f_spread_vol_60d` | C | 2–3 |
| Step 8 — feature 3: residual momentum 5d | §5 C3 `f_resid_mom_5d` (sign convention defined) | C | 2–3 |
| Step 8 — feature 4: market volatility (PC1, 20d) | §5 C3 `f_mkt_vol_20d` (factors_a for all tracks) | C | 2–3 |
| Step 8 — feature 5: relative volume 20d | §5 C3 `f_rel_volume_20d` | C | 2–3 |
| Step 8 — feature 6: days since last trigger | §5 C3 `f_days_since_trigger` (cap 126, NaN policy) | C | 2–3 |
| Step 8 — feature 7: cluster stability | §5 C3 `f_cluster_stability` (default-0 policy stated) | C | 2–3 |
| Step 8 — no future info; standardize on training stats only | §5 C3 look-ahead test + scaler fit on train inside `EntryModel.fit` | C | 2–3 |
| **Step 9 E0** — fixed rule, implemented fairly (same engine/costs) | §4 B6 (same decisions-table path, zero special cases) | B | 4 |
| Step 9 E1 — logistic, L2 tuned on val, balanced, τ on val never test | §5 C6 + C9 pre-registered τ rule | C | 3–5 |
| Step 9 E1 — coefficients reported | §5 C6 coefficient table + T3 | C | 4, 7 |
| Step 9 E2 — GDA (generative comparison) | §5 C7 (numpy GDA ≡ LDA, sklearn cross-check, QDA robustness row) | C | 3–4 |
| Step 9 E3 — small MLP, early stopping, weight decay, kept small | §5 C8 (7→8/16→1, fixed a priori) | C | 5–6 |
| Step 9 — bias-variance ladder discussion | §5 C8 expected-result framing + Day 8 discussion | C | 8 |
| **Step 10a** — chronological splits, test touched once (spec: test 2023–2025) | §1.4 K11 + §2.3 config (test 2023–2024; deviation flagged at kickoff D3) + C5 | C | 1 (frozen), 2 |
| Step 10b — purging (5d horizon) | §5 C5 rule 2 + purge property tests; rows retained and counted | C | 2 |
| Step 10c — embargo (10 trading days) | §5 C5 rule 3 + tests | C | 2 |
| Step 10d — AUC primary; precision/recall at τ; no accuracy | §5 C10 `classification_metrics` ("accuracy is never reported") | C | 3–5 |
| Step 10d — calibration / reliability diagram | §5 C10 `reliability_diagram` (quantile bins) + F6 + C11 commentary | C | 5–7 |
| Step 10d — strategy metrics: hit rate, mean/cum return, Sharpe, turnover, max drawdown | §2.2 metrics JSON schema + §4 B5/B7 | B/C | 5 |
| Step 10d — bootstrap CIs on everything | §5 C10 `bootstrap_ci` (1000 resamples; iid caveat disclosed) | C | 5–6 |
| Step 10e — smell test posted and applied | §6.2 Day 5 checkpoint (every cell, as a team) + C10 ≥0.65 bug rule + risk R6 | All | 5 |
| **Step 11** — costs per transaction, both legs, both ends | §1.4 K9/K10 + §4 B5 (booked −2c at entry and exit; cost tests) | B | 2–4 |
| Step 11 — cost sweep 0–50 bps, one line per strategy | §4 B8 + figure F7 | B | 5 |
| Step 11 — breakeven cost extraction | §4 B8 `breakeven_bps` (analytic + interpolated) | B | 5 |
| **Step 12a** — factorial grid runs (2×4, extensible) | §4 B7 `run_grid` + table T2 | B | 5 (val), 7 (test) |
| Step 12a — row/column/interaction analysis (the three questions) | §4 B7 `make_grid_table` (row/column marginal means built into T2) + B's Day 8 results section answers all three explicitly | B | 5, 8 |
| Step 12b — turnover-matched control | §4 B9 (per-quarter matching, seeded draws, percentile + histogram F8) | B | 6 (val), 7 (test) |
| Step 12c — consensus pairs (3 buckets: reversion rate, AUC, net perf) | §4 B10 (pair-quarter granularity) + F9 | B | 6 |
| Step 12d — factor-neutralization question (Han et al. engagement) | §4 B13 (reads the row effect through the 12d lens; Han et al. 2023 comparison paragraph + citation) | B | 7 PM |
| Step 12e — coefficient comparison across tracks (1h) | §5 C11 + T3 | C | 6–7 |
| Step 12e — base-rate decay by year (1h) | §5 C2 + figure F5 | C | 4, 6 |
| Step 12e — error analysis on confident false positives (3h) | §5 C11 (top-20 FP notebook, cross-tabs) | C | 6–7 |
| **Part 4 item 1** — PCA trailing windows only | §3 A7 (future-perturbation test, 2 cut dates + written note) | A | 4–5 |
| Part 4 item 2 — standardization window-local | §3 A7 (perturbation + grep audit) + §2.5 leakage helper on zscores | A | 4–5 |
| Part 4 item 3 — betas estimated before application | §3 A7 (perturbation + A2 in-loop assertion) | A | 4–5 |
| Part 4 item 4 — clustering formation-window only | §3 A7 (perturbation on labels/stability + code-path note) | A | 4–5 |
| Part 4 item 5 — labels purged at boundaries | §5 C5 tests + retained/countable rows; §1.1 audit map | C | 2, 5 |
| Part 4 item 6 — embargo applied | §5 C5 tests; §1.1 audit map | C | 2, 5 |
| Part 4 item 7 — test set touched exactly once | §4 B7 hard guard + §5 C13 freeze + §6.2 witnessed protocol (+§7.10 marker) | C (owns item), B (mechanism) | 6–7 |
| Part 4 item 8 — signal at t, returns at t+1 (hand-traced 3 dates) | §4 B5 shift test + B12 manual trace (3 regimes, template) | B | 2–3, 5 |
| Part 4 item 9 — costs at each transaction | §4 B5 cost-arithmetic test (daily P&L booking) | B | 2–3 |
| Part 4 item 10 — Bloomberg point-in-time or disclosed | §4 B3 (`FIELDS_USED.md` + limitations path) | B | 2, 8 |
| Part 4 item 11 — (Track C) precision matrix in-window, shrinkage documented | §3 A8.1/A8.5 (lam recorded, not tuned; audit note) | A | 6 (gated) |
| Part 4 item 12 — (Track D) AE trailing-only, schedule + standardization documented | §5 C12(ii) (leakage item 12 cited) | C | 6 (gated) |
| Part 4 — noise test (full pipeline on random walks) | §4 B11 + `make noise-test` + 3 concrete pass criteria; extensions rerun (A8.5, C12iii) | B | 5 (ext.: 6) |
| Part 4 — manual trace | §4 B12 `docs/manual_trace.md`, 3/3 MATCH | B | 5 |
| Part 4 — checklist published IN the report | §6.3 T4 (A formats, each item signed by verifying owner) | A + all | 8 |
| **Part 5** — honest novelty correction, cite Ekinci + ICBDEIM | §6.3 lit-review section ("incl. the honest novelty correction") + submission checklist citations | Kickoff-chosen (default C) | 5–6 |
| Part 5 — contribution sentence in report | §6.1 Day 8 (C: "abstract + contribution statement") | C | 8 |
| Part 5/7 — pre-registered primary comparison before running anything | §1.4 K12 + §2.3 `PRIMARY_COMPARISON` (frozen at kickoff) | All | 1 |
| **Part 6** — expectations calibration (weak results legitimate) | Distributed: smell-test bands (§6.2), expected-result framings (C8, C12, B10 "little overlap is a finding"); Day 8 discussion | All | 5–8 |
| **Part 7** — survivorship bias | B1 + B report section + limitations.md | B | 8 |
| Part 7 — only 40 stocks, thin correlation matrices | `docs/limitations.md` Day-1 seed (§2.6) → assembled by A | A | 1, 8 |
| Part 7 — Bloomberg not point-in-time | B3 → limitations | B | 8 |
| Part 7 — column-drop/median-impute choice | B4 (logged) → limitations | B | 8 |
| Part 7 — simplified z-score vs OU s-score | A Day 8 limitations item | A | 8 |
| Part 7 — beta drift | A Day 8 limitations item | A | 8 |
| Part 7 — k-means local optima / unstable membership | F4 stability figure + `docs/limitations.md` Day-1 seed (§2.6) | A | 1, 6, 8 |
| Part 7 — label-parameter sensitivity | K8 + §1.4 trigger-vs-entry wrinkle (disclosed) + limitations.md | C | 8 |
| Part 7 — 18×18 from 40 stocks statistically thin | B4 cap-at-5 caveat → limitations | B | 8 |
| Part 7 — (Track C) shrinkage is a modelling choice | A8.1 + A Day 8 ("shrinkage") | A | 8 |
| Part 7 — (Track D) coarser retrain schedule, small capacity | C12(ii) (goes in limitations) | C | 8 |
| Part 7 — small trigger sample → wide CIs | C10 bootstrap caveat (docstring + limitations bullet) | C | 8 |
| Part 7 — stylized cost model | B5 aggregation-policy disclosure + limitations | B | 8 |
| Part 7 — short-sale constraints / borrow costs ignored | `docs/limitations.md` Day-1 seed, owner-tagged → B (§2.6, §6.3) | B | 1, 8 |
| Part 7 — multiple-testing statement (expected false positives, emphasize patterns) | K12 pre-registration + `docs/limitations.md` seed; report statement with grid-size-dependent arithmetic assigned to C, Day 8 (§6.3) | C | 1, 8 |
| **Part 8.0** — Track C gate: full 2×4 end-to-end + audits passed, Day 5 evening | §6.2 (spec quoted verbatim; default NO-GO; noise test blocks gate) | All (B owns evidence) | 5 |
| Part 8.0 — Track D gate: C resolved + freeze realistic + E-models done, Day 6 morning | §6.2 + §5 C12 gate paragraph (any-red-item = cut) | All | 6 |
| Part 8.0 — priority: Track C first; never-cut list preserved | §6.2 + §6.5 (never-cut list restated verbatim) | All | — |
| 8.1 Track C step 1 — reuse in-window correlation matrix | A8 (refactor `pca_one_window` to return C) | A | 6 |
| 8.1 step 2 — diagonal shrinkage before inverting | A8.1 (lam=1e-3 in config, reported, not tuned) | A | 6 |
| 8.1 step 3 — precision → partial correlation | A8.1 (unit tests incl. analytic 3-variable case) | A | 6 |
| 8.1 step 4 — distance + identical k-means machinery | A8.2 (same cadence, k-range, seed, stability) | A | 6 |
| 8.1 step 5 — same Step-5 rules, `source="track_c"` | A8.3 (via shared builder; spreads from residuals_a) | A | 6 |
| 8.1 — overlap measurement (extends consensus) | A8.4 (Jaccard per window + figure; feeds B10 per 8.3) | A | 6 |
| 8.2 Track D — Baldi–Hornik linear-AE ≡ PCA check FIRST | §5 C12(i) (stop-if-mismatch; reported either way) | C | 6 (gated) |
| 8.2 — tiny AE, bottleneck matched to PCA m | C12(ii) + config `AE_BOTTLENECK` | C | 6 |
| 8.2 — monthly retrain on trailing 252d, mismatch → limitations | C12(ii) + config `AE_RETRAIN_EVERY=21` | C | 6, 8 |
| 8.2 — window-local input standardization; strictly OOS residual | C12(ii) | C | 6 |
| 8.2 — same downstream, `source="track_d"`; integration by A | C12(iii) (A runs A4–A6 on residuals_d; spec ownership preserved) | C + A | 6 |
| 8.3 — grid/consensus/multiple-testing absorption of extension rows | T2 auto-extends; A8.4 feeds consensus; multiple-testing arithmetic restated for the actual grid size by C (§6.3) | B/A/C | 6–8 |
| 8.4 — skip path: one-sentence citations | §6.2 gates + §6.5 cut-ladder + A8/C12 no-go branches (pre-written sentence) | A (C), C (D) | 6, 8 |
| **Report** — figures the spec calls out (scree, var-over-time 2020, calibration, cost sweep, control histogram, base-rate decay, consensus, stability) | §6.3 inventory F1–F9, T1–T4 with owners/scripts/due days | A/B/C per table | 6–8 |
| Report — leakage checklist reproduced in full | T4 | A + all | 8 |
| Report — contribution sentence + honest positioning | Day 8 C + lit-review owner | C | 8 |
