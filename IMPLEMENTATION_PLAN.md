# Implementation Plan v2 — Predicting Temporary Stock-Price Gap Convergence

**CSC311 Summer 2026 — Option 2 Final Project**
**Team:** Alex Bachynsky, Golam Eram, Jaskaran Narula
**Deadline:** Friday, August 7, 2026, 11:00pm EST
**Build window:** Day 1 = Sunday August 2 → Day 4 = Wednesday August 5 (results frozen Day 4 EOD). **Aug 6–7 = write-up**, planned separately after all results exist.
**Companion documents:** `project_spec_v2.md` (the *what* and *why*); `IMPLEMENTATION_PLAN_v1_8day.md` (the superseded horizontal-split plan, kept for reference).

---

## 0. How to use this plan

**What changed from v1.** Three team decisions drove this rewrite:

1. **Vertical slices instead of horizontal layers.** In v1, one person owned pairing, one owned data/backtesting, one owned models. In v2, each person owns a full vertical slice through all three stages of the project — **one pairing method, one model, and the backtesting/evaluation of that model** — so everyone touches every stage, understands the whole pipeline, and can push their slice to completion without waiting on anyone.
2. **The measuring instruments stay shared and are built exactly once.** One backtest engine, one trigger/label/feature dataset builder, one metrics/bootstrap module. This is not negotiable engineering taste — it is what makes the factorial grid a *fair* comparison (every cell scored by identical machinery) and what makes the turnover-matched control valid. Each person builds **one piece of this shared spine** (Days 1–2), which is also how everyone learns the parts of the pipeline outside their slice.
3. **Track C is promoted to a committed third pairing method** (the grid is now **3×4 = 12 cells**), so each person owns a real selection method. **Track D (autoencoder) is cut** — it gets the spec-8.4 one-sentence citation. The build compresses to **4 days at ~8–9 focused hours per person per day**; the cut ladder in §6.5 is the pressure valve if that pace slips.

**The operating principle is unchanged from v1:** interfaces between people are *frozen artifact contracts* (data files with fixed schemas, §2.2), not calls into each other's code, and everyone develops against fixtures from Day 1 so nobody blocks on real upstream data. The test set is touched exactly once, Day 4 ~noon, witnessed, from frozen models, through a guarded runner.

### Reading order per person

| Person | Read first | Then | Skim |
|---|---|---|---|
| P1 — Substrate · Track A · E2 | §3 | §2.2 contracts, §6.1 | §5 (the engine/runner your model flows through), §7 |
| P2 — Training data · Track B · E1 | §4 | §2.2, §2.4 fixtures, §6.1 | §3 (the machinery you call), §7.1, §7.6 |
| P3 — Evaluator · Track C · E3 | §5 | §2.2, §6.1 | §3 (the residuals/correlations you consume), §7.5 |

### Contents

| § | Section |
|---|---|
| 1 | The slice model and ownership map |
| 2 | Architecture, contracts, and engineering conventions |
| 3 | Slice P1 — Substrate: representation, Track A, E2 |
| 4 | Slice P2 — Training data: dataset builder, Track B, E1 |
| 5 | Slice P3 — Evaluator: engine, Track C, E3 |
| 6 | Timeline, coordination, and risk (4-day schedule, syncs, gates, risks, cut ladder) |
| 7 | Appendix: tooling notes and practical gotchas |
| 8 | Appendix: spec → plan traceability matrix |

---

## 1. The slice model and ownership map

### 1.1 The three slices

Real names map to slices at kickoff under two hard constraints: **P2 must have Bloomberg terminal access** (Track B pull); **P3 should be the strongest in PyTorch** (E3, engine plumbing). P1 takes the heaviest math (PCA/residuals) and the lightest model (GDA) to balance.

| | **P1 — Substrate** | **P2 — Training data** | **P3 — Evaluator** |
|---|---|---|---|
| **Spine piece** (shared, built once) | Prices/returns + rolling PCA + residuals + the shared clustering/pair-builder/spread-z machinery | The trigger → label → feature → split dataset builder (all models train on its output) | Backtest engine + costs + runner + metrics/bootstrap + freeze protocol + both fixtures' consumers |
| **Stage 1 — pairing** | **Track A**: cluster on factor-beta vectors → `pairs_a` | **Track B**: Bloomberg characteristics → `pairs_b` | **Track C**: partial-correlation distance → `pairs_c` |
| **Stage 2 — model** | **E2** (GDA, numpy, LDA-verified) | **E1** (logistic — the pre-registered primary model) | **E3** (small MLP) + the trivial **E0** row |
| **Stage 3 — backtest/eval** | E2 × {a,b,c} rows; calibration; **manual trace** (a non-author hand-verifies P3's engine); audit items 1–4 | E1 × {a,b,c} rows; base rate + decay figure; coefficient tables; consensus-lite; audit items 5–6, 10 | E3 + E0 × {a,b,c} rows; noise test; cost sweep; turnover-control machinery; audit items 7–9, 11 |
| **~Hours (Days 1–4)** | ~33 | ~34 | ~34 |

Every model flows through P3's engine and runner via the shared `EntryModel` API and decisions-table contract; every model trains on P2's trigger tables; every pairing track flows through P1's pair-builder and spread/z module. That triangle of mutual dependence is deliberate — it is the fair-comparison guarantee *and* the reason each person ends up reading the other two slices' interfaces.

### 1.2 File ownership (merge rights follow ownership)

| Files (`src/` unless noted) | Owner |
|---|---|
| `data.py`, `representation.py`, `models/e2.py`, `scripts/checks/`, `scripts/audit/` items 1–4, `scripts/figures/plotstyle.py` | P1 |
| `characteristics.py`, `dataset.py`, `models/e1.py`, `analysis.py`, `fixture_zscores.py`, audit items 5–6 + 10 | P2 |
| `engine.py`, `metrics.py`, `experiments.py`, `partial_corr.py`, `models/common.py` + `e3.py`, `contracts.py`, `make_synthetic.py`, `Makefile`, audit items 7–9 + 11 | P3 |
| `config.py`, `DECISIONS.md`, `docs/limitations.md` | all three (P1 scribes; change control per §2.3) |

### 1.3 The dependency spine (what actually blocks what, and the fixture that de-fangs it)

```
P1: returns.parquet (Day 1 midday) → factors + corr_windows.npz (Day 1 EOD)
   └─→ P1: residuals (Day 2 midday)
          ├─→ P1: clusters → pairs_a → zscores_a (Day 2 EOD)
          ├─→ P2: pairs_b (Day 2 EOD) ─→ P1: zscores_b (Day 2 EOD / Day 3 first thing)
          └─→ P3: Track C (Day 3 AM, from corr_windows.npz) → pairs_c at the midday checkpoint → zscores_c (P1 runs module) → triggers_c (P2 runs builder) by ~2–3pm
P2: real triggers_a + triggers_b (Day 3 AM) ← Integration checkpoint, Day 3 midday
   └─→ all three: fit/tune own model, τ, 3×4 grid on train+val (Day 3 PM) ← Validation gate, Day 3 evening
P3: freeze protocol → witnessed test run (Day 4 ~noon) → controls/figures → results freeze (Day 4 EOD)
```

Until real artifacts land, everyone codes against fixtures: P2's planted-OU fixture (with golden triggers and fixture prices) serves P2's own labeling code *and* P3's engine golden tests from Day 1; P3's random-walk fixture doubles as the Day 3 noise test. P1 goes straight to real data (returns land Day 1).

### 1.4 Decisions ratified in v1 that still stand, and the v2 deltas

All of v1's frozen technical decisions carry over unchanged: log returns; 252-day PCA window with the in-window beta reconstruction; download 2014-01-01 → 2025-01-01 (2014 = warm-up year); k-means/silhouette settings; the run/burn-in spread policy (`carry_with_burnin`, 60d); trigger 2.0 / reversion 50% / horizon 5d; splits 2015–20 / 2021–22 / 2023–24 with purge 5d + embargo 10d; t+1 execution with the disclosed exit-at-signal-close convention; cost grid {0…50} bps, headline c=10; the pre-registered τ rule; the pre-registered primary comparison **Track A × E1 vs Track A × E0**; seed 311.

| # | v2 delta | Decision |
|---|---|---|
| V1 | Ownership model | Vertical slices P1/P2/P3 with a shared, built-once spine (engine, dataset builder, metrics) |
| V2 | Track C | **Committed** third pairing method; grid = 3×4 = 12 cells. De-scope rule: if P3 is behind at the Day 3 midday checkpoint, Track C ships validation-only (or is cut, logged) and never blocks the Day 4 test run |
| V3 | Track D | **Cut.** One lit-review sentence per spec 8.4. Multiple-testing arithmetic now for 12 cells |
| V4 | Timeline | 4-day build, Aug 2–5; results freeze Day 4 EOD; single witnessed test run Day 4 ~noon; runner date-guard moves to 2026-08-05 |
| V5 | Write-up | Aug 6–7, planned separately after results — no prose drafting scheduled inside Days 1–4 (figures/tables are still Day 3–4 evaluation outputs) |
| V6 | 12e error analysis | Cut (the deep-dive); coefficient comparison and base-rate decay stay |
| V7 | Syncs | Two per day: midday 15 min + evening 30 min (9pm EST) |

---

## 2. Architecture, contracts, and engineering conventions

This section is the backbone every other section builds on: the repo layout, the authoritative artifact schemas, the frozen `src/config.py`, the fixtures that let all three people code in parallel from Day 1, the test harness, and the collaboration rules. Anything ambiguous elsewhere in the Plan resolves to what is written here. Owners are the slice owners P1/P2/P3 (Sections 3–5); the shared scientific instruments (engine, dataset builder, metrics/bootstrap, pair-builder, spread module, control machinery) are **built once** by their listed owner and consumed by everyone — that single-machinery rule is what keeps the 3×4 factorial comparison fair and the turnover-matched control valid.

### 2.1 Repository layout

```
pair_trading/
├── Makefile                  # all pipeline entry points (see 2.5) — owner: P3
├── README.md                 # 1-page quickstart: conda env create, make data, make test — owner: P3
├── DECISIONS.md              # append-only decision log (see 2.6) — owner: all three (P1 scribe)
├── environment.yml           # conda env: python 3.12 + pinned pip deps (see 2.7/7.9; committed by P3 at repo init)
├── .gitignore                # results/ (except results/final/), __pycache__, .ipynb_checkpoints
├── src/                      # ~16 files, ONE owner each; "modular" here means people, not files
│   ├── config.py             # THE frozen config (2.3) — all three; P1 scribe
│   ├── contracts.py          # schemas + validate_artifact() + write/read_parquet + seed_everything — P3
│   ├── data.py               # universe + yfinance download + cleaning + log returns — P1  [P1.1]
│   ├── representation.py     # P1's whole substrate: rolling PCA, residuals, corr_windows,
│   │                         #   SHARED k-means/co-membership/stability, SHARED pair-builder,
│   │                         #   SHARED spread/z-score — one owner, one file  [P1.2–P1.7]
│   ├── characteristics.py    # Track B approach file: Bloomberg load / fallback + 4B.2 cleaning + char-PCA — P2  [P2.6–P2.7]
│   ├── dataset.py            # P2's whole dataset builder: triggers + labels, the 7 features,
│   │                         #   splits/purge/embargo, assembly CLI — one owner, one file  [P2.2–P2.5]
│   ├── partial_corr.py       # Track C approach file: partial-correlation distance (reads corr_windows.npz) — P3  [P3.5]
│   ├── models/               # one file per model + the base; per-FILE owners
│   │   ├── common.py         # EntryModel base + E0 decisions + the freeze harness — P3  [P3.3, P3.10]
│   │   ├── e1.py             # logistic regression — P2  [P2.8]
│   │   ├── e2.py             # numpy GDA (LDA-verified) — P1  [P1.8]
│   │   └── e3.py             # PyTorch MLP — P3  [P3.6]
│   ├── engine.py             # the single backtest engine + trade ledger + cost columns — P3  [P3.2]
│   ├── metrics.py            # metrics + calibration + bootstrap — P3  [P3.4]
│   ├── experiments.py        # runner (grid/test subcommands, guarded) + noise test + cost sweep
│   │                         #   + turnover-control machinery — P3  [P3.3, P3.7–P3.9]
│   ├── analysis.py           # base rate (12e) + consensus-lite (12c) — P2  [P2.9, P2.11]
│   ├── make_synthetic.py     # random-walk fixture (doubles as the noise-test input) — P3  [P3.1]
│   └── fixture_zscores.py    # planted-OU fixture + golden triggers — P2  [P2.1]
├── scripts/                  # runnable scripts: checks/ + audit/ (items 1-4 P1, 5-6+10 P2, 7-9+11 P3), figures/ (+ plotstyle.py helper, P1)
├── tests/                    # one test file per src file (focused groups live inside them) + golden/ + leakage_utils.py
├── data/                     # committed artifacts (raw/, processed/, clusters/, pairs/, spreads/, datasets/, synth/)
├── results/                  # gitignored except results/final/ and results/figures/final/; results/FREEZE.md written Day 4 AM
├── docs/                     # leakage checklist copy, manual-trace worksheet, limitations.md — owner: all
└── notebooks/                # EXPLORATION ONLY — owner: individual
```

**Notebooks policy (hard rule):** notebooks never contain pipeline logic and are never imported. Anything a notebook discovers gets promoted into a `src/` module with a test before any other person depends on it. Every artifact in `data/` and `results/` must be reproducible by a `make` target alone.

**Why this shape:** the repo is deliberately coarse — **one file per owner per role**, ~16 files, so the whole codebase fits in a single `ls src/`. "Modular" in this project means *people*, not files: the two guarantees that let three of you work without colliding are (a) **every file has exactly one owner** (§1.2) and files with different owners are never merged, and (b) **hand-offs are artifact contracts** (§2.2), never imports of each other's half-written code. Within one owner, big files are fine — `representation.py` holds P1's entire returns→z-scores pipeline and conflicts with no one. The *approach files* (`characteristics.py`, `partial_corr.py`, `models/e*.py`) isolate what genuinely differs between tracks and models; the *shared instruments* (the clustering/pair-builder/spread machinery inside `representation.py`, the dataset builder, `engine.py`, `metrics.py`) exist once each — the fair-comparison guarantee. `models/` is the only subpackage; no new `src/` file without a `DECISIONS.md` entry.

### 2.2 Artifact contracts

These tables are the authoritative interface reference; other Plan sections point here rather than redefining columns. All parquet files use a `pandas.DatetimeIndex` named `date` (trading days, strictly increasing, tz-naive) unless the schema is listed as *long* (then `date`/`window_end` is an ordinary column). All floats are `float64`; identifiers are `string`. Producers must call `contracts.validate_artifact(df, name)` before writing (see 2.5).

#### `data/raw/prices.parquet`, `data/raw/volume.parquet`, `data/raw/spy.parquet`
| column | dtype | meaning |
|---|---|---|
| *(index)* `date` | datetime64 | trading day |
| one col per ticker (40) | float64 | adjusted close (prices) / share volume (volume) |
| `SPY` (spy.parquet only) | float64 | SPY adjusted close |

Producer **P1**, lands **Day 1**; refreshed never (frozen after download). Consumers: `src/data.py` (returns), `src/representation.py` (PC1-vs-SPY check), `src/dataset.py` (relative volume), `src/engine.py` (raw-return P&L).

#### `data/raw/universe.csv` *(glue artifact)*
| column | dtype | meaning |
|---|---|---|
| `ticker` | string | one of the 40 tickers, post-cleaning survivors flagged |
| `sector` | string | one of the 4 sectors |
| `included` | bool | False if dropped by the >2%-missing rule (replacement noted in DECISIONS.md) |

Producer **P1**, ratified at kickoff, lands **Day 1**. Consumers: everyone (canonical ticker list + ordering).

#### `data/processed/returns.parquet`
| column | dtype | meaning |
|---|---|---|
| *(index)* `date` | datetime64 | trading day |
| one col per ticker | float64 | daily **log** return `ln(P_t/P_{t-1})` |

Producer **P1**, **Day 1 EOD**; never refreshed. Consumers: P1 (`factors`), P2 (Track B fallback fields if Bloomberg is unavailable), P3 (the engine uses *raw* returns reconstructed from prices for P&L — `engine.py` reads `prices.parquet`, not this file; see Section 5).

#### `data/processed/factors_a.parquet` / `pca_meta.parquet` / `loadings_a.parquet`
| file | column | dtype | meaning |
|---|---|---|---|
| factors_a | *(index)* `date`; `pc_1..pc_5` | float64 | daily eigenportfolio factor returns — **all 5 columns stored every day**; `pca_meta.n_components` records that day's kept m (storing all 5 avoids NaN design matrices downstream) |
| pca_meta | *(index)* `date`; `n_components` | int64 | m chosen that day |
| pca_meta | `cum_var_explained` | float64 | cumulative variance explained by the m kept components |
| loadings_a *(long)* | `date`, `ticker`, `component` | datetime64/string/int64 | which window / stock / PC |
| loadings_a | `loading` | float64 | sign-fixed eigenvector element |
| loadings_a | `beta` | float64 | OLS beta of stock on that factor, trailing window |

Producer **P1**: `factors_a` + `pca_meta` + `loadings_a` (loading column) land **Day 1 EOD** (the PCA runs on real returns Day 1); the `beta` column and `residuals_a` land **Day 2 midday** with P1.3. Refreshed only on bugfix. Consumers: P1 (`clustering` — Track A cluster input is the beta vectors; `spreads` indirectly), P2 (`features`: `f_mkt_vol_20d` from `pc_1`).

#### `data/processed/corr_windows.npz` *(glue artifact — Track C's input)*
One 40×40 in-window correlation matrix per recluster window, keyed by `window_end` date string. Producer **P1** (written by the same Day 1 PCA loop; small file). Consumer **P3** (`partial_corr.py` reads the file directly — the hand-off is an artifact, not a call into P1's code). Lands **Day 1 EOD**.

#### `data/processed/residuals_a.parquet`
Same shape/schema as `returns.parquet`; each cell is the **out-of-sample** residual return at *t* using betas estimated on the trailing window excluding *t* (betas regress on factor returns **reconstructed from the current window's eigenvectors** — see Section 3 — so residuals begin ~253 trading days after `DOWNLOAD_START`, i.e. early 2015). Producer **P1**, **Day 2 midday**. (`residuals_d` removed in v2 — Track D is cut.) Consumers: P1 (`spreads`), P2 (`features`: `f_resid_mom_5d`).

#### The `EntryModel` API *(the one non-artifact contract, stated here because every slice touches it)*
`src/models/common.py`, owner **P3**, lands **Day 2** alongside the runner. Abstract base every model subclasses (P2 → E1, P1 → E2, P3 → E3; E0 bypasses it as a plain decisions generator):

```python
class EntryModel(abc.ABC):
    name: str                                        # "e1" | "e2" | "e3"
    def fit(self, X_train, y_train) -> "EntryModel"  # fits StandardScaler on TRAIN rows only, then the model; returns self
    def tune(self, X_val, y_val) -> dict             # hyperparam selection by validation AUC; refits at best config; returns report
    def predict_proba(self, X) -> np.ndarray         # shape (n,), P(label=1); applies the stored scaler
    def get_params_report(self) -> dict
    def save(self, path) -> None                     # model + fitted scaler in ONE bundle — what the freeze protocol hashes
    @classmethod
    def load(cls, path) -> "EntryModel"
```

Subclasses never re-implement scaling or persistence; the runner and freeze protocol call only this surface. The test-run single-shot marker is also contracted here: **`results/final/TEST_RUN_COMPLETE`** (timestamp + git sha), written by the runner itself as the last act of a successful `--split test` run — it lives in the committed `results/final/` path, so deleting it to rerun is a visible, logged team decision.

#### `data/raw/characteristics_raw.parquet` / `data/processed/characteristics_clean.parquet` *(glue artifacts, Track B)*
| column | dtype | meaning |
|---|---|---|
| `quarter_end` | datetime64 | quarter the snapshot applies from |
| `ticker` | string | stock |
| one col per field | float64 | raw: 19 Bloomberg fields (or the disclosed fallback columns: sector one-hots + price-derived fields, per the Day 1 evening Bloomberg decision); clean: sentiment merged, sizes logged, z-scored per Section 4 |

Producer **P2**; raw **and** clean land **Day 2** (Bloomberg pull Day 2 = Monday AM if terminal access was confirmed at the Day 1 evening sync, with the mid-session noon abort per P2.6; otherwise the fallback pipeline starts immediately — no waiting for Tuesday). Consumers: P2 (characteristics PCA, `clustering` for track b).

#### `data/clusters/labels_{track}.parquet` *(long)* and `stability_{track}.parquet` *(long)*
| file | column | dtype | meaning |
|---|---|---|---|
| labels | `window_end` | datetime64 | last day of the formation window |
| labels | `ticker` | string | stock |
| labels | `cluster_id` | int64 | k-means label (window-local; never compared across windows) |
| stability | `window_end` | datetime64 | as above |
| stability | `pair_id` | string | `AAA__BBB`, alphabetical |
| stability | `co_clustered` | bool | pair co-clustered in this window AND the previous window |

All three tracks run P1's shared clustering/stability machinery. Producers and landings: track **a** — **P1, Day 2** (21-trading-day cadence); track **b** — **P2, Day 2** (quarterly); track **c** — **P3, Day 3 PM** (21-trading-day cadence). (Track d removed in v2.) Consumers: each track owner (`pairs`), P2 (`features`: `f_cluster_stability`).

#### `data/pairs/pairs_{track}.csv`
| column | dtype | meaning |
|---|---|---|
| `pair_id` | string | `AAA__BBB`, tickers alphabetical — the universal join key |
| `stock_a`, `stock_b` | string | legs; `stock_a` < `stock_b` alphabetically |
| `group_id` | string | `{window_end:%Y%m%d}_{cluster}_{subgroup}` |
| `source` | string | `track_a` / `track_b` / `track_c` *(track_d branch removed in v2)* |
| `active_from`, `active_to` | date | window over which the pair is live (recluster to recluster) |

Producer: the shared pair-builder `src/representation.py` — **canonical frozen signature (authoritative here; Sections 3–5 refer back to this)**:

```python
def build_pairs(labels_path: str,
                features_by_window: dict[pd.Timestamp, pd.DataFrame],  # per-window feature matrices -> within-cluster distances for the 5+ split
                source: str,                    # "track_a" | "track_b" | "track_c"  (track_d removed in v2)
                calendar: pd.DatetimeIndex,
                out_csv: str) -> pd.DataFrame
```

Owner **P1**. Signature stub (raises `NotImplementedError`) committed at kickoff so P2 and P3 can import and mock it from Day 1; implementation tests-green by the **Day 2 midday sync**; called by P2 for track b Day 2 PM and by P3 for track c Day 3 PM. Lands: **a Day 2 EOD (P1), b Day 2 EOD (P2), c Day 3 PM (P3)**. Consumers: P1 (`spreads`, all tracks), P2 (`analysis/consensus-lite`), each track owner (trigger attribution).

#### `data/spreads/spreads_{track}.parquet` and `zscores_{track}.parquet`
Wide: *(index)* `date` × one float64 column per `pair_id`. Consecutive active windows for the same `pair_id` tile into **runs** (Section 3): within a run the spread accumulates continuously across window boundaries, and each run starts with a 60-trading-day backward-looking **burn-in** ending at `active_from` — so z is valid from a run's first active day (`SPREAD_POLICY="carry_with_burnin"`, `SPREAD_WARMUP_DAYS=60` in config). z is NaN **only outside active windows**; burn-in dates are internal and never emitted. Producer **P1** runs the shared module for every track's pair list as it lands: track **a Day 2 EOD**, track **b Day 2 EOD or first thing Day 3 AM** (from P2's `pairs_b`), track **c Day 3 PM** (from P3's `pairs_c`, one command). Consumers: P2 (`dataset.py`, trigger detection), P3 (`engine.py` exit rule).

#### `data/datasets/triggers_{track}.parquet` — the central contract
| column | dtype | meaning |
|---|---|---|
| `trigger_id` | string | `{pair_id}__{trigger_date:%Y%m%d}` — unique key (double underscore, matching the `AAA__BBB` pair_id style; encoded in `validate_artifact`) |
| `pair_id`, `source` | string | pair + track provenance |
| `trigger_date` | datetime64 | day \|z\| first crossed 2.0 from below |
| `z_trigger` | float64 | **signed** z at trigger |
| `f_abs_z` | float64 | feature: \|z_trigger\| |
| `f_spread_vol_60d` | float64 | feature: std of daily spread *changes* (Δspread) over the trailing 60d at trigger (Section 4's exact formula; the deviation from spec Step 8's literal "std of spread" is disclosed in methods) |
| `f_resid_mom_5d` | float64 | feature: `sign(z_trigger) ×` 5d sum of (resid_A − resid_B) — positive = still widening (Section 4) |
| `f_mkt_vol_20d` | float64 | feature: rolling 20d std of `pc_1` |
| `f_rel_volume_20d` | float64 | feature: mean of both legs' volume ÷ their 20d averages |
| `f_days_since_trigger` | float64 | feature: trading days since this pair's previous trigger (capped/NaN-policy in Section 4) |
| `f_cluster_stability` | float64 | feature: `co_clustered` (0/1) from latest `stability_{track}` row ≤ trigger_date |
| `label` | int8 | 1 iff \|z\| ≤ 0.5·\|z_trigger\| within next 5 trading days |
| `horizon_end_date` | datetime64 | trigger_date + 5 trading days |
| `split` | string | `train` / `val` / `test` / `purged` / `embargo` |

Producer **P2** (one builder, all tracks); fixture version **Day 1**, real **tracks a + b Day 3 AM** (reviewed at the midday Integration Checkpoint), **track c Day 3 PM** (one command on P3's zscores_c). Consumers: all model owners (`models/`, via P3's runner), P3 (`metrics.py`, `engine.py`, control matching).

#### `results/decisions_{track}_{model}.parquet`
| column | dtype | meaning |
|---|---|---|
| `trigger_id` | string | FK into triggers table |
| `enter` | bool | E0: always True; E1–E3: `p_hat > τ`; control: random matched draw |
| `p_hat` | float64 | model probability; NaN for E0 and the control |

`{model}` ∈ `e0, e1, e2, e3, ctrl-e1, ctrl-e2, ctrl-e3` (controls name the model they are matched to). Produced by the runner (`src/experiments.py`): `e0_decisions()` for E0, and each owner's `EntryModel` subclass via `load(...).predict_proba` + τ from `taus.json` for E1–E3 — run by **each model's owner**: P2 (e1), P1 (e2), P3 (e0, e3). Controls use P3's turnover-matched-control machinery, run by each model's owner for their own model (P3 for e3; E0 needs none). Lands: **Day 3 PM (validation)**, **Day 4 (test, witnessed run only)**. Consumers: P3 (`engine.py`).

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

Producer **P3**'s engine: fixtures **Day 1**, real **Day 3 PM (validation)** / **Day 4 (test)**. Consumers: P3 (`metrics.py`, metrics JSONs).

#### `results/metrics_{track}_{model}.json`
Nested JSON: `{"classification": {auc, precision_at_tau, recall_at_tau, brier, calibration_bins}, "strategy": {hit_rate, mean_ret_per_trade, cum_ret, sharpe, max_drawdown, n_trades, per_cost: {c: {...}}}, "ci": {...bootstrap 95% for each}, "meta": {seed, config_hash, git_sha, split}}`. Producer: P3's `metrics.py` called from the runner (`src/experiments.py`) (every one of the 12 grid cells is scored by this identical machinery). Lands **Day 3 PM (validation)** / **Day 4 (test)**. The `config_hash` + `git_sha` fields make every result file traceable to the exact code and config that produced it.

### 2.3 `src/config.py` (full draft)

```python
"""Frozen project configuration — CSC311 pairs-trading project (plan v2).

CHANGE CONTROL: every value in this file was ratified at the Day 1 kickoff
(Sat 2026-08-01). No value may change without (a) explicit agreement of all
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
DATA_END: str = "2025-01-01"    # frozen — ratified in v1; test = 2023-2024 matches the split dates
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
RECLUSTER_EVERY: int = 21       # trading days, tracks a/c, trailing 252d formation window
TRACK_B_REFRESH: str = "quarterly"
KMEANS_N_INIT: int = 10         # k-means++ init; k by max silhouette, formation window ONLY
K_RANGE: dict[str, range] = {
    "a": range(8, 14), "b": range(10, 14), "c": range(8, 14),  # track d removed in v2
}
# Track A cluster input: each stock's factor-beta vector from the formation window
# Pair rules (Zhang): drop singletons; 2-4 -> all pairs; 5+ -> greedy NN subgroups of 2-3
MAX_WHOLE_CLUSTER: int = 4
SUBGROUP_SIZES: tuple[int, int] = (2, 3)

# ----------------------------------------------------------------- Step 6 (spread / z)
SPREAD_KIND: str = "simple"     # spread_t = cumsum over the pair's RUN of (resid_A - resid_B)
Z_WINDOW: int = 60              # trailing rolling mean/std, window-local
SPREAD_POLICY: str = "carry_with_burnin"  # runs tile across recluster windows; see Section 3
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
# NOTE: tau lives in results/frozen/taus.json (sole authority, written by each model's
# owner on Day 3 evening via the one pre-registered rule in Section 5) — deliberately
# NOT in this file, so config can never disagree with what the runner actually reads.
                                     # each set Day 3 evening by that model's owner via the ONE
TAU_RULE: str = "see-plan-sec-5"     # pre-registered validation rule (unchanged from v1);
                                     # never touched on test
E3_HIDDEN: tuple[int, ...] = (16,)   # small MLP; details in Section 5 (Slice P3)
TORCH_DEVICE: str = "cpu"            # NEVER autodetect cuda — reproducibility requires same-device compute everywhere

# ----------------------------------------------------------------- Step 11 (execution / costs)
ENTRY_LAG_DAYS: int = 1         # enter at close of trigger_date + 1 trading day
EXIT_Z: float = 0.5             # exit when |z| < 0.5, or ...
MAX_HOLD_DAYS: int = 5          # ... 5 trading days after entry, whichever first
NOTIONAL_PER_LEG: float = 1.0   # long lagging / short leading leg by sign of z
COST_GRID_BPS: tuple[int, ...] = (0, 5, 10, 15, 20, 30, 40, 50)  # per leg per transaction
HEADLINE_COST_BPS: int = 10

# ----------------------------------------------------------------- Track C (committed in v2)
PARTIAL_CORR_SHRINKAGE: float = 1e-3   # ridge; in-window only (leakage item 11)
# (Track D constants AE_BOTTLENECK / AE_RETRAIN_EVERY removed in v2 — Track D is cut)

# ----------------------------------------------------------------- pre-registration
PRIMARY_COMPARISON: tuple[str, str] = ("track_a__e1", "track_a__e0")
```

**Task (P1 as scribe, Day 1, 0.5h):** commit this file verbatim after the kickoff read-aloud. Consumes: kickoff decisions. Produces: `src/config.py`. Done when: `python -c "import src.config"` passes and every teammate has confirmed the values match the freeze.

### 2.4 Fixtures — the parallelism enablers (both land Day 1)

#### `src/make_synthetic.py` — pure-noise prices (also the Day 3 noise-test input)
**Task (P3, Day 1, ~1.5h — P3 consumes it first: it feeds engine development Day 1–2 and the Day 3 noise test P3 owns; P1 no longer needs it, since real returns land Day 1 and P1 unit-tests PCA on tiny inline frames; schemas fixed by 2.2).** Consumes: nothing (seeded RNG only). Produces: `data/synth/raw/prices.parquet`, `volume.parquet`, `spy.parquet` — **byte-for-byte the same schemas as the real raw artifacts**, which is what makes the Day 3 noise test trivial: point the whole pipeline at `data/synth/raw/` instead of `data/raw/` and change nothing else. Done when: schema contract tests pass on its outputs and two runs with the same seed are identical.

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

CLI: `python -m src.make_synthetic --seed 311 --out data/synth/raw` (argparse mirrors the signature). The `make noise-test` target (2.5) reruns it with the frozen seed before piping it through the full pipeline.

#### `src/fixture_zscores.py` — planted-OU pairs + golden triggers
**Task (P2, Day 1, ~2.5h — P2 and P3 consume it throughout Days 1–3: it is the golden input for P2's trigger/label code and for P3's engine ledger and runner APIs; the golden file is hand-derived from the exported z-matrix, never from the labeling code it will test).** Consumes: nothing. Produces, under `data/synth/fixture/`: `zscores_a.parquet` and `spreads_a.parquet` (date × 6 pair_ids), **`prices.parquet`**, `residuals_a.parquet` and `volume.parquet` for the 12 fake leg tickers (leg prices are seeded walks whose returns embed the planted residual differences — this is what lets P3's engine compute P&L on the fixture, and the 3-trade golden ledger is hand-computed from these prices), `factors_a.parquet` (a fake `pc_1` series for `f_mkt_vol_20d`), `pairs_a.csv` — **including one pair whose rows tile across two consecutive active windows**, exercising the run/burn-in convention and the cross-boundary trigger case — `stability_a.parquet`, and **`tests/golden/golden_triggers.csv`** (`trigger_id, pair_id, trigger_date, z_trigger, label, horizon_end_date`). The calendar is pinned (start 2019-07-01, 500 trading days, through ~mid-2021) so it **spans the train/val boundary**; a contract test asserts the fixture triggers table contains both train and val rows — otherwise the model owners' `tune()` and τ dry-runs would silently exercise nothing. Done when: P2's trigger/label code reproduces the golden file exactly (`make test` includes this as a contract test).

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
      F1-F3 'fast OU':  z_{t+1} = 0.55*z_t + N(0,0.4) with 4 injected
             excursions each to |z|>2  -> reverts within 5d: labels mostly 1
      F4-F5 'slow OU':  z_{t+1} = 0.97*z_t + N(0,0.3), excursions decay
             slower than the 5d horizon -> labels mostly 0
      F6    'break':    one excursion crosses 2.0 and RAMPS to 4 and stays
             -> exactly one trigger, label 0 (tests onset-only: no re-trigger
             while |z| stays above 2)
    Spreads are back-computed as z * (fixed sigma) + fixed mean; residuals as
    spread first-differences split across the two legs; volume/pc_1 are
    seeded noise so all 7 features are computable.
    """
```

**Golden-file derivation:** the golden CSV is **not** produced by the labeling code it will test. P2 exports the z-matrix to CSV and scans it manually (spreadsheet threshold-crossing pass: mark rows where prev |z|<2 and current |z|≥2, then check the next 5 rows against `0.5*|z_trigger|`), records every trigger and label by hand, and commits the result with a note in the file header naming who verified it. Expected count: ~15–20 triggers — small enough to hand-check in ~30 minutes, large enough to exercise both labels, both signs of z, and the onset-only rule.

### 2.5 Testing strategy

`tests/` holds one test file per `src/` file (`tests/test_representation.py` ↔ `src/representation.py`, etc.), plus the shared leakage helper `tests/leakage_utils.py`; focused test groups (cost arithmetic, GDA-vs-LDA, the runner guard, the control identity) live as sections inside their file's test module.

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

Applied on fixture data (fast) to at least: **(1)** `factors.compute_factors` — factors/loadings/betas up to *t* unchanged when returns after *t* change (catches the PCA off-by-one); **(2)** `factors.compute_residuals` — same; **(3)** `spreads.compute_zscores` — z up to *t* unchanged (catches full-sample rolling stats); and, via Section 4, the feature builder. Each is one ~10-line test calling the helper. (Owner: each stage's owner; the helper itself: **P1, Day 2 AM, 1h** — P1.3's residual tests are its first consumer that morning, and P2's Day 2 feature look-ahead tests use it the same afternoon.)

#### Schema contract tests (`src/contracts.py` + `tests/test_contracts.py`)
`src/contracts.py` holds one `ArtifactSchema(columns: dict[str, str], index: str | None, checks: list[Callable])` per artifact in 2.2, and `validate_artifact(df, name)` which asserts: exact column set, exact dtypes, index monotonic-increasing and unique, and per-artifact invariants (pair_id alphabetical and matching `stock_a`/`stock_b`; `split` ∈ the 5 allowed values; `label` ∈ {0,1}; every `net_ret_{c}bps` column present for the full cost grid). Producers call it before every write; `tests/test_contracts.py` runs it against the committed fixture artifacts as goldens. Done when: deliberately renaming a column in a fixture file fails the suite. (Owner: **P3, Day 1, 1.5h**.)

#### Property tests worth having (owners as per module)
- **Sign-fix determinism** (P1): for random symmetric matrices, `fix_signs(eigvecs)` output is invariant to pre-negating any eigenvector; every output column sums > 0.
- **Purge/embargo correctness** (P2): no row with `split == "train"` has `horizon_end_date` past the train boundary; no `val`/`test` row's `trigger_date` falls inside an embargo window; `purged`/`embargo` rows are never consumed by fit or metrics code.
- **Onset-only triggers** (P2): on the fixture, consecutive days with |z| ≥ 2 yield exactly one trigger (pair F6 covers this).
- **Decisions ⊆ triggers** (P3): every `trigger_id` in a decisions file exists in the triggers table; E0 decisions have `enter` all-True and `p_hat` all-NaN.
- **Cost monotonicity** (P3): for every trade, `net_ret_{c}bps` is strictly decreasing in c, and `net_ret_0bps == gross_ret`.
- **Determinism** (all): running any stage twice with `SEED=311` produces identical files.

#### Make targets (the only sanctioned pipeline entry points; owner P3, Day 1–4)
| target | runs |
|---|---|
| `make data` | download + clean + `returns.parquet` + fixture generation (Steps 0–1, 2.4) |
| `make tracka` | factors → residuals → clusters/stability → pairs_a → spreads/zscores_a (Steps 2–6, track a) |
| `make trackb` | characteristics clean → char-PCA → clusters_b → pairs_b → spreads/zscores_b |
| `make trackc` | partial-corr distance → clusters_c → pairs_c → spreads/zscores_c *(new row in v2 — Track C is committed)* |
| `make dataset` | triggers_{track}.parquet for all available tracks (Steps 7–8 + splits) |
| `make grid` | full experiments: decisions → engine → metrics for every (track, model) cell, **train+val only** |
| `make test-run` | the **Day 4 ~noon witnessed run only** — the guarded path: `--i-am-sure`, date ≥ **2026-08-05**, `results/FREEZE.md` present (written Day 4 AM; see Section 5), no `results/final/TEST_RUN_COMPLETE` marker |
| `make noise-test` | regenerates synthetic raw data, runs `data`→`tracka`→`dataset`→`grid` against `data/synth/raw/`, asserts the no-signal criteria (Section 5 binding criteria, all four models; pass/fail printed) |
| `make test` | `pytest -q` — the merge gate |
| `make figures` | all figures from committed artifacts into `results/figures/` (Agg backend) |

### 2.6 Git and collaboration conventions

- **Repo init Day 1** during kickoff: private GitHub repo, `main` protected (no direct pushes), all three added as admins. First commit = skeleton tree of 2.1 + `config.py` + `contracts.py` stubs (schemas + io helpers) + Makefile + the frozen `build_pairs` signature stub (raises `NotImplementedError`; P2 and P3 mock against it from Day 1) + `docs/limitations.md` seeded with every spec Part 7 bullet as an owner-tagged stub (so the append-on-compromise mechanism only ever *adds* items — known disclosures never depend on memory).
- **Branches:** branch-per-person, prefixed `p1/`, `p2/`, `p3/` (e.g. `p1/rolling-pca`, `p2/labeling`). Short-lived; rebase on `main` daily.
- **File ownership = merge rights.** The owner tagged on each file in 2.1 is the only person who approves PRs touching that file (shared modules — `representation.py`'s clustering/pair-builder/spread machinery, the dataset builder, `engine.py`, `experiments.py`, `metrics.py` — belong to their single listed builder; callers PR against them, owners approve). PRs touching multiple owners' files need each touched owner's approval. This replaces heavyweight review — approvals at the syncs, in person.
- **Merge cadence:** merges to `main` happen at **both daily syncs** — a quick-merge at the midday sync is allowed if `make test` passes locally on the branch; the evening sync is the full merge. Broken `main` is an all-stop: whoever broke it fixes or reverts before anything else merges.
- **`DECISIONS.md`:** append-only, one line per decision: `YYYY-MM-DD | decision (old -> new where applicable) | initials of all agreeing`. Required for: any `config.py` change, any contract/schema change, universe substitutions, the Bloomberg-vs-fallback Day 1 evening decision, the Track C de-scope decision (Day 3 midday checkpoint), each model's τ (set Day 3 evening by its owner), and the model-freeze hashes (Day 4 AM). Scribe: P1.
- **Data policy:** raw pulls **are committed** — 40 tickers × 10y of prices/volume is a few MB and the characteristics table is tiny — so all three machines reproduce byte-identical pipelines with no re-download drift. `data/` is therefore in git; a `data/raw/PROVENANCE.md` records pull timestamp and yfinance version.
- **Results policy:** `results/` is gitignored **except** `results/final/` (the Day 4 frozen test-run outputs: decisions, trades, metrics JSONs, with `git_sha` + `config_hash` embedded), `results/FREEZE.md` (written Day 4 AM before the witnessed run), and `results/figures/final/`. Nothing lands in `results/final/` before the witnessed Day 4 run.

### 2.7 Environment

Python **3.12** via **conda** (Miniforge, Miniconda, or Anaconda — whatever each person already has): `conda env create -f environment.yml && conda activate pair-trading` — one command, torch included via the `+cpu` pin behind the extra CPU wheel index (7.9). The pins live in `environment.yml`'s **pip section** — pip-inside-conda deliberately, because the exact version pins were verified against PyPI and `yfinance`/`curl_cffi` are pip-native; conda-forge equivalents would drift from the verified set. torch is ONE version team-wide — `2.2.2`, the Intel-mac ceiling (see 7.5) — with platform markers; the linux `+cpu` pin is only satisfiable from the CPU wheel index, so nobody can accidentally download CUDA. Packages: `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `statsmodels`, `torch` (CPU), `yfinance`, `matplotlib`, `pytest`.

**WSL2 note:** no display server — every plotting module sets `matplotlib.use("Agg")` before any pyplot import (enforced by the two-line helper `src.plotstyle.apply_style()` (Section 7.7) that also fixes figure DPI/size), and all figures are **saved** to `results/figures/`, never `plt.show()`n. `make figures` must run headless on all three machines.

### 2.8 Day 1 kickoff agenda (1h, all-hands, TODAY — Sun Aug 2, as early as possible)

Scribe: **P1**, capturing every ratified item into `DECISIONS.md` and `config.py` live. The old v1 end-date debate (D3) is closed: download window **2014-01-01 → 2025-01-01** and the split dates in 2.3 were ratified in v1 and are stated here as frozen, not reopened.

| time | item |
|---|---|
| 0:00–0:10 | **Slice-to-person mapping.** Ratify by the two hard constraints: the person with **Bloomberg terminal access = P2**; the strongest **PyTorch person = P3**; the remainder (heaviest math, no Bloomberg needed) **= P1**. Name→slice assignments logged explicitly. |
| 0:10–0:25 | **Config freeze + universe.** P1 shares `src/config.py` on screen and reads **every value aloud** (2.3 — values unchanged from v1 except the Track D deletions). P1 presents the proposed 40 tickers (4 sectors × 10); ratify list + the substitution rule (same-sector replacement, logged). Frozen dates re-stated as ratified. Commit `config.py` + `universe.csv`; state the change-control rule (all-3 + DECISIONS.md) out loud. |
| 0:25–0:35 | **Availability + Bloomberg + rhythm.** Each person explicitly commits **~8–9h/day, Aug 2–5** — say the number out loud, per person. Assign P2 the **Bloomberg terminal-access question** (the pull is Monday morning, Day 2) with the decision rule: at the **Day 1 evening sync**, Monday access confirmed → P2 pulls Day 2 AM (mid-session abort at Monday ~noon per P2.6); not confirmed → Track B builds on the fallback columns (sector one-hots + price-derived fields from our own data) immediately, disclosed — no waiting for Tuesday. Fix the **two-syncs-per-day rhythm**: midday 15 min, evening 30 min at 9pm. |
| 0:35–0:45 | **Repo walkthrough.** The repo, branch protection, and the first skeleton commit per 2.6 (tree, `config.py`, `contracts.py` stubs (schemas + io helpers), Makefile, the frozen `build_pairs` signature stub, `docs/limitations.md` seed) are **pre-created by P3 before the kickoff** (~30 min of prep — repo creation and three environment builds do not fit a 10-minute agenda slot). At the kickoff: everyone confirms their clone works; `conda env create` runs (incl. the ~200MB CPU-torch wheel) in the background during the remaining agenda items. |
| 0:45–0:55 | **Contracts walk-through (10 min).** Each person names, from memory, every artifact they *produce* and every artifact they *consume*, including landing days per 2.2. Any hesitation = re-read the table together. This is the cheapest integration insurance we can buy. |
| 0:55–1:00 | **4-day anchors + start-of-work.** Walk the Day 1–4 anchors (Section 6), the Day 3 midday Track C de-scope checkpoint, the Day 3 evening validation gate, the Day 4 ~noon witnessed test run, and the never-cut list (purge/embargo, cost model, turnover-matched control, limitations). Confirm first tasks: P1 — real download + cleaning + returns; P2 — `fixture_zscores` + hand-computed golden, then trigger detection; P3 — `make_synthetic`, contracts/Makefile, then the engine against the fixtures. Disperse. |

Done when: `DECISIONS.md` contains the kickoff ratifications, `config.py` and `universe.csv` are on `main`, and all three have the repo cloned, the conda env built, and `make test` green on the skeleton.

---

---

## 3. Slice P1 — Substrate: representation, Track A, E2

P1 owns the substrate every other slice stands on: the price/returns data everyone builds from, the rolling-PCA factor model and out-of-sample residuals that define the tradeable object for *all three* tracks, and the three shared scientific instruments on the representation side — the k-means/co-membership/stability machinery, the pair builder, and the spread/z-score module. Each of those is **built once by P1 and run per-track by whoever owns the track** (P1 for a, P2 for b, P3 for c); that single-implementation rule is what keeps the 3×4 factorial comparison fair. The slice hangs together deliberately: the substrate plus Track A is the math-heaviest pairing work in the project, so P1 carries the *lightest* model (E2 GDA — closed-form, no hyperparameters) to compensate, and needs no Bloomberg access. P1 also carries two cross-checks on other people's work: the manual trace of P3's engine (a non-author hand-verifies the ledger) and leakage-audit items 1–4. Finally, P1 is the **config/DECISIONS scribe**: every ratified decision, config fill-in, and fallback invocation gets a dated `DECISIONS.md` entry the same day it happens.

The workstream is a strict pipeline — P1.1 → P1.2 (Day 1), P1.3 → P1.4 → P1.5 → P1.6 → P1.7 (Day 2), P1.8–P1.10 (Days 3–4). All config constants referenced below (`PCA_WINDOW`, `N_COMPONENTS` rule, `Z_WINDOW`, recluster cadence, k ranges, seed 311) live in `src/config.py` per Section 2 and are imported, never re-declared.

Module layout owned by P1:

```
src/data.py               # P1.1: universe (the table below) + download + cleaning + returns
src/representation.py     # P1.2: rolling PCA -> factors_a, pca_meta, loadings, corr_windows
                          # P1.3: rolling OLS -> residuals_a, betas (merged into loadings_a)
                          # P1.5: SHARED k-means + k selection + co-membership + stability + recluster loop
                          # P1.6: SHARED pair builder (P2 calls for track b, P3 for track c)
                          # P1.7: SHARED spread + z-score -> spreads_{t}, zscores_{t}
src/models/e2.py          # P1.8: numpy GDA behind the shared EntryModel API
scripts/checks/           # P1.4: check_scree.py check_pc1_loadings.py check_pc1_vs_spy.py check_var_over_time.py
scripts/audit/audit_p1_items_1_4.py   # P1.10
docs/manual_trace.md      # P1.9
tests/test_data.py  tests/test_representation.py  tests/test_models.py (GDA-vs-LDA section)
```

---

### P1.1 — Universe + prices, volume, SPY, returns (Day 1 AM)

**Goal.** Fix the 40-ticker universe and land real `returns.parquet` by early Day 1 PM — it is P1.2's same-day input and the whole team's calendar.

**Universe** (`src/data.py`): 4 sectors × 10 large-cap S&P 500 names, every ticker continuously listed 2014-01-01 through 2025-01-01 (one year of pre-2015 history feeds the first 252-day PCA window). Sectors chosen for beta diversity: Information Technology (high beta), Financials (rate-sensitive), Energy (commodity-driven), Consumer Staples (defensive) — this guarantees PC2/PC3 have real sector structure to find.

```python
SECTORS: dict[str, list[str]]   # the table below, as a literal
TICKERS: list[str]              # sorted flat list of 40
def get_universe() -> pd.DataFrame:
    """Returns DataFrame[ticker, sector]. Single source of truth for every module."""
```

Proposed list (ratified at kickoff; auto-validated by the cleaning checks — any ticker failing the 2%-missing or calendar checks is replaced from the same sector same-day, logged in `DECISIONS.md`):

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

All 40 were listed and in the S&P 500 well before 2014 (MPC spun off 2011, PSX 2012 — both safely pre-window). Deliberately excluded: post-2015 IPOs/spinoffs, dual-class duplicates, anything acquired/delisted mid-sample. Survivorship bias is acknowledged per spec Step 0 and Section 2; it is disclosed, not fixed.

**Prices** (`src/data.py`):

```python
def download_prices(tickers: list[str], start: str = "2014-01-01",
                    end: str = "2025-01-01", max_retries: int = 3,
                    cache_dir: str = "data/raw/cache/") -> tuple[pd.DataFrame, pd.DataFrame]:
    """yfinance pull, auto_adjust=True asserted explicitly (do not trust the default
    silently). Per-ticker retry with exponential backoff; raw per-ticker CSVs cached
    and committed, so the pull is reproducible even if yfinance data shifts later.
    Returns (prices, volume), date x ticker."""

def clean_prices(prices: pd.DataFrame, volume: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Runs every spec Step 0 check as an explicit assert + printed report."""

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """np.log(p / p.shift(1)), first row dropped. LOG_RETURNS per config."""
```

Cleaning checks — each an `assert` (hard) or a printed WARN table (review), all echoed into a report pasted into `DECISIONS.md`:

1. **≤2% missing rule:** per ticker vs the union calendar; any ticker >2% → hard fail (replacement protocol).
2. **Forward-fill isolated single-day gaps only:** fillable iff exactly 1 NaN with valid values on both sides; multi-day gaps are *not* filled.
3. **Common trading calendar:** intersect all tickers + SPY to NYSE trading days; assert identical index across all output frames.
4. **|return| > 50% flags:** manual-review table; each row needs a one-line verified-event note or the ticker is escalated. Expected count on adjusted large-caps: zero.
5. **~252 rows/year:** assert `250 <= rows_per_calendar_year <= 254` for every full year.

**Consumes:** nothing external. **Produces:** `src/data.py`, `data/raw/prices.parquet`, `volume.parquet`, `spy.parquet`, `data/processed/returns.parquet` (all per Section 2 contracts), committed cache CSVs. P2's fallback path (sector one-hots + price-derived fields) reads these same artifacts, so P1.1 landing on time is also the Bloomberg-fallback insurance.

**Tests:** `tests/test_data.py` — planted 1-day gap (filled), 3-day gap (not filled), fake +80% return (flagged); returns of a hand-priced 3-day series equal hand-computed logs.

**Hours:** 3h, Day 1 AM (list is pre-drafted above; kickoff ratifies).
**Done when:** all four parquets exist, cleaning report shows 40/40 tickers passing, shape ~2,760 × 40 including the 2014 warm-up year, P2 and P3 confirm `returns.parquet` loads.

---

### P1.2 — Rolling PCA engine (`src/representation.py`, Day 1 PM)

**Goal.** Daily rolling-window PCA per spec Step 2, producing eigenportfolio factor returns and loadings for every day t from day 253 onward. **Running on real returns by Day 1 EOD** (canonical anchor). Coding can start before P1.1's download finishes — unit tests run on hand-built toys and on P3's `make_synthetic` fixture prices, which land Day 1 too.

**Consumes:** `data/processed/returns.parquet`.
**Produces:** `data/processed/factors_a.parquet`, `pca_meta.parquet`, `loadings_a.parquet` (loading column; `beta` filled by P1.3), and `data/processed/corr_windows.npz` — the per-recluster-window 40×40 in-window correlation matrices, saved as a **Section 2.2 contracted artifact** (small; one array per window-end key). This is Track C's entire input: P3 reads the *file*, never calls into P1's code — keeping the hand-off on the artifact-contract principle. The `pca_one_window(..., return_corr=True)` hook exists only so the loop can collect them.

**Signatures:**

```python
def pca_one_window(window_returns: pd.DataFrame, return_corr: bool = False) -> dict:
    """One formation window (PCA_WINDOW x 40), EXCLUDING day t.
    Returns {"eigvals": np.ndarray, "eigvecs": np.ndarray (40 x m, sign-fixed),
             "weights": np.ndarray (40 x m, eigvec/sigma), "sigma": pd.Series,
             "n_components": int, "cum_var": float, ["corr": np.ndarray]}."""

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

**Output writing.** `factors_a.parquet`: index=date, columns `pc_1..pc_5` (all five stored every day; `pca_meta.n_components` records that day's kept m — storing all five avoids NaN design-matrix issues downstream). `pca_meta.parquet`: per date, `n_components`, `cum_var_explained`. `loadings_a.parquet` long format per the Section 2 contract (`date, ticker, component, loading, beta`), `beta=NaN` until P1.3 fills it — one `melt` per day appended to a list, `pd.concat` once at the end (do not append to parquet per-day).

**Unit tests (`tests/test_representation.py`):** (a) hand-built 6×3 toy frame with a known dominant common factor → PC1 explains most variance, loadings all positive after sign fix; (b) eigenvalues descending; (c) sum of eigenvalues ≈ 40 (trace of correlation matrix); (d) window-exclusion assertion fires on a deliberately wrong slice; (e) `choose_n_components` on crafted eigenvalue vectors hits each branch of the {3,4,5,else-5} rule; (f) determinism: two runs byte-identical (no stochasticity in PCA, but locks the sign fix).

**Hours:** 3.5h, Day 1 PM.
**Done when:** all three parquets **and `corr_windows.npz`** exist for real data by Day 1 EOD, tests pass, and `factors_a` has a value for every trading day ≥ index 252.

---

### P1.3 — Out-of-sample residuals (`src/representation.py`, Day 2 AM)

**Goal.** Per-stock OLS betas on the trailing 252d window, applied out-of-sample at t (spec Step 3).

**Consumes:** `returns.parquet`, `factors_a.parquet`, `pca_meta.parquet`.
**Produces:** `data/processed/residuals_a.parquet` (date × ticker, same shape as returns, NaN for the first 252+1 days); fills the `beta` column of `loadings_a.parquet`. Note: `residuals_a` is the residual input for **all three tracks** (track d branch removed in v2) — tracks b and c change only how pairs are grouped, not the tradeable object.

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
- Fit **with** an intercept column; residual at t subtracts both `alpha_i` and `beta_i @ f_t` so residuals are centered relative to the window fit. `RESIDUAL_INCLUDE_ALPHA = True` in `src/config.py` (ratified at kickoff, logged in DECISIONS.md).
- **Build the in-window factor matrix from the CURRENT window's eigenvectors, not by slicing the stored daily `factors_a` series** (this is Avellaneda & Lee's actual construction): `F = window_returns @ (eigvecs / sigma)`, a 252 × m matrix available for every day of the window. Regressing on stored daily factor returns would compound warm-ups — each stored factor return itself needs 252 prior days, so a full factor window would first exist ~504 days after data start, silently pushing the first residual into 2016 and losing a year of train triggers; it would also NaN-poison the design matrix wherever the stored m changed inside the window. With the in-window construction, residuals begin at day 253 of the data (early 2015, given the 2014 warm-up year). `factors_a` remains the artifact for `f_mkt_vol_20d` and the SPY check — it is simply not the beta-regression input. Read m per-day from `pca_meta`; never hardcode.
- Betas from the window ending t−1 applied to factor returns at t = the "train-then-apply" structure the spec calls the thing that keeps it honest. Assert the beta-fit window excludes t.
- Write per-day betas into `loadings_a.parquet` alongside the loadings (join on `date, ticker, component`).

**Unit tests:** (a) synthetic data generated as exact linear factor model + known noise → recovered betas within tolerance, residuals ≈ the injected noise; (b) shape/NaN mask of `residuals_a` matches returns; (c) residuals at t change if factor return at t changes but NOT if returns after t change (mini future-perturbation, previews P1.10); (d) per-window residual mean ≈ 0 in-sample but out-of-sample residual at t is computed from t's actual return.

**Hours:** 2.5h, Day 2 AM.
**Done when:** `residuals_a.parquet` written on real data, `loadings_a.beta` populated, tests pass — Day 2 midday hard deadline (P1.5 and P2's real features both consume this).

---

### P1.4 — PCA sanity-check suite (`scripts/checks/`, Day 2)

**Goal.** The four spec Step 2 checks, each a standalone script that writes one figure to `results/figures/` and prints a single `PASS`/`FAIL` line. These four figures go in the report.

**Consumes:** `factors_a.parquet`, `pca_meta.parquet`, `loadings_a.parquet`, `data/raw/spy.parquet` (P1's own artifact from P1.1 — convert SPY prices to log returns inside the script).
**Produces:** `results/figures/scree.png`, `pc1_loadings.png`, `pc1_vs_spy.png`, `var_explained_over_time.png` + printed pass/fail.

| Script | Figure | Pass criterion |
|---|---|---|
| `check_scree.py` | eigenvalue spectrum, median window + 3 sample dates | PC1 clearly dominant (λ₁/Σλ ≥ 0.25 on median window) |
| `check_pc1_loadings.py` | bar chart of PC1 loadings, median window | ≥ 38/40 loadings positive |
| `check_pc1_vs_spy.py` | scatter + rolling corr of daily PC1 factor return vs SPY log return | full-sample corr **> 0.9** (spec: "single best check") |
| `check_var_over_time.py` | top-3 cumulative variance across rolling windows, 2015–2025 | visible 2020 spike (crisis correlation); eyeball + printed max-vs-median ratio |

Each script: `python scripts/checks/check_X.py` → figure + one line, e.g. `PASS pc1_vs_spy corr=0.94`. A FAIL on any check blocks P1.5 — debug PCA before clustering garbage.

**Unit tests:** none beyond the scripts themselves (they *are* checks); smoke-test each on fixture data.

**Hours:** 1.5h, Day 2 (canonical anchor: sanity checks PASS Day 2 EOD).
**Done when:** four figures exist from real data, all four print PASS, links dropped in the team channel.

---

### P1.5 — SHARED clustering machinery + Track A application (`src/representation.py`, Day 2)

**Goal.** The shared rolling-recluster machinery — k-means wrapper with k selection, co-membership sets, stability frames, the 21-trading-day loop — plus its Track A application: cluster the 40 stocks on formation-window factor-beta vectors → `labels_a`/`stability_a`. **Shared-instrument rule:** this is the only clustering code in the repo. P2 calls `fit_kmeans_select_k`/`comembership`/`stability_frame` for Track B's quarterly snapshots (its own k-range per config); P3 calls them for Track C's partial-correlation distance vectors. Track and feature matrix are parameters, never hardcoded.

**Consumes:** `loadings_a.parquet` (beta vectors), trading calendar from `returns.parquet`.
**Produces:** `data/clusters/labels_a.parquet`, `data/clusters/stability_a.parquet` (Section 2 schemas); the importable machinery.

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

- **Cluster input** (config freeze, Zhang's compressed option): each stock's vector of factor betas from the formation window. Use the betas fitted at the **window-end date** (the P1.3 betas dated `window_end` were estimated on exactly the trailing 252 days — no separate re-fit needed). Feature dim = that date's `n_components`; standardize columns of the 40×m beta matrix before k-means (window-local, trivially — it's one cross-section).
- k selection by **max silhouette, k ∈ 8..13, on formation data only** — never on trading outcomes. Log `(window_end, best_k, silhouette)` to `data/clusters/kmeans_log_a.csv` for the report.
- `sklearn.cluster.KMeans(init="k-means++", n_init=10, random_state=311)`. **Empty clusters:** sklearn reseeds internally, so no custom handling — but assert `len(np.unique(labels)) == k` after fit and log a warning if a cluster came out empty-then-reseeded (with 40 points and k=13 it can happen); this is the note the spec asks for.
- **Label switching:** never compare raw `cluster_id` across windows. All cross-window logic goes through `comembership()` sets — invariant to relabelling by construction. `cluster_id` in `labels_a.parquet` is only meaningful within one `window_end`.
- Stability metric for the report: per window, `mean(co_clustered)`. `stability_a.parquet` is also the direct upstream of `f_cluster_stability` in P2's dataset builder — the contract is the per-pair boolean, not the aggregate.
- Recluster dates: every 21 trading days on the actual trading calendar (positional stride over `returns.index`, starting at index 252), NOT calendar days.
- **Clustering-input decision (spec Step 4 says "try both"):** the committed input is Option 2 — factor-beta vectors — on Zhang's decisive evidence (PC-based clustering passed the statistical-arbitrage test at p = 0.01 vs p = 0.785 for raw features). Recorded as an explicit kickoff decision in `DECISIONS.md`, so the deviation from "try both" is deliberate, not an omission. The v1 plan's Option-1 robustness run is **cut in v2** (no slack in the 4-day calendar); the DECISIONS.md entry plus the Zhang citation carry the justification in the report.

**Unit tests:** (a) planted 3-cluster synthetic betas → silhouette selects a k that recovers the planted partition (compare via co-membership sets, not labels); (b) `comembership` invariant under label permutation; (c) `stability_frame` correct on hand-built two-window example; (d) seed 311 → identical labels across two runs; (e) recluster dates are exactly every 21st trading day.

**Hours:** 3h, Day 2.
**Done when:** `labels_a` + `stability_a` written for every recluster date on real data; k-log saved; tests pass; P2 confirms the machinery imports cleanly for the quarterly-snapshot case.

---

### P1.6 — The SHARED pair builder (`src/representation.py`, Day 1 PM stub + Day 2)

**Goal.** One function that turns any track's cluster labels into `pairs_{track}.csv` under Zhang's rules. **Shared interface: P2 imports `build_pairs` for Track B (Day 2), P3 for Track C (Day 3 PM).** The canonical signature lives in Section 2.2; a stub raising `NotImplementedError` is committed at kickoff so P2/P3 can import and mock immediately. The compressed calendar merges v1's two-day split: the split algorithm and its hand-built-distance-matrix tests are written Day 1 PM (fixture-testable, no real-data dependency), final wiring + `v1-frozen` git tag Day 2.

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
                source: str,               # "track_a" | "track_b" | "track_c"  (track_d removed in v2)
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
5. Active windows tied to the recluster cadence: `active_from` = first trading day strictly after `window_end`; `active_to` = the next recluster's `window_end` (inclusive), or the last data date for the final window. Consecutive selections of the same `pair_id` produce consecutive rows whose windows tile without gap — P1.7 and P2's trigger logic both rely on this to detect run continuity. (Track B passes quarterly windows through the same fields.)

**Unit tests:** (a) cluster sizes 1/2/3/4 → 0/1/3/6 pairs; (b) size-5 with a hand-built distance matrix → known subgroups and exactly 1+3=4 pairs; (c) size-6 → three 2-subgroups, 3 pairs; (d) `pair_id` alphabetical regardless of input order; (e) active windows tile the calendar exactly; (f) `source` propagates. These tests are the acceptance gate for P2/P3's imports — tag `v1-frozen` once green: **Day 2 midday, the single deadline stated everywhere** (P2's Track B pipeline calls it Day 2 PM).

**Hours:** 1.5h — Day 1 PM: stub + split algorithm + tests (1h); Day 2 AM: wiring + tag by midday (0.5h).
**Done when:** tests green, `pairs_a.csv` written on real data Day 2 EOD (canonical anchor), P2 has successfully called `build_pairs(..., source="track_b")`.

---

### P1.7 — SHARED spread/z-score module + all tracks' z-scores (`src/representation.py`, Days 2–3)

**Goal.** Per-pair spread (simple version per config) and rolling 60d z-score. **P1 runs this module for every track's pair list as it lands:** `zscores_a` Day 2 EOD; `zscores_b` as soon as P2's `pairs_b.csv` lands (Day 2 EOD or first thing Day 3 — canonical anchor); `zscores_c` Day 3 PM, one function call when P3 hands over `pairs_c`.

**Consumes:** `residuals_a.parquet` (for **all** tracks — track d branch removed in v2; tracks b/c change only the grouping, the tradeable object is still the Track A residual spread), `pairs_{track}.csv`.
**Produces:** `data/spreads/spreads_{track}.parquet`, `data/spreads/zscores_{track}.parquet` (date × pair_id).

**Signatures:**

```python
def build_spread_for_pair(residuals: pd.DataFrame, pair_rows: pd.DataFrame,
                          warmup: int = 60) -> pd.Series:
    """All active rows for one pair_id -> spread series over its active runs."""

def zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    """(spread - rolling_mean) / rolling_std, window-local trailing stats.
    min_periods=window; NaN during warmup."""

def run_spreads(track: str, residuals_path: str = "data/processed/residuals_a.parquet") -> None:
```

**The re-anchoring policy (the flagged subtlety).** With a 21-day recluster cadence and a 60-day z-window, restarting the spread at every `active_from` would leave every pair NaN-z for its first ~3 windows — most pairs would never become tradeable, and Track B's quarterly (63-day) windows would barely clear warmup. So the frozen policy is:

> **`SPREAD_POLICY = "carry_with_burnin"`, `SPREAD_WARMUP_DAYS = 60` (added to `src/config.py` at kickoff).** Group each pair_id's rows in `pairs_{track}.csv` into maximal **runs** of consecutive active windows (windows tile, so a gap = the pair dropped out and re-entered). Within a run, the spread accumulates continuously across window boundaries — same pair, same economic relationship, no re-anchor. At the **start** of each run, accumulation begins `60` trading days *before* `active_from`, using already-written residuals from those dates. This burn-in is backward-looking history available at `active_from` — **no leakage** — and it means z-scores are valid from the first active day of every run. z-scores are still only *emitted* (non-NaN in `zscores_{track}`) for dates inside active windows; burn-in dates exist only internally.

Justification, one line for the report: continuity within a run reflects that the pair relationship persists across refreshes; the historical burn-in makes new pairs immediately tradeable without touching future data. (The rejected alternative — hard restart + overlap requirement — is dominated: it either forbids new pairs or silently shrinks the tradeable set.)

Other notes: z-score uses **trailing** rolling mean/std (`.rolling(60, min_periods=60)` — stats at t may include t's spread value per spec 6b's plain rolling z; P2's trigger comparison uses these z values as-is). Spread per config: `spread_t = Σ (resid_A − resid_B)` over the run from burn-in start; no hedge ratio.

**Unit tests:** (a) planted OU fixture from P2's `src/fixture_zscores.py` → z-scores match the golden file within tolerance (the *shared* golden fixture P2's trigger code also tests against — exactness here is what makes Day 3 integration boring); (b) a pair active in windows 1–3, gone in 4, back in 5 → two runs, second run's spread re-anchors with burn-in, no value bleeds across the gap; (c) z is NaN nowhere inside active windows after burn-in; (d) columns of outputs == unique pair_ids.

**Hours:** 3h — Day 2: module + golden test + `zscores_a` (2h); Day 2 EOD/Day 3 AM: `zscores_b` (0.5h — one function call; the half-hour is eyeballing distributions); Day 3 PM: `zscores_c` (0.5h).
**Done when:** `zscores_a.parquet` Day 2 EOD (P2's real trigger build depends on it Day 3 AM); `zscores_b.parquet` first thing Day 3 at the latest; `zscores_c.parquet` Day 3 PM within the hour of P3 delivering `pairs_c`.

---

### P1.8 — E2: Gaussian Discriminant Analysis + its full evaluation (`src/models/e2.py`, Days 3–4)

**Goal.** E2 behind the shared `EntryModel` API (Section 2.2 contract: `fit / tune / predict_proba / get_params_report / save / load`, StandardScaler fit on train rows only inside `fit` and serialized with the model), then everything the slice model owes the grid: fit + tune on train/val Day 3, τ Day 3 evening, E2 × {a,b,c} grid rows, calibration figures, freeze Day 4 AM, turnover control Day 4 PM.

**Decision, stated now:** primary E2 = **GDA with shared covariance** (mathematically LDA), implemented **directly in numpy** — class priors `φ`, class means `μ_0, μ_1`, pooled covariance `Σ`, posterior via Bayes' rule with the Gaussian class-conditionals. It is ~30 lines and is the course-aligned choice (this is exactly the GDA derivation from lecture); the sklearn cross-check makes it safe. `sklearn QuadraticDiscriminantAnalysis` (per-class covariances, small `reg_param` for stability on 7 features) runs as a one-line **robustness row** in the report, not a grid cell. GDA has no hyperparameters, so `tune` is a no-op returning val AUC; class imbalance is handled naturally through the priors (report both empirical-prior and balanced-prior posteriors if the base rate is skewed; empirical is primary). Report framing: discriminative-vs-generative — does the Gaussian assumption on these 7 features help or hurt vs P2's E1?

**Evaluation duties (all through the shared instruments — P1 writes none of this machinery):**

- **Fit/tune Day 3 PM** on `triggers_{a,b,c}` train/val rows via P3's runner; all models tune together Day 3 in v2.
- **τ Day 3 evening** by the one pre-registered rule (Section 5, P3's τ protocol — grid `{0.40…0.80}`, max validation net P&L at c=10bps subject to ≥25 accepted trades, ties toward higher τ; rule text unchanged from v1). Recorded in `taus.json` + DECISIONS.md before the validation gate closes.
- **Grid rows:** E2 × {a,b,c} on train+validation as part of the Day 3 full-grid run — 3 of the 12 cells.
- **Calibration figures:** `calibration_{track}_e2.png` via P3's reliability-diagram module (10 quantile bins), Day 4 PM final.
- **Freeze Day 4 AM:** `e2_{a,b,c}.joblib` (model + scaler in one object via `EntryModel.save`) through P3's freeze protocol; sha256 hashes into DECISIONS.md; loaded — never refit — at the noon witnessed test run.
- **Turnover control Day 4 PM:** P1 runs P3's turnover-matched-control machinery for E2's cells (each person runs their own model's controls in v2).

**Consumes:** `triggers_{track}.parquet` (P2), `EntryModel` API + runner + metrics/calibration + freeze + control machinery (P3). **Produces:** `src/models/e2.py`, frozen `e2_{track}.joblib` × 3, E2 grid rows, calibration figures, `control_{track}_e2.json`.

**Tests:** `tests/test_models.py` — our numpy `predict_proba` matches `sklearn LinearDiscriminantAnalysis.predict_proba` to `1e-6` on fixture data. Written Day 3 against P2's fixture triggers before any real fit.

**Hours:** 3.5h — Day 3: implementation + LDA verification + fits/tuning/τ (2.5h); Day 4: freeze (0.5h) + turnover control (0.5h). Calibration-figure polish inside the Day 4 figures block.
**Done when:** 1e-6 test green; E2 rows present in the Day 3 validation grid; τ in `taus.json` Day 3 evening; frozen artifacts hashed Day 4 AM; test-run rows produced from `load` only.

---

### P1.9 — Manual trace of the engine ledger (Day 3 PM, `docs/manual_trace.md`)

**Goal.** Leakage-checklist item 8: hand-verify the t/t+1 alignment on paper. **Deliberately assigned to P1, who did not write the backtest engine** — P3 built it, so a non-author independently re-deriving three ledger rows by hand is a genuine cross-check, not the author grading their own homework. This is a hard criterion of the Day 3 evening validation gate.

**Procedure.** Pick 3 trigger dates from the real Track A × E0 ledger — one calm-regime (2017), one crisis (March–April 2020), one 2022 — one trade each. For each, on paper: the z-score path around the trigger, confirmation the crossing is an onset (prev day |z| < 2.0), entry at close of t+1, the two legs' raw prices and daily simple returns for each holding day, hand-summed gross P&L, cost deduction at c=10, exit day and reason. Then compare every number against the ledger row; any mismatch is a bug, full stop.

**Output:** `docs/manual_trace.md` from this template (committed Day 2 so the structure is agreed before use):

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

The template's convention line reads: *decision from day-t signal, entry at the t+1 close, first P&L day t+2; exit signal and exit fill share the same close (disclosed convention, per Section 5's engine spec).*

**Consumes:** `results/trades_a_e0.parquet` (P3's engine output), `zscores_a`, `prices.parquet`. **Produces:** `docs/manual_trace.md` (cited in the report's leakage audit).
**Hours:** 2h, Day 3 PM (as soon as the real Track A × E0 ledger exists).
**Done when:** 3/3 trades verdict MATCH, or a bug filed against P3's engine, fixed, and the trace redone. May not slip past the Day 3 evening gate (a red trace is a gate-red item burned down Day 4 AM before the freeze).

---

### P1.10 — Leakage-audit items 1–4 (`scripts/audit/audit_p1_items_1_4.py`, Days 3–4)

**Goal.** Mechanically verify the four Part 4 checklist items P1 owns, using the future-perturbation helper defined in Section 2 (`tests/leakage_utils.py`: `perturb_after(df, date, seed)` replaces all rows after `date` with seeded noise; `assert_no_future_dependence(...)` wraps the comparison). **The helper itself lands Day 2 AM** (1h, in the budget row, not here): P1.3's residual tests are its first consumer that morning, and P2's Day 2 feature look-ahead tests consume it the same afternoon — it cannot wait for this Day 3 task, which adds only the audit script and the perturbation passes.

**Consumes:** all P1-pipeline modules + real data. **Produces:** printed PASS/FAIL per item + `results/audit/audit_items_1_4.md` (four short written notes, one paragraph each, pasted into the report's checklist section).

| Item | Test |
|---|---|
| 1. PCA on trailing windows only | Run `run_rolling_pca` on real returns and on `perturb_after(returns, T)`; assert `factors_a`, `loadings_a` **bit-identical** for all dates ≤ T. Repeat for two choices of T (mid-2018, mid-2022). |
| 2. Standardization window-local | Same perturbation test catches full-sample stats automatically (full-sample mu/sigma would change pre-T output); additionally grep-audit `src/representation.py` for any `.mean()`/`.std()` call not on a window slice, and cite the code lines in the note. |
| 3. Betas estimated before application | Perturbation test on `residuals_a` (identical ≤ T); plus the explicit in-loop assertion from P1.3 that the beta window excludes t. |
| 4. Clustering on formation data only | Perturbation test on `labels_a`/`stability_a` for recluster dates ≤ T; plus a written note that k selection reads silhouette, never trading outcomes, with the code path cited. |

Runtime is a few full-pipeline re-runs — minutes, not hours. Failures are release-blocking for the Day 4 witnessed test run.

**Hours:** 2h — Day 3: audit script + perturbation passes (1h; helper pre-built Day 2 AM); Day 4: final pass in the team leakage review + the four written notes (1h).
**Done when:** four PASS lines from a run on the Day 4 code state, notes committed before the results freeze.

---

### P1 — day-by-day hour budget

| Day | Date | Tasks | Hours |
|---|---|---|---|
| 1 | Sun Aug 2 | Kickoff + P1.6 stub commit (1h) · P1.1 universe/prices/returns → `returns.parquet` early PM (3h) · P1.2 rolling PCA, tests + running on real returns EOD (3.5h) · P1.6 split algorithm + hand-built-D tests (1h) · Day 1 evening sync (Bloomberg decision — attend) | 8.5 |
| 2 | Mon Aug 3 | **AM, in this order:** P1.3 residuals (2h, tests use the helper) · shared perturbation helper `tests/leakage_utils.py` (1h) · P1.6 wiring + `v1-frozen` tag **by midday** (0.5h) · **PM:** P1.5 shared clustering + `labels_a`/`stability_a` (3h) · P1.4 four sanity checks PASS (1h; figure polish deferred to Day 4) · P1.7 spread module + golden test + `zscores_a` EOD (2h) · `zscores_b` if `pairs_b` lands before EOD, else first thing Day 3 | 9.5 |
| 3 | Tue Aug 4 | `zscores_b` if slipped (0.5h) · **midday integration checkpoint** (0.5h) · P1.8 GDA + 1e-6 test + fit/tune E2×{a,b,c} + τ (2.5h) · P1.7 `zscores_c` on P3's `pairs_c` (0.5h, early PM) · P1.9 manual trace (2h) · P1.10 audit script + perturbation passes (1h) · **evening validation-gate sync** (0.5h) | 7.5 |
| 4 | Wed Aug 5 | AM: red-item burn-down / buffer (1.5h) · P1.8 freeze E2 + hashes (0.5h) · **~noon witnessed test run, all three present** (1h) · PM: E2 turnover controls (0.5h) · F1–F4 (+`pc1_loadings`, folded into F1's script) finalization + E2 calibration commentary (1.5h) · **T4 assembly**: format the leakage checklist from the three slices' audit notes, each item signed by its verifying owner, incl. the line "Item 12: N/A — Track D not run; cited per spec 8.4" (1h) · P1.10 audit notes (1h) · **EOD results-freeze sync + git tag** (0.5h) | 7.5 |

**Total: 33h** (~8.25/day; Day 2 is the slice's peak and is deliberately all-substrate — the AM ordering exists so the two artifacts other slices consume that day, the pair-builder tag and the clustering machinery, are green by midday/early PM). Standing scribe duty (DECISIONS.md entries, config fill-ins) is absorbed in sync slots.

---

### Slice P1 — definition of done

- [ ] `prices/volume/spy/returns` parquets committed; cleaning report 40/40 PASS, pasted in DECISIONS.md (Day 1)
- [ ] `factors_a`, `pca_meta`, `loadings_a` (with betas), `residuals_a` written from real data; P1.2/P1.3 tests green (Day 2)
- [ ] Four sanity checks PASS on real data; four figures in `results/figures/` (Day 2 EOD)
- [ ] Shared clustering machinery imported and run successfully by all three slices; `labels_a`, `stability_a`, `pairs_a.csv` written; pair-builder `v1-frozen` and called with `source="track_b"` (P2) and `source="track_c"` (P3)
- [ ] `zscores_a` Day 2 EOD; `zscores_b` by first thing Day 3; `zscores_c` Day 3 PM; golden-fixture z-score test green (shared with P2's trigger tests)
- [ ] E2 numpy GDA matches sklearn LDA to 1e-6; E2 × {a,b,c} rows in the validation grid Day 3; τ in `taus.json` Day 3 evening; frozen `e2_*.joblib` hashed in DECISIONS.md Day 4 AM; test rows from `load` only
- [ ] `docs/manual_trace.md`: 3/3 MATCH — the engine hand-verified by a non-author
- [ ] Leakage items 1–4: perturbation tests PASS at two cut dates; four written audit notes committed before the Day 4 freeze
- [ ] P1 stack passes P3's noise test (no structure found in random-walk input); E2 clears both binding criteria
- [ ] Every stochastic call seeded 311; two consecutive full P1-pipeline runs byte-identical
- [ ] All config values imported from `src/config.py`; the P1-added constants (`RESIDUAL_INCLUDE_ALPHA`, `SPREAD_POLICY`/`SPREAD_WARMUP_DAYS`) ratified at kickoff and logged in DECISIONS.md; scribe log current at every sync

---

## 4. Slice P2 — Training data: dataset builder, Track B, E1

P2 owns the training data every model learns from — the trigger/label definition, the 7 features, the splits with purge and embargo, and the assembly CLI that turns any track's pair list into a `triggers_{track}.parquet` with one command — plus the only externally-dependent pairing track (Track B, Bloomberg characteristics) and the **primary model of the pre-registered comparison** (E1 logistic regression). **P2 must be the person with Bloomberg terminal access**; the Monday terminal-access question is a Day 1 kickoff action item and the go/no-go fires at the Day 1 evening sync (see P2.6). Everything in the labeling → features → dataset stack develops against P2's own planted-OU fixture on Day 1 and is golden-tested before any real z-score exists; real data flows through it for the first time Day 3 AM. The scientific instrument here is **shared and built once**: the same trigger detector, feature builder, and split logic score every track — Track C's dataset on Day 3 PM is a re-run of P2's CLI, not new code.

Files P2 owns:

```
src/fixture_zscores.py   # P2.1 (planted-OU fixture + golden triggers, per Section 2.4)
src/dataset.py           # P2.2 triggers + labels · P2.3 splits/purge/embargo · P2.4 the 7 features · P2.5 assembly CLI
src/characteristics.py   # P2.6 (loader) + P2.7 (pipeline)
src/models/e1.py         # P2.8 (behind P3's EntryModel API, Section 5)
src/analysis.py          # P2.9 (base rate) + P2.11 (consensus-lite)
tests/test_dataset.py  tests/test_characteristics.py
```

Consumed from other slices (Section 2 contracts by name): P1's `zscores_{track}` / `spreads_{track}` / `residuals_a` / `factors_a` / `stability_{track}` / `pairs_{track}` and the shared pair-builder + k-means machinery (Section 3); P3's `EntryModel` API, runner, metrics/bootstrap, τ rule, freeze protocol, and turnover-control machinery (Section 5).

---

### P2.1 — Planted-OU fixture + golden triggers (Day 1 AM–PM, ~2.5h)

**Consumes:** nothing (seeded RNG; schemas per Section 2.2/2.4).
**Produces:** `data/synth/fixture/*` and `tests/golden/golden_triggers.csv`.
**Done when:** fixture artifacts pass the schema contract tests and the golden file is committed with a header note naming who hand-verified it — Day 1 midday, because P2.2 and P3's engine golden test both consume it the same afternoon.

Per the Section 2.4 fixture spec, the outputs under `data/synth/fixture/` are: `zscores_a.parquet` and `spreads_a.parquet` (date × 6 pair_ids), **`prices.parquet`**, `residuals_a.parquet` and `volume.parquet` for the 12 fake leg tickers — leg prices are seeded walks whose returns embed the planted residual differences, which is what lets **P3's engine compute P&L on the fixture** (P3's 3-trade golden ledger is hand-computed from these prices) — `factors_a.parquet` (a fake `pc_1` series for `f_mkt_vol_20d`), `pairs_a.csv` **including one pair whose rows tile across two consecutive active windows** (exercising the run/burn-in convention and the cross-boundary trigger case), and `stability_a.parquet` for the 6 fake pairs. The calendar is pinned to 2019-07-01 + 500 trading days so triggers **span the train/val boundary**; a contract test asserts the fixture triggers table contains both train and val rows — otherwise `tune()` and the τ dry-runs would silently exercise nothing.

The hand-computed `tests/golden/golden_triggers.csv` (`trigger_id, pair_id, trigger_date, z_trigger, label, horizon_end_date`) is derived by exporting the z-matrix and scanning it in a spreadsheet, **never by running the labeling code it will test** (Section 2.4).

---

### P2.2 — Trigger detection and labeling (Day 1 PM, ~4h; golden test green Day 1 EOD)

**Consumes:** `data/spreads/zscores_{track}.parquet` (fixture version Day 1; real `zscores_a` Day 2 EOD), `data/pairs/pairs_{track}.csv`, config constants.
**Produces:** the trigger/label core of `triggers_{track}` (columns `trigger_id, pair_id, source, trigger_date, z_trigger, label, horizon_end_date`) — features and split added by P2.4/P2.3.
**Done when:** `pytest tests/test_dataset.py` passes the golden-file test exactly (every expected trigger found, no extras, labels match) — this is a **Day 1 EOD anchor**. The v1 schedule gave this task a Day 2 tail; the compression is safe because the fixture and golden file exist by early afternoon (P2.1) and the semantics below are fully pre-specified. If the golden test is red at Day 1 EOD, the first hour of Day 2 is the pre-agreed buffer, taken from P2.7's component-naming polish.

```python
# src/dataset.py
def detect_triggers(
    zscores: pd.DataFrame,            # date x pair_id
    pairs: pd.DataFrame,              # pairs_{track}.csv incl. active_from/active_to
    z_entry: float = config.TRIGGER_Z,        # 2.0
    horizon: int = config.LABEL_HORIZON,      # 5 trading days
    reversion_frac: float = config.REVERSION_FRACTION,  # 0.5
) -> pd.DataFrame: ...
```

**Onset semantics (exact).** Consecutive tiling rows for the same `pair_id` are first merged into maximal **runs** — the identical construction P1's spread module uses (windows tile, so a gap means the pair genuinely dropped out and re-entered). Active-window boundaries *within* a run are invisible to the trigger logic. A trigger then fires for pair *p* at date *t* iff:
1. both *t−1* and *t* lie inside the same **run** with non-NaN z on both days — so only a run's true first day can never trigger, not every 21-day refresh boundary (applying the test per pairs-row instead of per run would silently discard roughly a quarter of legitimate triggers);
2. `|z_{t-1}| < 2.0` and `|z_t| >= 2.0` (onset only, per spec 7a);
3. the pair is **re-armed** (below).

**Overlap policy (decide now, write in the docstring):** after a trigger at *t*, the pair is dis-armed. It re-arms only when **both** (a) the prior trigger's 5-day horizon has closed (`date > horizon_end_date`) **and** (b) `|z|` has printed a value `< 2.0` on some day at-or-after horizon close. A new trigger then requires a fresh below-to-above crossing. *Justification:* spec 7a exists precisely to prevent hundreds of near-duplicate overlapping examples; without condition (b), a spread that stays pinned above 2.0 through the horizon would immediately re-trigger on a trivial 1.99→2.01 wiggle, recreating the duplicate problem the onset rule was built to kill. One trigger = one independent widening episode.

**Label:** `label = 1` iff `min(|z_s|) <= reversion_frac * |z_trigger|` for *s* in the 5 trading days after *t* (i.e. *t+1…t+5*, pair-active days); `horizon_end_date = t+5` in trading days. `z_trigger` is stored **signed**. Triggers whose horizon would extend past the pair's **run end** (the `active_to` of the run's final row) or past the last data date are dropped with a logged count (they have undefined labels; report the count in the appendix, expected to be small). The drop applies only at true run ends — never at internal window boundaries.

`trigger_id = f"{pair_id}__{trigger_date:%Y%m%d}"` (unique because at most one trigger per pair per day).

**Test:** golden-file comparison against `tests/golden/golden_triggers.csv` (hand-derived per Section 2.4 — never produced by this code), plus unit cases for: crossing on a run's first day (no trigger), a trigger straddling an internal window boundary (fires — the fixture's tiling pair covers this), NaN gap at *t−1* (no trigger), pinned-above-2 re-arm suppression, exact-equality boundary (`|z_t| == 2.0` triggers; `|z| == 0.5*|z_trigger|` labels 1).

---

### P2.3 — Splits, purge, embargo (Day 1 PM, ~1.5h — moved off Day 2, the slice's pinch day; it is fixture-only and has no Day 2 dependency)

**Consumes:** `trigger_date`, `horizon_end_date` columns + the trading calendar (index of `returns.parquet`; fixture calendar until Day 3). Boundaries from `src/config.py` (train ≤ 2020-12-31, val 2021–2022, test 2023–2024).
**Produces:** the `split` column with values `train | val | test | purged | embargo`.
**Done when:** deterministic unit tests on synthetic dates prove both mechanisms (see below).

```python
# src/dataset.py
def assign_split(
    trigger_dates: pd.Series,
    horizon_end_dates: pd.Series,
    calendar: pd.DatetimeIndex,
) -> pd.Series: ...  # categorical: train/val/test/purged/embargo
```

Rules, applied in order:
1. Base assignment by `trigger_date` vs the chronological boundaries.
2. **Purge:** any observation whose `horizon_end_date` falls on or after the first trading day of the *next* split → `purged`. With a 5-day horizon this is exactly "the last 5 trading days of labels before each boundary" from the config freeze — but implement it via `horizon_end_date` comparison, not day-counting, so it stays correct if the horizon ever changes on validation (P2.9 escalation path).
3. **Embargo:** any observation whose `trigger_date` falls within the first 10 trading days of val or test → `embargo`.

Rows marked `purged`/`embargo` are **kept in the parquet, never silently dropped** — P3's runner and all model code filter on `split`, and the report can state exact counts per category (leakage-audit items 5–6, P2.12).

**Tests:** synthetic 30-day calendar with a boundary in the middle; assert (i) a trigger 3 days before the boundary with a 5-day horizon → `purged`; (ii) a trigger 6 days before → `train`; (iii) triggers on embargo days 1–10 of the later split → `embargo`, day 11 → `val`; (iv) idempotence and no row loss.

---

### P2.4 — The seven features (Day 2, ~3h)

**Consumes:** trigger rows (P2.2); `zscores_{track}`, `spreads_{track}`, `residuals_a`, `factors_a`, `data/raw/volume.parquet`, `data/clusters/stability_{track}.parquet`, `pairs_{track}.csv`. Fixture versions of all of these exist Day 1.
**Produces:** the `f_*` columns of `triggers_{track}`, **raw (unstandardized)** — standardization happens inside model `fit` (P3's `EntryModel` contract, Section 5).
**Done when:** `tests/test_dataset.py` passes hand-computed values on the fixture for every feature, and an assertion sweep confirms no feature uses any value dated after `trigger_date` — Day 2 EOD anchor.

```python
# src/dataset.py
def build_features(
    triggers: pd.DataFrame, *,
    zscores: pd.DataFrame, spreads: pd.DataFrame,
    residuals: pd.DataFrame,          # always residuals_a (track d branch removed in v2)
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
| `f_mkt_vol_20d` | std of `pc_1` column of `factors_a` over trailing 20 trading days ending at *t*. Used for **all** tracks — Track B/C have no own return-factor model, and "market volatility" is a property of the market, not the track. |
| `f_rel_volume_20d` | `0.5 * [ vol_A_t / SMA20(vol_A)_t + vol_B_t / SMA20(vol_B)_t ]`, SMA over trailing 20 days including *t* |
| `f_days_since_trigger` | trading days since this pair's previous trigger (per the P2.2 trigger stream), capped at **126**; pairs with no prior trigger get 126 |
| `f_cluster_stability` | the pair's `co_clustered` bool (as 0/1) from `stability_{track}` at the most recent `window_end <= trigger_date`; if the pair has no prior-window record (newly formed), **0** — "no evidence of persistence" is the conservative default, stated in the report |

**Standardization contract:** `StandardScaler` is **fit on train-split rows only** inside `EntryModel.fit`, applied unchanged to val/test at `predict_proba` time, and serialized with the model (P3's freeze protocol). The parquet always holds raw features so P3's runner and the coefficient analysis see interpretable values.

**Methods-section note:** `f_spread_vol_60d` uses the std of daily spread *changes*, a deliberate sharpening of spec Step 8's literal "rolling std of spread" — the spread is a cumulative sum, so its level-std mostly measures drift, not jumpiness. Disclosed in one sentence in the report.

---

### P2.5 — Dataset assembly CLI, all tracks (Day 2 fixture ~0.5h; Day 3 AM real a+b ~1h; Day 3 PM track c ~0.5h)

**Consumes:** P2.2 triggers + P2.4 features + P2.3 splits, `pairs_{track}.csv` (for `source`).
**Produces:** `data/datasets/triggers_{track}.parquet`, exactly the Section-2 contract schema (all 7 `f_` columns, `label`, `horizon_end_date`, `split`).
**Done when:** schema-validation test passes (column names/dtypes exact, `trigger_id` unique, no NaN labels, split categories complete); fixture version Day 2 EOD; **real `triggers_a` + `triggers_b` are Day 3 AM deliverables**, built with row counts reviewed at the midday integration checkpoint; `triggers_c` is a Day 3 PM one-command re-run the moment P1 ships `zscores_c`.

```python
# src/dataset.py
def assemble_dataset(track: Literal["a","b","c"]) -> pd.DataFrame: ...
# CLI: python -m src.dataset --tracks a  (comma list accepted: --tracks a,b,c — the Makefile uses it)
```

One function, one track argument — the track/source of the input is a parameter, never hardcoded. This is the crux of the shared-instrument agreement: every grid row's training data comes off the same assembly line.

---

### P2.6 — Bloomberg Track B pull, with terminal-access contingency (Day 2 = Monday AM, ~3h; fallback branch ~1.5h)

**Goal.** One sitting at the Bloomberg terminal: quarterly observations of the 19 spec-4B.1 fields, 40 tickers, 2015Q1–2024Q4 (~30k cells).

**The contingency is decided at the Day 1 (Sun) evening sync, not later.** The kickoff assigns P2 the question "do I have terminal access tomorrow (Monday) morning?"; the answer is due at Sunday's evening sync. **Access confirmed** → P2 pulls Day 2 (Mon) AM before anything else. **Not confirmed** → Track B builds on the fallback columns **immediately on Day 2** — there is no waiting for Tuesday; a Tuesday pull would land after `pairs_b`, `zscores_b`, and `triggers_b` are already needed, so it would buy nothing but slippage. Either branch feeds P2.7 unchanged (the pipeline just sees fewer columns) and gets a dated `DECISIONS.md` entry.

**Day 1 stand-in → Monday swap (optional, recommended if Day 1 has slack, ~0.5–1h):** nothing stops Track B *code* from being finished before the terminal session. P2 generates `data/raw/characteristics_standin/` in the **exact long schema the loader expects** (`date, ticker, field, value`): the sector one-hots and price-derived fallback fields computed from our own price data (real values — and reused verbatim if the fallback ever fires, so this doubles as pre-building the fallback branch), plus obviously-fake placeholders for the fundamental fields (e.g. P/E = 15 + noise, tagged in a provenance column). The whole P2.7 pipeline then builds and golden-tests against it, and Monday becomes mechanical: drop the Bloomberg CSVs into `data/raw/characteristics/`, rerun `make trackb && make dataset`, and `pairs_b` → `zscores_b` → `triggers_b` regenerate — a data swap, not a code change. Three hard rules: **(1) stand-in fundamentals never touch a results run** — free-source fundamentals are current-only snapshots, and back-filling them across 2015–2024 is exactly the look-ahead trap spec 4B warns about; the stand-in exists to shape and test code, nothing else; **(2) the swap deadline is Monday (Day 2) EOD** — past it, the price-derived fallback becomes the *final* Track B (disclosed in limitations), never stale stand-in fundamentals; **(3)** every downstream Track B artifact regenerates after the swap (the one make command above — minutes), and the swap gets a dated `DECISIONS.md` line.

**Session plan (write it on paper before sitting down; compressed from v1 §4 B3, whose candidate-mnemonic table for all 19 fields is carried over verbatim into `data/raw/characteristics/FIELDS_USED.md` as the pre-session checklist):**
1. First 20 minutes — verify every candidate mnemonic with `FLDS` on one ticker (AAPL); record finals in `FIELDS_USED.md`.
2. Pull via Excel BDH (quarterly, `Per=CQ`, `Days=A`, `Fill=P`), one sheet per field, dates × tickers; export each sheet to CSV immediately — never leave the terminal with data only in a live spreadsheet.
3. Point-in-time caveat (spec 4B.1): prefer as-reported variants where `FLDS` shows them; where only restated values exist, pull and add to the limitations disclosure list. 15 minutes max hunting, then disclose. (This is leakage-audit item 10 — P2.12.)
4. Before leaving: spot-check 5 random cells on screen vs the CSVs; confirm ≥90% cell coverage per field (fields below that are P2.7 column-drop candidates).

**Mid-session abort trigger:** access being *confirmed* is not the same as the session *succeeding*. If committed CSVs are not on disk by **Day 2 ~noon** (terminal occupied, entitlement errors, export failures), the fallback branch starts immediately — it is only 1.5h — and any partial pull is discarded; same one-way `DECISIONS.md` logging. P2 does not spend the afternoon fighting the terminal on the slice's pinch day.

**Post-session:** `src/characteristics.py::load_raw()` melts the per-field CSVs into the long schema and validates it:

```python
def load_raw(raw_dir: str = "data/raw/characteristics/") -> pd.DataFrame:
    """Returns long DataFrame[date, ticker, field, value]; asserts 19 fields,
    40 tickers, 40 quarterly dates; prints per-field coverage table."""
```

**Fallback column list (unchanged from v1):** a reduced characteristic set from our own data — **sector** (one-hot, from `get_universe()` in `src/data.py`), **market cap** (shares outstanding from free filings snapshots × our price — approximate), and **trailing price-derived fields computed from our own `prices.parquet`/`volume.parquet`**: 60d volatility, RSI-14, close price, 20d dollar volume, 12m momentum. Historical *fundamentals* (P/E, ROE, growth) from free sources are unreliable point-in-time and are **not** substituted. ~8 usable columns instead of 18; the factorial design still runs with a weaker Track B, and the degradation gets a prominent limitations paragraph plus a `DECISIONS.md` entry.

- **Consumes:** terminal access (or `prices.parquet`/`volume.parquet` on fallback); P1's `get_universe()` (`src/data.py`). **Produces:** `data/raw/characteristics/*.csv` + `FIELDS_USED.md` — or the fallback frame, logged.
- **Done when:** `load_raw()` passes validation with the coverage table printed, or the fallback rule is invoked and logged — either way by Day 2 midday, so P2.7 has its input.

---

### P2.7 — Track B pipeline: characteristics → clusters → pairs_b (Day 2 PM, ~2.5h; `pairs_b` Day 2 EOD)

**Goal.** Zhang's pipeline (spec 4B.2–4B.6) from raw characteristics to `pairs_b.csv`, per quarterly snapshot. Leaner than v1's 6h version because clustering, pair construction, and stability are now **calls into P1's shared machinery** (Section 3) — P2 writes only the characteristic-specific stages:

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
```

Clustering per snapshot runs **P1's shared k-means/co-membership machinery** (k-means++, `n_init=10`, k ∈ 10..13 by max silhouette on this snapshot only, seed 311) on the score matrices; pairs come from **P1's shared pair-builder** (canonical frozen signature in Section 2.2, stub committed at kickoff) — P2 passes the quarterly labels plus per-quarter score matrices as `features_by_window` (they supply the within-cluster distances for the 5+ split), with `source="track_b"` and **quarterly active windows** (`active_from` = first trading day after quarter end, `active_to` = last day of the following quarter). Stability: `stability_b.parquet` by the same co-membership convention as Track A — pair co-clustered in consecutive quarterly snapshots.

- **Consumes:** P2.6 output (either branch), P1's shared machinery. **Produces:** `data/clusters/labels_b.parquet`, `data/clusters/stability_b.parquet`, `data/pairs/pairs_b.csv`, `results/tables/track_b_components.md`.
- **Tests:** `tests/test_characteristics.py` — sentiment formula on hand values (15 buys/5 sells → +0.5); z-scored columns have mean≈0, std≈1; a planted low-coverage column gets dropped; PCA on a rank-2 synthetic characteristic matrix recovers 2 dominant components.
- **Done when:** `pairs_b.csv` validates against the Section 2 schema **Day 2 EOD** and P1 confirms it ingests into the shared spread module (P1 produces `zscores_b` by EOD or first thing Day 3). `name_components` table polish is the designated slip-to-Day-3 item if Day 2 runs long.

---

### P2.8 — E1 logistic regression: fit, tune, τ, grid rows, freeze (Day 3 PM ~2.5h + Day 4 ~1.5h)

**Consumes:** `triggers_{track}.parquet`; P3's `EntryModel` API (`src/models/common.py`, Section 5 — fit fits the scaler on train rows, `tune` selects by validation AUC, `save`/`load` round-trip; P2 implements the subclass, never the interface).
**Produces:** `src/models/e1.py`; `results/tables/e1_coefficients_{track}.csv` per track; E1's rows of the 3×4 grid; frozen `e1_{track}.joblib` (model + scaler bundled) Day 4 AM.
**Done when:** E1 fits on fixture data with sane output Day 2 evening (fixture AUC well above 0.5 — the fixture has planted OU signal); first real fit at the Day 3 midday checkpoint ("E1 fits end-to-end" is a checkpoint criterion); E1 × {a,b,c} tuned rows in the train+validation grid Day 3 PM; τ per track recorded Day 3 evening; frozen with hashes Day 4 AM.

**E1 spec (unchanged from v1):** `sklearn.linear_model.LogisticRegression(penalty="l2", class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=config.SEED)`; `C ∈ {0.01, 0.1, 1, 10}` chosen by validation AUC in `tune`. **Coefficient deliverable:** table of standardized coefficients (features are scaled, so magnitudes are comparable) with feature names, per track — a headline report table ("market volatility carries a negative weight…" per Step 9).

**τ selection (Day 3 evening):** P2 applies **the one pre-registered rule** — owned and specified in Section 5 (P3), unchanged from v1 — to E1 on each track, using validation net P&L at c=10bps through P3's engine; values land in `results/frozen/taus.json` plus a dated `DECISIONS.md` entry, before any test-set run exists. All four models select τ the same evening; the old E3-finalizes-later split is gone.

**Day 4:** AM — freeze E1 per track through P3's freeze protocol (`EntryModel.save`, sha256 into `DECISIONS.md`, load-only smoke test); attend the ~noon witnessed test run. PM — run **P3's turnover-matched-control machinery for E1** (P2 runs its own model's control, ~1h; the machinery and the per-quarter matching rationale live in Section 5).

---

### P2.9 — Base-rate analysis + decision table + by-year decay figure (Day 3 AM ~1.5h; figure finalized Day 4 PM ~1h)

**Consumes:** real `triggers_a` and `triggers_b` (Day 3 AM), `triggers_c` when it lands.
**Produces:** the base-rate table presented at the **Day 3 midday integration checkpoint** — overall rate, by year, by track, by regime (calm vs stressed, split on median `f_mkt_vol_20d`) — and `results/figures/base_rate_by_year.png` (the 12e decay figure: reversion rate per calendar year 2016–2024 with trigger counts as bar annotations), finalized Day 4 PM.
**Done when:** the table is reviewed by all three at the midday checkpoint and the decision row below is executed. "Train base rate inside 15–85%" is a named checkpoint criterion.

Decision table (spec 7c), pre-agreed:

| Observed train base rate | Action |
|---|---|
| 15% – 85% | Proceed; `class_weight='balanced'` (already in config) handles moderate skew |
| < 15% or > 85% | **Escalate at the Day 3 midday checkpoint.** Label parameters (reversion fraction, horizon) may be revisited **on train+validation only**, never test, with a dated DECISIONS.md entry recording old→new values and the observed rate that forced it |
| > ~90% | Also flag as a possible bug (label logic or z-score construction) before touching parameters |

---

### P2.10 — E1 coefficient cross-track comparison (Day 4 PM, ~1h)

**Consumes:** fitted E1 models for tracks a, b, c.
**Produces:** `results/tables/e1_coefficient_comparison.csv` — one table, three tracks as columns, standardized E1 coefficients as rows — plus one written paragraph: does the model learn the same physics under all three selection methods (e.g. is `f_mkt_vol_20d` negative everywhere)? Sign agreement matters more than magnitude.
**Done when:** table committed and the paragraph's three-sentence skeleton is in `results/notes/` for the write-up week. *(v1's C11 error-analysis deep-dive — the top-20 false-positive notebook — is cut in v2; the coefficient comparison and calibration commentary carry the 12e analysis load.)*

---

### P2.11 — Consensus-lite (Day 4 PM, ~2h; spec 12c/8.3)

**Consumes:** `pairs_a.csv`, `pairs_b.csv`, `pairs_c.csv`, `triggers_{a,b,c}`.
**Produces:** `results/tables/consensus.csv`.
**Done when:** the table is populated and cited at the Day 4 EOD freeze sync. This is v1's B10 compressed to ~2h: **pair-quarter buckets + per-bucket reversion rate + per-bucket mean net return at c=10** (the net-return column is a cheap groupby joining trigger buckets to the existing trades ledgers — spec 12c names it, so it is committed, not polish); the grouped-bar figure and the per-bucket AUC slice are polish, done only if time remains — and if the AUC slice is skipped, that compression is disclosed in one methods sentence.

Granularity is **pair-quarters** (a pair-quarter is "selected" by a track iff the pair is active in that track during that quarter — pair lists change over time, so plain pair-level intersection would be wrong): expand each pairs file to (pair_id, quarter) rows from `active_from`/`active_to`, then bucket **by selection count** — chosen by 1, 2, or all 3 tracks — which is exactly spec 8.3's extension and now applies naturally with Track C committed. Per bucket: n pair-quarters, n triggers, reversion rate (label base rate). Report the overlap sizes even if consensus is tiny — "little overlap" is itself a spec-anticipated finding.

---

### P2.12 — Leakage-audit items 5–6 and 10 (~1h, written Day 4)

P2 owns three Part-4 checklist items, each closed with a short written note (one paragraph each) pasted into the team's leakage-audit section:
- **Item 5 — labels purged at split boundaries:** evidence = P2.3's purge unit tests + the retained-and-countable `purged` rows (exact counts quoted).
- **Item 6 — embargo applied:** evidence = P2.3's embargo tests + counted `embargo` rows.
- **Item 10 — Bloomberg point-in-time or disclosed:** evidence = `FIELDS_USED.md` per-field notes (as-reported vs restated) — or, on the fallback branch, the disclosure that price-derived fields are trivially point-in-time and fundamentals were not substituted.

**Done when:** three PASS notes with evidence pointers are committed before the Day 4 EOD results freeze.

---

### P2 — day-by-day hour budget

| Day | Date | Tasks | Hours |
|---|---|---|---|
| 1 | Sun Aug 2 | Kickoff (1h) · P2.1 fixture + golden triggers (2.5h) · P2.2 trigger/label code, golden test green EOD (4h) · P2.3 splits + tests (1.5h) · evening sync incl. Bloomberg go/no-go (0.25h) | ~9.25 |
| 2 | Mon Aug 3 | P2.6 Bloomberg session + `load_raw` — or fallback build (3h / 1.5h) · P2.4 features + tests (3h) · P2.7 Track B pipeline → `pairs_b` EOD (2.5h) · P2.5 fixture CLI + schema smoke + E1 fixture smoke-fit through the `EntryModel` base (0.5h) | ~9 (~7.5 on fallback) |
| 3 | Tue Aug 4 | P2.5 real `triggers_a`+`triggers_b` (1h) · P2.9 base-rate table (1.5h) · **midday checkpoint** (0.5h) · P2.8 E1 fit/tune × 3 tracks + τ (2.5h) · P2.5 `triggers_c` re-run (0.5h) · team smell test (0.5h) · **evening validation gate** (0.5h) | ~7 |
| 4 | Wed Aug 5 | Red burn-down buffer (1h) · P2.8 E1 freeze + hashes (0.5h) · **witnessed test run, present** (1h) · E1 turnover control via P3's machinery (1h) · P2.10 coefficient comparison (1h) · P2.11 consensus-lite (2h) · P2.9 decay figure (1h) · P2.12 audit notes (0.75h) · **EOD results-freeze sync** (0.5h) | ~8.75 |

Total ≈ 33h. **Day 2 is P2's peak and the whole slice's pinch point** — four mitigations are pre-agreed: P2.3 moved to Day 1 so Day 2 stays ≤ ~9h; the fallback branch is 1.5h cheaper than the terminal session (with the Day-2-noon mid-session abort trigger in P2.6); `name_components` polish is the designated slip-to-Day-3 item; and nothing downstream starves overnight because P1 takes `pairs_b` the same evening and `zscores_b` may land first thing Day 3. Nothing on Day 3 may slip: real triggers, the base-rate review, and E1's tuned grid rows are all validation-gate criteria.

---

### Slice P2 — definition of done

- [ ] Fixture artifacts pass Section 2.4 contract tests; golden triggers file hand-derived, committed with verifier's name, spans the train/val boundary
- [ ] Golden-file trigger test green Day 1 EOD; onset + re-arm policy documented in the `detect_triggers` docstring
- [ ] Split tests prove purge and embargo on synthetic dates; `purged`/`embargo` rows retained and countable
- [ ] All 7 features match hand-computed fixture values; no feature reads past `trigger_date`; scaler fit on train rows only and serialized with the model
- [ ] `triggers_a`, `triggers_b` (Day 3 AM) and `triggers_c` (Day 3 PM) conform exactly to the Section-2 contract schema, each built by the one shared CLI
- [ ] Bloomberg CSVs + `FIELDS_USED.md` with verified mnemonics and point-in-time notes — **or** fallback invoked at the Day 1 evening sync, built Day 2, logged, and disclosed in limitations
- [ ] `labels_b`, `stability_b`, `pairs_b` validate against Section 2 schemas Day 2 EOD; component-naming table exists
- [ ] Base-rate table reviewed at the Day 3 midday checkpoint; decision-table row executed and (if triggered) logged in DECISIONS.md; by-year decay figure in `results/figures/`
- [ ] E1 behind P3's `EntryModel` API, runnable by the runner without modification; tuned E1 × {a,b,c} rows in the Day 3 grid; τ per track in `taus.json` Day 3 evening, before any test-set contact
- [ ] E1 frozen Day 4 AM through P3's protocol, sha256 in DECISIONS.md; the witnessed run loaded it read-only
- [ ] E1 turnover-control percentile produced Day 4 PM from P3's machinery
- [ ] Coefficient tables per track + cross-track comparison delivered; error-analysis deep-dive explicitly cut, not half-built
- [ ] Consensus-lite table at pair-quarter granularity, bucketed by selection count across all three tracks
- [ ] Leakage items 5, 6, 10 closed with written evidence notes before the Day 4 EOD freeze

---

## 5. Slice P3 — Evaluator: engine, Track C, E3

P3 owns the scientific instruments every grid cell is scored by — the single backtest engine, the experiment runner with the guarded test path, the metrics/bootstrap/calibration module, the noise test, the turnover-matched control machinery, and the model-freeze protocol. These are built once and shared (Section 1): every (track × model) cell flows through identical machinery, which is what keeps the 3×4 factorial comparison fair and the turnover-matched control valid. On top of the spine, P3 owns the cheapest pairing track — Track C's partial-correlation distance reuses the in-window correlation matrices P1's PCA already computes and P1's shared clustering/pair-builder/spread machinery, so it is ~4h of new code — and the model most likely to demonstrate the course's bias–variance lesson: E3, the small MLP that is expected *not* to beat E1 (spec Step 9), reported honestly with overlapping CIs if that's what we get. Should be the strongest PyTorch person. **Days 1–2 run entirely on fixtures** (P2's `fixture_zscores` output + P3's own `make_synthetic`); real artifacts are touched for the first time at the Day 3 midday integration checkpoint.

Module map (all under the repo root):

```
Makefile                    # P3.1
src/contracts.py            # P3.1  (schema registry + validate_artifact + io helpers + seed_everything)
src/make_synthetic.py       # P3.1  (random-walk fixture, reused by the noise test)
src/engine.py               # P3.2
src/models/common.py        # P3.3  (EntryModel base + E0 decisions + the freeze harness; freeze execution = P3.10)
src/experiments.py          # P3.3  runner: grid/test subcommands, guarded test path
                            # P3.7  noise test · P3.8 cost sweep · P3.9 turnover-control machinery
src/metrics.py              # P3.4  (metrics + calibration + bootstrap, one module)
src/partial_corr.py         # P3.5
src/models/e3.py            # P3.6
tests/test_contracts.py  tests/test_engine.py (incl. cost tests)  tests/test_experiments.py (guard + control tests)  tests/test_partial_corr.py
```

---

### P3.1 — Repo plumbing: contracts (+io helpers), Makefile, `make_synthetic` (Day 1 AM, 4h)

**Goal.** The Section 2 skeleton everyone codes against from hour one, plus the pure-noise price fixture.

- **`src/contracts.py` (1.5h).** One `ArtifactSchema(columns: dict[str, str], index: str | None, checks: list[Callable])` per artifact in Section 2.2, and `validate_artifact(df, name)` asserting: exact column set, exact dtypes, index monotonic-increasing and unique, and per-artifact invariants (pair_id alphabetical and matching `stock_a`/`stock_b`; `split` ∈ the 5 allowed values; `label` ∈ {0,1}; every `net_ret_{c}bps` column present for the full cost grid; `source` ∈ {`track_a`, `track_b`, `track_c`} — track_d branch removed in v2). Producers call it before every write; `tests/test_contracts.py` runs it against the committed fixture artifacts as goldens. Done when deliberately renaming a column in a fixture file fails the suite.
- **io helpers + Makefile (1h).** `contracts.py`'s `write_parquet`/`read_parquet` helpers route every write through `validate_artifact`; Makefile targets per Section 2.5 (`make data / tracka / trackb / trackc / dataset / grid / noise-test / test-run / test / figures`), stubs where the pipeline doesn't exist yet. `make test` green on the skeleton is a Day 1 kickoff exit criterion.
- **`src/make_synthetic.py` (1.5h).** The random-walk fixture per the Section 2.4 spec verbatim: NYSE calendar 2014-01-01→2025-01-01, tickers SYN00..SYN39 + independent SPY walk, log prices `p_t = p_{t-1} + eps`, `eps ~ N(0, sigma_i)` with `sigma_i ~ U(0.008, 0.025)` drawn once per ticker, `p_0 = ln(U(20, 500))`, volume `v_t = round(base_i * exp(N(0, 0.3)))`, `base_i ~ logU(1e5, 1e7)`, single `np.random.default_rng(seed=311)`. Byte-for-byte the same schemas as the real raw artifacts — which is what makes P3.7's noise test trivial: point the pipeline at `data/synth/raw/` and change nothing else. CLI: `python -m src.make_synthetic --seed 311 --out data/synth/raw`.

- **Consumes:** nothing (Section 2.2 schemas; seeded RNG). **Produces:** `src/contracts.py` (schemas + io helpers), `Makefile`, `data/synth/raw/{prices,volume,spy}.parquet`.
- **Definition of done:** schema tests pass on the synthetic outputs; two same-seed runs byte-identical; P1 develops the rolling PCA against the synthetic prices the same afternoon.
- **Hours:** 4h. **Day:** 1 AM (immediately after kickoff).

### P3.2 — Backtest engine + trade ledger + cost columns (Day 1 PM + Day 2 AM, 5.5h)

**Goal.** The single engine every strategy flows through (Key Architectural Rule, Section 2). Built and golden-tested on **P2's `fixture_zscores` output** (fixture leg prices + golden triggers land Day 1) plus a **hand-written decisions table** — so the engine runs before any model or trigger code exists. Engine v1 golden-tested is a Day 1 EOD anchor.

**Module.** `src/engine.py`

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
- **Exit:** first trading day with `|z| < 0.5` (exit at that day's close, `exit_reason="reverted"`), else at close 5 trading days after entry (`exit_reason="timeout"`). `days_held` = trading days entry→exit. **Disclosed asymmetry:** entry is lagged one day (t+1), but the exit fills at the same close that generated the exit signal — this matches the spec's own E0 wording (Step 9) and is stated explicitly in the report's execution paragraph and the manual-trace template so leakage-checklist item 8 reads airtight. (The manual trace itself is **P1's task** in v2 — deliberately a non-author of this engine hand-verifies the ledger.)
- **P&L from RAW stock returns**, not residuals: the engine derives **simple** returns internally from `prices` (`p/p.shift(1) - 1`; log returns would misprice the short leg). Per-day trade P&L in dollars: `pnl_t = r_long_t − r_short_t`; `gross_ret = Σ pnl_t` over `(entry_date, exit_date]` — accumulated per-day so `daily_strategy_returns` can reuse the same numbers.
- **Costs at transaction time:** `c` bps per leg per transaction → 2 legs at entry + 2 legs at exit = **4 applications**. With fixed $1/leg notional: `net_ret_{c}bps = gross_ret − 4 × c × 1e-4`, but *booked* as −2c bps on the entry day and −2c bps on the exit day in the daily P&L stream (spec Step 11: at the point of each transaction, never a lump at the end). One `net_ret_{c}bps` column per grid value, precomputed — P3.8's sweep then reads columns, no reruns; P3.9's 1000 control runs become groupbys over sampled rows.
- **Aggregation policy (disclosed in the report):** strategy daily return = **equal-weight mean of all concurrently open trades' daily P&L** (in per-$1-per-leg units); days with no open trades contribute 0; **no capital constraint** — every accepted trigger is taken regardless of how many trades are already open. Stylized, stated, identical across all strategies, therefore fair.

**Tests** (`tests/test_engine.py`, `tests/test_engine.py`) — written against fixtures, must stay green forever:

1. **3-trade golden test:** hand-computed fixture (one reverted, one timeout, one z<0 direction flip) — assert `entry_date`, `exit_date`, `exit_reason`, `days_held`, `gross_ret` to 1e-10 against the hand math committed as a CSV next to the test. P&L is derived from the fixture's 12-ticker `prices.parquet` (Section 2.4 — P2's fixture ships leg prices precisely so this test can run); the decisions input is a literal hand-written table (`trigger_id, enter=True, p_hat=NaN`) committed next to the golden CSV.
2. **Shift test (leakage):** perturb day-t return of a traded stock with the trade triggered at t; assert ledger unchanged. Then perturb day t+2 (first P&L day); assert `gross_ret` moves by exactly the perturbation. Proves day-t signal never touches day-t return.
3. **Cost arithmetic test:** for the golden trades, assert `net_ret_10bps == gross_ret − 0.004` exactly, and that the daily P&L stream books −2c bps on entry and exit days.
4. **Concurrency test:** two overlapping trades → `daily_strategy_returns` equals hand-computed equal-weight mean.

- **Consumes:** P2's fixture (Day 1–2), then `zscores_{track}`, `triggers_{track}`, `prices.parquet`, decisions tables (Day 3). **Produces:** `results/trades_{track}_{model}.parquet` per the Section 2 contract (writing routed through P3.3's runner).
- **Definition of done:** golden + shift tests green Day 1 EOD (anchor); all four tests green Day 2 AM; first real Track A ledger produced Day 3 AM when P2's `triggers_a` lands.
- **Hours:** 5.5h (Day 1 PM: 3.5h core loop + golden + shift tests; Day 2 AM: 2h costs/aggregation + remaining tests).

### P3.3 — E0 baseline + `EntryModel` base + experiment runner with the guarded test path (Day 2, 4.5h)

**Goal.** The fixed rule as a decisions table — through the SAME engine, so E0 is never special-cased — plus the **canonical `EntryModel` base class and freeze harness**, and one command per grid cell with the test set physically guarded. Day 2 EOD anchor: runner + metrics + the `EntryModel` API round-trip all working on fixture triggers.

**`src/models/common.py` (1h) — owner P3, the canonical interface (contract text lives in Section 2.2):** the abstract `EntryModel` with `fit(X_train, y_train)` (fits the StandardScaler on train rows only, then the model; returns self), `tune(X_val, y_val)` (hyperparameter selection by validation AUC, refits at best config, returns the report dict), `predict_proba(X)`, `get_params_report()`, `save(path)` / `load(path)` (model + fitted scaler bundled in one artifact — what the freeze protocol hashes). P1 (E2) and P2 (E1) subclass it, never modify it; merge rights P3. Alongside it, the `src/models/common.py (freeze harness)` harness (hashing + `FREEZE.md` writer + load-only smoke test) is written now, Day 2 — it is a noon-Day-4 hard dependency and is not born in the Day 4 buffer window (execution of the freeze is Day 4 AM, P3.10).

**E0, in `src/models/common.py` (0.5h):**

```python
def e0_decisions(triggers: pd.DataFrame) -> pd.DataFrame:
    """Every trigger: enter=True, p_hat=NaN. Columns: trigger_id, enter, p_hat."""
```

Three lines of logic; its value is architectural. E1 (P2), E2 (P1), and E3 (P3.6) produce the identical schema via the shared `EntryModel` API (Section 2 contract), and the turnover control (P3.9) produces it too — the engine cannot tell strategies apart.

**`src/experiments.py` (2.5h):**

```python
def run_cell(track: str, model: str, split: str) -> dict:
    """Loads triggers_{track}, filters to split rows, obtains decisions:
    e0 -> e0_decisions(); e1/e2/e3 -> the shared EntryModel API
    (EntryModel.load(model, track).predict_proba(features) -> p_hat, with tau
    applied from the frozen taus.json; fit/tuning happen in each owner's model
    code on train/val only). Runs run_backtest, computes metrics via P3.4,
    writes decisions/trades/metrics artifacts. Returns metrics dict."""

# CLI: python -m src.experiments grid --tracks a,b,c --models e0,e1,e2,e3 --split val
```

**Test-set guard (hard, not procedural):** `--split test` additionally requires `--i-am-sure` AND the date check **`date.today() >= date(2026, 8, 5)`** (Day 4) AND the presence of `results/FREEZE.md` (written Day 4 AM after the freeze protocol, listing frozen taus/hyperparameters/hashes) AND the **absence of the no-second-run marker** `results/final/TEST_RUN_COMPLETE`. Any missing condition → `sys.exit` with an explanatory message. On a successful test run the runner writes `TEST_RUN_COMPLETE` (timestamp + git sha) as its last act — the witnessed run is physically single-shot; a rerun requires deleting a committed marker, which is a team decision logged in `DECISIONS.md`. Guarded in `tests/test_experiments.py` (monkeypatched date + marker fixture). Additionally, `run_cell` asserts it never reads rows with `split` ∈ {`purged`, `embargo`} into any fit or metric.

**Aggregation:** `make_grid_table()` → `results/tables/grid_3x4.md` + `.csv`: rows = tracks (a, b, c), columns = models (E0–E3), each cell showing AUC [CI], net return @10bps [CI], trade count — the report's main table, **with row and column marginal means** so spec 12a's three questions (row effect, column effect, interaction) are read directly off it. The grid is committed at **3×4 = 12 cells** (Track D cut in v2; multiple-testing arithmetic in Section 6 uses 12). If Track C is de-scoped at the Day 3 checkpoint, the table degrades to 2×4 with the decision logged — it never blocks the run.

- **Consumes:** triggers (fixture Day 2, real Day 3), P3.2, P3.4. **Produces:** `src/models/common.py` + `src/models/common.py (freeze harness)`, `results/decisions_*.parquet`, `results/trades_*.parquet`, `results/metrics_{track}_{model}.json`, `results/tables/grid_3x4.*`.
- **Definition of done:** Day 2 EOD — fixture triggers × {E0, dummy EntryModel subclass} run end-to-end with one command; guard test green; P1/P2 have subclassed `EntryModel` against the committed base.
- **Hours:** 4.5h. **Day:** 2.

### P3.4 — Metrics, calibration, bootstrap (Day 2, 3h)

**Consumes:** `triggers_{track}` + `decisions_*` + `results/trades_{track}_{model}.parquet`. **Produces:** the classification half of `results/metrics_{track}_{model}.json`; `results/figures/calibration_{track}_{model}.png`; bootstrap CI machinery used by every metric in the project (P1 and P2 call the same `bootstrap_ci`).

```python
# src/metrics.py
def classification_metrics(y_true: np.ndarray, p_hat: np.ndarray, tau: float) -> dict:
    ...  # {"auc": ..., "precision_at_tau": ..., "recall_at_tau": ..., "brier": ..., "n": ...}

# src/metrics.py  (calibration section)
def reliability_diagram(y_true, p_hat, n_bins: int = 10, out_path: Path) -> pd.DataFrame:
    ...  # 10 QUANTILE bins (equal-count, not equal-width — our p_hat range is narrow);
         # plot predicted vs observed rate with per-bin counts annotated; returns the bin table

# src/metrics.py  (bootstrap section)
def bootstrap_ci(values_or_frame, stat_fn: Callable, n_boot: int = 1000,
                 seed: int = config.SEED) -> tuple[float, float, float]:
    ...  # (point, lo, hi) — percentile CIs at 2.5/97.5
```

Bootstrap protocol: **classification metrics resample TRIGGERS i.i.d.** (rows of the trigger table); **strategy metrics resample TRADES i.i.d.** (rows of the trade ledger). 1000 resamples, seed 311, percentile intervals. **Independence caveat, in the `metrics.bootstrap_ci` docstring verbatim and as a limitations bullet:** trades overlap in time and cluster by pair and regime, so i.i.d. resampling understates true uncertainty; a block bootstrap would be more faithful but is out of scope — CIs are therefore *optimistic lower bounds on width*. AUC is the primary classification metric (spec 10d); accuracy is never reported. Sanity anchor: expected AUC 0.52–0.58; **anything ≥ 0.65 is treated as a bug report** (spec 10e) and triggers a leakage re-check before anyone celebrates. All figures land in `results/figures/` as `{figure}_{track}_{model}.png`.

- **Definition of done:** metrics run on fixture outputs Day 2 EOD (anchor); every reported number in the JSONs carries a CI; calibration figures for all cells finalized Day 4 PM (each slice owner reads their own model's diagram — P1 writes the calibration commentary for E2, per Section 3).
- **Hours:** 2.5h. **Day:** 2 (figure finalization inside Day 4 PM's figure block).

### P3.5 — Track C: partial-correlation distance (Day 3 AM, 3.5h — committed, with a de-scope valve)

Track C is **committed, not gated** — the grid is 3×4. One de-scope rule survives: **if Track C is visibly behind at the Day 3 midday checkpoint, it ships on validation only (or drops entirely, logged in `DECISIONS.md`) and never blocks the Day 4 test run.** Everything downstream of pairs is untouched — that is the point: Track C changes only the *distance metric for grouping*; the tradeable object is still the Track A residual spread.

**Scheduling (v2 fix):** the whole task runs **Day 3 AM** — its only real-data input (`corr_windows.npz`, below) exists from Day 1 EOD, so there is no reason to start after the checkpoint. Target: `pairs_c.csv` lands **at or before the midday checkpoint**, so the checkpoint judges a nearly-done artifact and the `zscores_c` → `triggers_c` → tune chain closes by mid-afternoon, not at the 9pm gate.

**Consumes:** `data/processed/corr_windows.npz` — the per-window 40×40 in-window correlation matrices, a **Section 2.2 contracted artifact P1's rolling PCA writes Day 1 EOD** (a file hand-off, per the plan's operating principle — not a call into P1's code); P1's shared P1.5/P1.6 clustering + pair-builder machinery; `residuals_a`. **Produces:** `labels_c.parquet`, `stability_c.parquet`, `pairs_c.csv` (source `track_c`).

| | Task | Hours |
|---|---|---|
| P3.5.1 | `src/partial_corr.py`: `partial_corr_distance(C: np.ndarray, lam: float = 1e-3) -> np.ndarray` — diagonal shrinkage `(1-lam)*C + lam*I`, invert to precision `P`, `pcorr = -P/outer(sqrt(diag(P)))` with unit diagonal, `dist = sqrt(2*(1-pcorr))`. Unit tests: pcorr symmetric, diag 1, known 3-variable analytic case; near-singular C inverts stably after shrinkage. `lam` = `config.PARTIAL_CORR_SHRINKAGE`, recorded in the report; not tuned. | 1.5 |
| P3.5.2 | Wire into P1's recluster loop at the same 21-day cadence, formation-window matrices only (leakage item 11). k-means needs vectors, not distances: fit on each stock's 40-dim row of the distance matrix (standard workaround; state it in the report), but **select k via `silhouette_score(D, labels, metric="precomputed")` on the actual distance matrix** — D is the true dissimilarity. Same k-range 8–13, seed 311. Write `labels_c`, `stability_c`. | 1.5 |
| P3.5.3 | Pairs via P1's shared builder (`build_pairs(..., source="track_c")`) → `pairs_c.csv`. **Hand-offs, named:** P3 hands `pairs_c.csv` to **P1**, who runs the shared spread/z-score module against `residuals_a` → `zscores_c.parquet` (one function call); P3 then pings **P2**, whose dataset builder produces `triggers_c` with one command (`python -m src.dataset --tracks c`). Target: `zscores_c`/`triggers_c` by ~2–3pm so Track C's models tune with everyone else's, well before the gate. | 0.5 |
| P3.5.4 | Overlap measurement (spec 8.1 — committed now that Track C is committed): mean per-window **Jaccard overlap of Track C vs Track A co-membership sets** — a ~15-minute computation on the existing `stability`/`labels` artifacts, one committed number + row in `results/tables/` feeding one report sentence. The figure `trackc_overlap.png` stays optional Day 4 polish. | 0.25 (+0.5 optional figure) |

- **Definition of done:** `pairs_c.csv` validates against the Section 2 schema by the midday checkpoint; `zscores_c` (P1) and `triggers_c` (P2) land early-mid PM; Track C row present in the validation grid at the Day 3 evening gate; the Jaccard overlap number committed; noise test covers the Track C path (P3.7); de-scope decision, if taken, logged.
- **Hours:** 3.5h (excl. the optional overlap figure). **Day:** 3 AM (module first thing; `pairs_c` by the midday checkpoint; hand-offs early PM).

### P3.6 — E3 small MLP + τ selection (Day 3 PM/evening, 2.5h; freeze Day 4 AM)

**Consumes:** `triggers_{track}.parquet` (real, from Day 3 AM). **Produces:** `src/models/e3.py` behind the shared `EntryModel` API; E3 and E0 rows × {a, b, c} on train+validation; τ values for E3.

Architecture and training, fixed a priori (unchanged from v1):

- `7 → h → 1`, `h ∈ {8, 16}`, ReLU hidden, sigmoid output (implemented as `BCEWithLogitsLoss` on the raw logit, with `pos_weight = n_neg/n_pos` — the `class_weight='balanced'` analog).
- Adam, lr `1e-3`, full-batch (a few hundred–low thousands of triggers; batching is pointless), weight decay grid `{1e-4, 1e-3, 1e-2}` → 6 configs total.
- Early stopping on **validation AUC**, patience 25 epochs, `max_epochs = 500`; keep the best-val-AUC checkpoint.
- Seeding: `torch.manual_seed(config.SEED)`, `numpy` and `random` likewise, deterministic algorithms on; **CPU only** — a 7→16→1 net on ~1–2k rows trains in **under 10 seconds per config, under 2 minutes for the full grid**; anyone waiting longer has a bug.

**Expected result (spec Step 9): E3 does not beat E1.** The deliverable is not "a neural net" but the **bias–variance comparison** — the E0→E1→E3 ladder on real noisy data, reported with overlapping CIs if that's what we get. Keep it small on purpose; do not add layers to "fix" a null result.

**τ selection — the one pre-registered rule, quoted here once; all three slices apply it identically to their own model on Day 3 evening** (v2 change: all models tune together Day 3; the old E3-finalizes-later split is gone). For each (track, model), τ is chosen from the grid `{0.40, 0.45, …, 0.80}` to **maximize validation net P&L at c = 10 bps, subject to ≥ 25 accepted validation trades**; ties (including P&L ties within $1e-6) break toward the **higher** τ (prefer trading less when indifferent). If no τ on the grid meets the 25-trade floor, choose the τ that **maximizes accepted validation trades** (ties toward higher τ); if even that maximum is 0 trades, the cell is reported as **degenerate** with τ = 0.5 and a `DECISIONS.md` note — the rule has no undefined branch. τ is chosen per model per track, on validation only, never revisited after the P3.10 freeze — the Day 4 test run loads it from `results/frozen/taus.json` read-only. Mechanics: each owner computes decisions tables for each candidate τ; the engine (interface: `decisions → trades → net P&L`) scores them; the owner picks and records. E0 has no τ (`p_hat = NaN`, `enter = True` always). The rule is written into `DECISIONS.md` **before** any test-set run exists.

- **Definition of done:** tuned E3 reproduces byte-identical `p_hat` across two runs (seeding verified); E3 and E0 rows for all three tracks in the Day 3 evening validation grid; τ values for E3 in `taus.json` Day 3 evening; E3 frozen through P3.10 Day 4 AM.
- **Hours:** 2.5h. **Day:** 3 PM/evening.

### P3.7 — Noise test (Day 3 PM, 2h — validation-gate criterion)

**Goal.** Run the FULL pipeline — PCA → residuals → clusters → pairs → z-scores → triggers → all models → engine → metrics — on `make_synthetic` random-walk prices (P3.1's Day-1 fixture generator, reused verbatim, same seed). If the pipeline finds signal in pure noise, we have a leakage bug. Because all four models exist by Day 3 (v2 change), **one run covers E0/E1/E2/E3 at once** — no rerun needed before the freeze.

**Module.** `src/experiments.py`, invoked as `make noise-test` (writes everything under `results/noise/`, seeded 311, never mixed with real artifacts).

**Concrete pass criteria** (evaluated automatically, printed as PASS/FAIL per line). One subtlety first: on random walks the label is *mechanically* related to the features without any leakage — P(|z| halves within 5 days) genuinely falls with |z_trigger| (a 3.5σ excursion must travel farther than a 2.1σ one) and varies with the level-vol/increment-vol ratio, and both quantities are features. So "AUC above 0.5 on noise" is **not** by itself evidence of a bug, and the binding criteria are:

1. **Returns (binding):** every strategy's mean net return per trade at c=10 is ≤ 0, or its 95% CI covers 0. (Gross may be mildly positive — z-score rules harvest some mean-reversion even from random walks via the rolling-window construction; *net-of-cost with a CI excluding zero on noise* is the red flag.)
2. **AUC vs the mechanical baseline (binding):** each model's AUC is not significantly above the AUC of an `f_abs_z`-only logistic model on the same synthetic triggers (bootstrap the AUC *difference*; its CI must cover 0). The raw model-AUC-vs-0.5 CI is printed as an **advisory** line only — it is expected to exclude 0.5 for the mechanical reason above and does not trigger the stop-the-line protocol by itself.
3. **Trigger count sanity:** triggers exist (> 50 — the machinery demonstrably fires) and the base rate is strictly inside (0, 1).

If Track C is built by run time, its path is included (Part 4: any extension that runs must pass); if Track C lands after the noise run, the rerun is one `make noise-test` — minutes, not hours.

**Failure = stop-the-line:** immediate all-hands message; nobody merges anything downstream of the suspected stage until the cause is found; the leakage checklist (spec Part 4) is re-walked item by item as a team; resolution logged in `DECISIONS.md`. The Day 3 evening validation gate cannot pass while the noise test is red.

- **Consumes:** `make_synthetic`, the entire pipeline (P1's and P2's modules included — this is also an integration test). **Produces:** `results/noise/metrics_*.json`, `results/noise/PASS_FAIL.md` (goes in the report's leakage section).
- **Definition of done:** `make noise-test` runs green end-to-end before the Day 3 evening sync.
- **Hours:** 2h (the pipeline already runs; this is orchestration + criteria). **Day:** 3 PM.

### P3.8 — Cost sweep + signature figure (Day 4 PM, 2h)

**Goal.** Spec Step 11's sweep and breakeven analysis, from the ledger's precomputed `net_ret_{c}bps` columns — zero backtest reruns. Both the table and the figure land Day 4 PM (validation numbers refreshed with test numbers after the noon witnessed run — reading the frozen test ledgers, never rerunning them).

**Module.** `src/experiments.py`

```python
def breakeven_bps(trades: pd.DataFrame) -> float:
    """Mean net return per trade is linear in c (slope -4e-4), so breakeven is exact:
    c* = mean(gross_ret) / 4e-4; also verified by interpolating across the grid columns."""

def cost_sweep_figure(trades_by_strategy: dict[str, pd.DataFrame],
                      out: str = "results/figures/cost_sweep.png") -> None:
    """Net cumulative return vs c, one line per strategy, breakeven c* annotated
    on each line, vertical reference at headline c=10."""
```

- **Consumes:** `results/trades_{track}_{model}.parquet`. **Produces:** `results/tables/breakevens.csv`, `results/figures/cost_sweep.png` — all 12 strategies (or 8, if Track C was de-scoped).
- **Tests:** golden trades → breakeven matches hand computation; interpolated and analytic breakevens agree to 1e-6.
- **Definition of done:** table + figure in `results/` before the Day 4 EOD results-freeze sync.
- **Hours:** 2h. **Day:** 4 PM.

### P3.9 — Turnover-matched control: machinery + P3's runs (machinery Day 4 AM 1.5h; runs Day 4 PM 1h — spec 12b)

**Goal.** The project's headline experiment: is the filter's edge signal, or just fewer trades?

**Module.** `src/experiments.py`

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

**Ownership split (v2):** P3 builds the machinery and runs it for its **own** cells — E3 × {a, b, c}. E0 needs no control (it accepts everything, so its matched control ≡ E0 by construction — which is exactly the machinery's built-in correctness test, see below). **P1 and P2 run their own models through the same one-call interface** on Day 4 PM: `run_control(track, model)` reading their `results/decisions_{track}_{model}.parquet` — no P3 involvement needed beyond the module being green.

- **Consumes:** `results/decisions_{track}_{model}.parquet` (test versions from the frozen noon run), `triggers_{track}`, P3.2. **Produces:** `results/control_{track}_{model}.json`, `results/figures/control_hist_{track}_{model}.png`.
- **Tests (`tests/test_experiments.py`):** with a degenerate model accepting everything, control ≡ E0 per quarter; per-quarter counts of every draw exactly match the model's; seed 311 reproduces bit-identical distributions.
- **Definition of done:** machinery tests green **before the noon test run** (built Day 4 AM, right after the freeze, so P1/P2 can call it immediately after the noon run without serializing behind P3); percentile + histogram for E3 × {a, b, c}; the Track A × E1 primary-comparison control is P2's Day 4 PM deliverable through this interface.
- **Hours:** 1.5h machinery (Day 4 AM) + 1h P3's own runs (Day 4 PM).

### P3.10 — Model-freeze execution (Day 4 AM, 0.5h for P3; machinery built in P3.3)

**Goal.** The noon witnessed run is load-only. The `src/models/common.py (freeze harness)` machinery (hashing, `FREEZE.md` writer, load-only smoke test) was **built Day 2 as part of P3.3** — Day 4 AM only *executes* it. **Each person freezes their OWN model through it** (v2 change): P2 → `e1_{track}.joblib`, P1 → `e2_{track}.joblib`, P3 → `e3_{track}.pt` (state_dict + scaler params + architecture config in one torch archive), each bundling model + fitted scaler via `EntryModel.save`, plus the completed `results/frozen/taus.json`.

- Per (track, model) artifact under `results/frozen/`; after all three have frozen, a `sha256sum results/frozen/* >> DECISIONS.md` block with timestamp and all three names.
- **Load-only smoke test:** a clean process runs `EntryModel.load` on each file, `predict_proba` on 5 validation rows, byte-identical to pre-freeze output.
- `results/FREEZE.md` written Day 4 AM (frozen taus/hyperparameters/hashes) — one of the runner's hard preconditions for `--split test` (P3.3).
- **The noon witnessed test run loads ONLY these artifacts** — it calls `load`, never `fit` or `tune`; any code path in the test-run script that could reach `fit` raises. If a frozen hash doesn't match at load time, the run aborts.

- **Definition of done:** smoke test green for all frozen cells; hashes in `DECISIONS.md`; `FREEZE.md` committed; the ~noon run (all three present, guarded runner path, single-shot marker) completes from frozen artifacts only.
- **Hours:** 0.5h (P3's own freeze + FREEZE.md; machinery hours live in P3.3). **Day:** 4 AM, first thing.

### P3.11 — Leakage-audit items 7–9 and 11 (Day 4, 0.5h)

P3 owns the four Part 4 checklist items that live in its machinery, each verified mechanically by an existing test and written up as a one-paragraph note in `results/audit/audit_p3_items.md` (pasted into the report's checklist section):

| Item | Evidence |
|---|---|
| 7. Day-t signal never touches day-t P&L | P3.2 shift test (perturb day-t return → ledger unchanged; perturb t+2 → moves exactly); cite the test. |
| 8. t/t+1 entry alignment + disclosed exit convention | The t+1 rule and the exit-at-signal-close asymmetry stated in P3.2; independently hand-verified by **P1's manual trace** (Section 3) on P3's ledger — a non-author of the engine checks the engine; cite the 3/3 MATCH verdict. |
| 9. Costs at transaction time | P3.2 cost-arithmetic test (−2c bps booked on entry day and exit day, never a lump); cite the test. |
| 11. Track C distances from formation-window matrices only | P3.5.2 wiring note (the correlation matrices are P1's in-window matrices, which the item-1 perturbation audit already covers) + Track C path green in the noise test. |

- **Definition of done:** four PASS lines + notes committed before the Day 4 EOD results freeze.
- **Hours:** 0.5h. **Day:** 4.

---

### P3 — day-by-day hour budget (~34h)

| Day | Date | Tasks | Hours |
|---|---|---|---|
| 1 | Sun Aug 2 | Kickoff (1h) · P3.1 contracts (+io helpers) + Makefile + `make_synthetic` (4h) · P3.2 engine core + golden/shift tests on P2's fixture (3.5h) — **engine v1 golden-tested EOD (anchor)** | 8.5 |
| 2 | Mon Aug 3 | P3.2 costs/aggregation + remaining tests (2h) · P3.3 E0 + `EntryModel` base + freeze harness + runner + guard, round-trip on fixture triggers (4.5h) · P3.4 metrics/bootstrap/calibration (2.5h) — **runner + metrics + API working on fixtures EOD (anchor)** | 9 |
| 3 | Tue Aug 4 | **AM:** P3.5 Track C module + wiring → `pairs_c` at the midday checkpoint (3h) · checkpoint (0.5h) · **PM:** P3.5.3 hand-offs + Jaccard number (0.75h) · P3.6 E3 fit/tune + τ (2.5h) · P3.7 noise test (2h) · **evening validation gate** | 8.75 |
| 4 | Wed Aug 5 | P3.10 freeze execution + `FREEZE.md`, first thing (0.5h) · P3.9 control machinery green pre-noon (1.5h) · red burn-down buffer (0.5h) · witnessed test run, ~noon (1h) · P3.8 cost sweep (2h) · P3.9 controls for E3 (1h) · T2 grid table + F6 assembly (1h) · P3.11 audit notes (0.5h) · **EOD results-freeze sync (git tag)** | 8 |

Total ≈ 34h. Day 3 AM is Track C by design — its input (`corr_windows.npz`) exists from Day 1 EOD, so `pairs_c` lands at the midday checkpoint and the afternoon holds E3 + the noise test with real slack. The Track C de-scope rule is the pressure valve; the engine, runner, E3, and noise test are the never-cut items because every other slice's Day 3–4 work flows through them. Aug 6–7 (write-up) is out of scope for this plan.

### Slice P3 — definition of done

- [x] `contracts.py` validates every Section 2.2 artifact; deliberately corrupting a fixture fails the suite; `make test` green on the Day 1 skeleton
- [x] `make_synthetic` outputs match the real raw schemas byte-for-byte; two same-seed runs identical
- [ ] Engine test suite green: golden 3-trade, shift test, cost arithmetic, concurrency — golden + shift by Day 1 EOD
- [ ] Every strategy (E0, E1, E2, E3, control) flows through `run_backtest` via a decisions table — zero special cases
- [ ] `--split test` guard proven by test (date ≥ 2026-08-05, `--i-am-sure`, `FREEZE.md` present, no `TEST_RUN_COMPLETE` marker); test set executed exactly once, Day 4 ~noon, witnessed, from frozen artifacts; marker written
- [ ] `grid_3x4` table with row/column marginal means on train+val Day 3 evening, on test Day 4 PM
- [ ] Every reported metric carries a 1000-resample percentile CI (triggers for classification, trades for strategy); independence caveat in docstring + limitations; any AUC ≥ 0.65 investigated as a bug before being reported
- [ ] Track C: `pairs_c.csv` (source `track_c`) via the shared builder; `zscores_c` (P1) and `triggers_c` (P2) hand-offs landed Day 3 evening — or the de-scope decision logged in `DECISIONS.md`, never blocking the Day 4 run
- [ ] E3 behind the shared `EntryModel` API, byte-reproducible under seed 311; τ selected Day 3 evening by the pre-registered rule, recorded in `taus.json` before any test-set contact
- [ ] `make noise-test` green with all three pass criteria across all four models (Track C path included if built); `PASS_FAIL.md` in repo; the Day 3 validation gate did not pass on a red noise test
- [ ] Cost-sweep figure + breakeven table for all grid strategies; analytic and interpolated breakevens agree to 1e-6
- [ ] Turnover-control machinery tests green (degenerate-model identity, exact per-quarter counts, seed-311 reproducibility); E3 controls run by P3; P1/P2 ran their own models through `run_control` Day 4 PM
- [ ] `results/frozen/` complete Day 4 AM with sha256 hashes in `DECISIONS.md`; load-only smoke test green; the witnessed run touched no `fit`/`tune` path
- [ ] Leakage items 7–9 + 11: PASS lines + notes committed
- [ ] Every stochastic component in P3 code seeded from 311; two consecutive runs of any P3 stage produce bit-identical artifacts

---

## 6. Timeline, coordination, and risk

This section is the operating schedule for the plan in Sections 3–5. Task IDs (P1.1…, P2.1…, P3.1…) refer to the per-slice task lists in those sections; every reference below also names the task. All times EST. Artifact names are the canonical contracts from Section 2; a **bold** entry means that artifact lands (is merged to `main`) that block.

**Calendar:** Day 1 = **today, Sunday August 2** through Day 4 = Wednesday August 5. Results — including the single witnessed test run — freeze **Day 4 EOD**. August 6–7 is the write-up window, which the team plans separately: **report prose is deliberately out of scope for this plan and is not scheduled anywhere in Days 1–4.** Figures and tables *are* produced on Days 3–4, because they are evaluation outputs regenerated from frozen artifacts, not writing.

### 6.1 Day-by-day plan

Conventions: AM ≈ before 2pm, PM ≈ after 2pm through the 9pm sync. Consumes/produces for each task is specified in Sections 3–5; here we list only the landings. **Day 1 is a Sunday** and the build runs through midweek — the entire schedule rests on the availability commitment made at the Day 1 kickoff: each person confirms **~8–9h/day for all four days, August 2–5**, verbally, recorded in `DECISIONS.md`, against the actual plan below and not a bare floor. There is no slack day in this schedule; if anyone cannot commit, the coverage rules in the risk register (6.4, risk R7) apply immediately, not reactively.

#### Day 1 — Sunday, August 2

| | P1 — Substrate | P2 — Training data | P3 — Evaluator |
|---|---|---|---|
| **AM** | **1h kickoff, as early as possible (all three):** ratify slice assignments against actual skills (P2 = Bloomberg access, P3 = strongest PyTorch, P1 = heaviest math, no Bloomberg needed); freeze `src/config.py`; confirm the ~8–9h/day Aug 2–5 availability commitment; assign the Bloomberg Monday-terminal-access question to P2 (answer due at tonight's sync); repo init + stubs; seed `docs/limitations.md` with every spec Part 7 bullet as an owner-tagged stub. Then P1.1 (universe + `yfinance` download + cleaning): **`data/raw/prices.parquet`, `data/raw/volume.parquet`, `data/raw/spy.parquet`**. | Kickoff. Then P2.1 (planted-OU fixture): **`src/fixture_zscores.py`** + the hand-computed golden triggers file, finished and committed. | Kickoff. Then P3.1 (`src/contracts.py` v1 (schemas + io helpers) + `Makefile` + **`src/make_synthetic.py`**, the random-walk fixture later reused verbatim as the noise-test input) so every stub imports against real contracts from hour 2. |
| **PM** | P1.1 cleaning checks (missing-day threshold, forward-fill, calendar align, return-outlier scan) + log returns: **`data/processed/returns.parquet`** by EOD. P1.2 (rolling PCA engine: window loop, in-window standardization, `eigh`, sign fix, component-count rule) unit-tested on P3.1's synthetic prices, then **running end-to-end on real `returns.parquet`** by EOD. | P2.2 (trigger detection + labels) against fixture z-scores; done when the golden-file test asserts exact expected trigger dates and labels — **trigger/label code golden-tested green** by EOD. P2.3 (chronological splits + purge-5d + embargo-10d) done + tested on synthetic dates spanning both boundaries (moved to Day 1 to keep Day 2 ≤ ~9h). | P3.2 (backtest engine v1: t+1 entry, exit on \|z\|<0.5 or 5 days, $1/leg, raw-return P&L, per-transaction cost columns) against `fixture_zscores` output + a hand-written fixture decisions table. Done when a hand-computed 3-trade fixture ledger matches engine output to the cent: **engine v1 golden-tested** by EOD. |
| **Evening sync (9pm)** | **Bloomberg decision** (see 6.2): Monday-morning terminal access confirmed → P2 pulls Day 2 AM; not confirmed → Track B builds on the fallback columns immediately. One-way decision, logged. | | |
| Hours | P1: 8.5 (1 + 7.5) | P2: 9.25 (1 + 8.25) | P3: 8.5 (1 + 7.5) |

EOD state (anchor): repo + contracts on `main`; both fixtures committed; real `returns.parquet` landed; engine and trigger/label code both golden-tested; rolling PCA run on real returns (**`factors_a` + `corr_windows.npz` on disk**); P2.3 splits tested.

#### Day 2 — Monday, August 3

| | P1 — Substrate | P2 — Training data | P3 — Evaluator |
|---|---|---|---|
| **AM** | **In this order:** P1.3 (per-stock OLS betas + out-of-sample residuals — **`residuals_a.parquet` by midday**, on top of Day 1's `factors_a`/`pca_meta`/`loadings_a`/`corr_windows.npz`) · the shared future-perturbation leakage helper (`tests/leakage_utils.py`, 1h — P2's afternoon look-ahead tests consume it) · P1.6 pair-builder wiring + `v1-frozen` tag **by midday** (P2 calls it this afternoon). | Per last night's decision: P2.6 (Bloomberg pull, full 40 tickers × 19 fields × ~40 quarters in one terminal session, raw CSVs committed; **mid-session abort: no committed CSVs by ~noon → fallback immediately**) **or** the disclosed fallback columns (sector one-hots + price-derived fields from our own data) — either way Track B is building this morning. | P3.3 (E0 enter-always decisions + the canonical `EntryModel` base `src/models/common.py` + the freeze harness (in `models/common.py`) + experiment runner: for each (track, model) → model API → `results/decisions_{track}_{model}.parquet` → engine → trades + metrics; the guarded `--split test` path with its **date guard set to `date >= 2026-08-05`**). |
| **PM** | P1.5 (shared k-means/co-membership/stability machinery) built + Track A application via the frozen P1.6 builder: **`labels_a.parquet`, `stability_a.parquet`, `pairs_a.csv`**. P1.4 sanity checks PASS (scree, PC1 loadings ≥ 0, PC1-vs-SPY, variance-over-time incl. 2020 spike; figure polish deferred to Day 4). P1.7 (shared spread + z-score module, simple spread, run/burn-in policy, Z_WINDOW=60) run for Track A: **`spreads_a.parquet`, `zscores_a.parquet`** by EOD. | P2.7 (Track B cleaning: sentiment merge, log size, z-score, drop-sparse + median-impute; characteristics PCA; k-means on component scores via P1.5) → P1.6's pair-builder: **`labels_b.parquet`, `stability_b.parquet`, `pairs_b.csv`** by EOD. P2.4 (the 7 features, with the look-ahead unit-test pattern: shifting future data must not change features) **done on fixtures**. | P3.4 (metrics/bootstrap/calibration module v1: AUC, precision/recall at τ, seeded bootstrap), exercised end-to-end with the runner and `EntryModel` base: **runner + metrics + API round-trip all working on fixture triggers** (fixture E0/dummy-subclass decisions → valid metrics JSONs). |
| Hours | P1: 9.5 | P2: 9 (7.5 on fallback) | P3: 9 |

Cross-slice rider: **`zscores_b`** (P1 runs P1.7 on P2's `pairs_b`) lands EOD or first thing Day 3 — it is the first input P2 needs Day 3 AM.

#### Day 3 — Tuesday, August 4 — **integration day: checkpoint at midday, VALIDATION GATE at the evening sync**

| | P1 — Substrate | P2 — Training data | P3 — Evaluator |
|---|---|---|---|
| **AM** | **`spreads_b.parquet`, `zscores_b.parquet`** if not landed last night. P1.8 (E2 GDA, numpy, verified against sklearn LDA) finished on fixture triggers. P1.10 (leakage-audit items 1–4: window-slice tests, standardization audit, beta-timing audit) begun. | P2.5 (dataset assembly: triggers + features + splits) on **real** `zscores_a` and `zscores_b`: **`triggers_a.parquet`, `triggers_b.parquet`** with real labels, features, split column. P2.9 base-rate analysis computed for the checkpoint review. | P3.5 (Track C: partial-correlation module + recluster wiring, consuming `corr_windows.npz` — on disk since Day 1): **`labels_c`, `stability_c`, `pairs_c.csv` at the midday checkpoint**, so the de-scope decision judges a nearly-done artifact. Also the designated reader of P2's code (6.4, R7), on call for assembly debugging. |
| **Midday** | **Integration checkpoint, all three, 30 min** (criteria in 6.2): triggers validate, base rate in band, E1 fits end-to-end; **Track C de-scope decision point.** | | |
| **PM** | Runs P1.7 for Track C the moment `pairs_c` lands (early PM): **`spreads_c.parquet`, `zscores_c.parquet`**. Fits + tunes E2 on train/val, selects τ_E2 by the pre-registered rule. **Manual trace (P1.9)** — three dates hand-traced through the ledger by a non-author of the engine (that is why this task moved to P1): decision from day-*t* signal, entry at the *t+1* close, first P&L day *t+2*, exit fill at the exit-signal close. | Runs the dataset builder for Track C, one command: **`triggers_c.parquet`**. Fits + tunes E1 (P2.8) on train/val, selects τ_E1. Base-rate-by-year decay figure (F5) drafted. Leakage-audit items 5–6, 10 (P2.12). | Track C hand-offs close early PM (`zscores_c` from P1 → `triggers_c` from P2, ~2–3pm) + the committed Jaccard overlap number (P3.5.4). Fits + tunes E3 (P3.6), selects τ_E3; E0 needs no tuning. Drives the **full 3×4 grid on train+validation**: **`metrics_{a,b,c}_{e0,e1,e2,e3}.json` (train+val), decisions + trades ledgers**. **Noise test (P3.7)** on `make_synthetic` — covers all four models at once, binding criteria per Section 5. Smell test applied to every cell **as a team** over the grid outputs. |
| **Evening sync (9pm)** | **THE VALIDATION GATE** (see 6.2): grid + smell + noise + trace all green, or Day 4 AM burns down the reds with the cut ladder. All τ selections recorded in `DECISIONS.md` tonight. | | |
| Hours | P1: 7.5 | P2: 7 | P3: 8.75 |

#### Day 4 — Wednesday, August 5 — **witnessed test run (~noon); RESULTS FREEZE (EOD)**

| | P1 — Substrate | P2 — Training data | P3 — Evaluator |
|---|---|---|---|
| **AM** | Burn down any red gate items (cut ladder order, 6.5). Then: **each person freezes their own model + scaler + τ via P3.10** (machinery pre-built Day 2 in P3.3; serialized to `results/frozen/`, sha256 hashes in `DECISIONS.md`); **`results/FREEZE.md`** written listing the frozen set. P1.10 leakage items closed with written evidence. | Same: fix reds, freeze E1. Leakage item 10 closed. | P3.10 freeze execution first thing + `FREEZE.md`; **P3.9 turnover-control machinery green pre-noon** (so P1/P2 call it right after the run, no serialization); fix reds. Leakage items 7–9, 11 (P3.11) close in the PM. Verifies the guarded runner refuses to run without `FREEZE.md`. |
| **~Noon** | **The single witnessed test run** (protocol in 6.2): all three present, frozen artifacts only, guarded runner path. **`metrics_*_*.json` (test), `trades_*` (test).** | | |
| **PM** | Turnover-matched control for E2 (own model, P3.9 machinery). Stability figure F4 (A vs B vs C) + F1–F3 (incl. `pc1_loadings`, folded into F1's script) finalized. Formats **T4** (leakage checklist) from the three audit-note sets; each item signed by its verifying owner, incl. the line "Item 12: N/A — Track D not run; cited per spec 8.4." | Turnover control for E1. Consensus-lite (P2.11, ~2h: pair-quarter intersection buckets + reversion-rate table across the three pair lists; figure polish only if time). F5 final, T1, T3 (E1 coefficients + cross-track). | Turnover controls for E3 (E0 needs none — the identity case is the machinery's built-in correctness test). Cost sweep (P3.8) with breakevens. T2 (3×4 grid table with bootstrap CIs + marginal means). Assembles F6 calibration diagrams from all model owners. |
| **Evening sync (9pm)** | **RESULTS FREEZE** (see 6.2): signed git tag `results-freeze`. Aug 6–7 write-up is planned at this sync — as a separate exercise, outside this document. | | |
| Hours | P1: 7.5 | P2: 8.75 | P3: 8 |

**Hour totals (Days 1–4):** P1 = 33h, P2 = 34h, P3 = 34h — inside the ~33–34h envelope at ~8–9h/day; the slice budget tables in Sections 3–5 are authoritative and these rows are generated from them. There is no slack row; the buffer is Day 4 AM plus the cut ladder.

### 6.2 Syncs, checkpoints, and gates

**Two syncs every day.** Midday (~1pm, 15 min): landings on track yes/no, blockers only, no discussion that isn't a blocker. Evening (9:00pm EST, 30 min): the fixed 4-item agenda, same order every night, no other business until these are done —

1. **Artifact status vs plan** — each person states which contracted artifacts landed vs the tables in 6.1, by name.
2. **Blockers** — anything preventing tomorrow's committed landings.
3. **Contract-change requests** — any proposed change to `src/config.py` or an artifact schema requires all-3 agreement *at this meeting* and a line in `DECISIONS.md` (date, change, reason, who agreed). No silent contract changes, ever.
4. **Next-day commitments** — each person names tomorrow's landings out loud.

Merges to `main` happen at (or immediately after) the evening sync: feature branches reviewed by one other person, merged with tests green. Between syncs, `main` always passes the fixtures-based suite.

#### Day 1 evening sync — Bloomberg decision (one-way)

P2 reports whether Monday-morning terminal access is confirmed (booked slot, working entitlements). **Confirmed** → P2 pulls Day 2 AM (P2.6), with the mid-session abort: no committed CSVs by Monday ~noon → fallback immediately. **Not confirmed** → Track B builds on the fallback columns (sector one-hots + price-derived fields from our own price data — free-source historical fundamentals are not point-in-time reliable and are excluded) starting Day 2 AM, disclosed in limitations. There is no "wait for Tuesday" option — Day 3 is integration day and `pairs_b` must exist Day 2 EOD either way. The decision is one-way and logged in `DECISIONS.md`.

#### Day 3 midday — Integration checkpoint (all three, 30 min)

- **Entry criteria:** `zscores_a` and `zscores_b` landed (P1); P2's builder golden tests green.
- **Exit criteria:** (i) real **`triggers_a.parquet` and `triggers_b.parquet`** exist and validate against the contract schema; (ii) the base rate has been computed and reviewed *together* and falls inside the **15–85%** band (outside it, the P2.9 escalation protocol is invoked before proceeding: verify label code against the golden file first — an extreme rate is a suspected bug — then `class_weight='balanced'`, AUC primary, threshold/horizon adjustment on validation evidence only); (iii) E1 trains end-to-end on the train split and emits a valid decisions table.
- **Track C de-scope decision point:** if `pairs_c` is not essentially done here, Track C ships on validation only (or drops entirely, logged) — it never blocks the Day 4 test run. This is the single de-scope rule Track C retains; it is otherwise committed, not gated.
- **If exit criteria are not met by ~2pm:** P1 pauses new work and pairs with P2/P3 on the triggers pipeline; Track C auto-drops to validation-only; the PM plan compresses via the cut ladder.

#### Day 3 evening sync — THE VALIDATION GATE

Passes only if **all** of the following hold, agreed by all three; any "probably fine" is a red:

- All **12** `results/metrics_{a,b,c}_{e0,e1,e2,e3}.json` files exist for train+validation.
- The spec 10e **smell test** applied to *every* cell, together: AUC expected in the 0.52–0.58 band (materially above ⇒ treat as a bug report, not a result); Sharpe above 2 ⇒ suspect a bug; above 3 ⇒ almost certainly leakage. Any smelly cell gets a named investigator.
- The **noise test** passed on its binding criteria (net-of-cost ≤ 0 within CI on noise; no model AUC significantly above the mechanical `f_abs_z`-only baseline — raw AUC-vs-0.5 is advisory, see Section 5 P3.7). Coverage is all four models at once, since E3 exists by Day 3.
- The **manual trace** passed, 3/3 dates MATCH (P1.9 — hand-verified by a non-author of the engine).
- The leakage checklist walked through as a team, each item marked with its evidence status (formal sign-off completes Day 4 AM).

**Any red item = Day 4 AM burn-down using the cut ladder (6.5), in order, decided at this sync and logged.** The gate does not slip to Day 4 PM: whatever is still red at Day 4 ~noon is cut, not deferred.

#### Day 4 ~noon — the single witnessed test run

- Happens **exactly once**, all three present on a screen-share, from the frozen models + scalers + τ in `results/frozen/` (each person froze their own model via P3.10; hashes in `DECISIONS.md`; `results/FREEZE.md` lists the frozen set), via the runner's guarded path (`experiments.py grid --split test` requires `--i-am-sure`, checks its date guard `date >= 2026-08-05`, refuses to run without `FREEZE.md`, and refuses to run twice — it checks for existing test outputs).
- **The results are what they are.** Spec, quoted: *"You are not being marked on how good the results are."* Nobody proposes a "quick re-check" after seeing test numbers.
- If a genuine bug is discovered after the run, the fix and the **single** rerun are documented in the limitations material, with both the before and after noted. Honesty over cosmetics.

#### Day 4 EOD sync — RESULTS FREEZE

**Frozen at this sync:** all `results/metrics_*.json` (train, val, test), all `results/trades_*.parquet` ledgers, all figures/tables in `results/figures/final/` with their source scripts (`scripts/figures/`), `results/FREEZE.md`, **T4** (the leakage checklist, assembled by P1 from the three slices' audit notes, each item signed by its verifying owner), and `DECISIONS.md`. The freeze is a signed git tag (`results-freeze`).

**Post-freeze rule:** any code change that could alter a frozen number requires (i) a documented, reproducible bug written into `DECISIONS.md`, (ii) sign-off from all three, and (iii) a rerun of the noise test before the corrected numbers replace the frozen ones. Cosmetic edits are unrestricted; number-changing edits are not.

**The write-up (Aug 6–7) is deliberately out of scope for this plan**, per the team's decision — it is planned at the Day 4 EOD sync as a separate exercise. Nothing in Days 1–4 schedules prose. The figures and tables below are evaluation outputs, produced Days 3–4 and regenerable from frozen artifacts; who writes which report section is decided in the write-up phase, not here.

### 6.3 Figure/table inventory

One script per figure in `scripts/figures/`, regenerable from frozen artifacts. **Inventory only** — report-section ownership is a write-up-phase decision.

| ID | Description | Source script | Owner | Due |
|---|---|---|---|---|
| F1 | Scree plot (representative window) | `fig_scree.py` | P1 | Day 4 PM (sanity draft Day 2) |
| F2 | PC1 factor return vs SPY overlay | `fig_pc1_spy.py` | P1 | Day 4 PM (sanity draft Day 2) |
| F3 | Top-3 cumulative variance over time (2020 spike) | `fig_var_time.py` | P1 | Day 4 PM |
| F4 | Cluster co-membership stability over time, Track A vs B vs C | `fig_stability.py` | P1 | Day 4 PM |
| T1 | Characteristics-PC interpretation table ("expensiveness" etc.) | `tab_char_pcs.py` | P2 | Day 4 PM (drafted Day 2) |
| F5 | Base rate by year 2016–2024 (decay check; triggers start ~2016 after warmups) | `fig_base_rate.py` | P2 | Day 4 PM (draft Day 3) |
| F6 | Calibration / reliability diagrams, per model (val + test) | `fig_calibration.py` | Model owners; assembled by P3 | Day 4 PM |
| F7 | Cost-sensitivity sweep, one line per strategy, breakevens annotated | `fig_cost_sweep.py` | P3 | Day 4 PM |
| T2 | **3×4** grid results table with bootstrap CIs + row/column marginal means (answers spec 12a) | `tab_grid.py` | P3 | Day 4 PM |
| F8 | Turnover-matched control histograms (1000 seeded draws), one per filtered model | `fig_turnover_ctrl.py` | Machinery P3; each model's owner runs their own | Day 4 PM |
| F9 | Consensus-lite table (pair-quarter buckets × reversion rate across the three pair lists; figure polish only if time) | `fig_consensus.py` | P2 | Day 4 PM |
| T3 | E1 coefficient table + cross-track comparison | `tab_coefs.py` | P2 | Day 4 PM |
| T4 | Leakage checklist, all items with evidence | manual | P1 formats; each item signed by its verifying owner | Day 4 freeze |

#### Banked inputs for the write-up phase (Aug 6–7 — outside this plan, parked here so nothing is lost)

The following spec deliverables are *prose*, deliberately unscheduled in Days 1–4; each has its data source ready at the freeze: the **12a three-questions reading** (from T2's marginal means); the **12d factor-neutralization paragraph** engaging Han et al. (2023) (from T2's row effect — either answer is a real finding per the spec); the **honest-novelty correction** citing Ekinci et al. (2025) and ICBDEIM (2025); the **Track D one-sentence citation** per spec 8.4 (Krause & Calliess, 2024) and Rotondi & Russo (2024) as Track C's source; the **contribution sentence** (spec Part 5 supplies the template); the **multiple-testing statement** using the 12-cell arithmetic (owner-tagged stub already seeded in `docs/limitations.md`); and the **limitations assembly** from the seeded file. Report-section ownership is decided at the Day 4 EOD freeze sync.

### 6.4 Risk register

| # | Risk | L | Impact | Early-warning signal | Mitigation | Owner |
|---|---|---|---|---|---|---|
| R1 | Bloomberg terminal access fails (Day 1 is a Sunday; the pull is Monday AM) | M | Track B columns thin | Access not confirmed at the **Day 1 evening sync**, or no committed CSVs by Monday ~noon | Decision is made **Sunday night**, one-way: fallback columns (sector one-hots + price-derived fields from own data) build immediately Day 2, disclosed in limitations; Track B row survives with fewer columns. Mid-session abort at Monday ~noon. No waiting for Tuesday. | P2 |
| R2 | The 4-day crunch itself | H | Any slip cascades — there is zero slack | Any evening sync where a committed landing missed | The cut ladder (6.5) is the pressure valve, invoked at syncs, in order, logged; the ~8–9h/day availability commitment is made explicitly at kickoff against this table, not assumed | All |
| R3 | Single integration day (Day 3) slips | M | Grid/gate lost | `triggers_a`/`triggers_b` not building by Day 3 ~noon | Day 4 AM is the only buffer: burn-down absorbs it; Track C auto-drops to validation-only; ladder cuts start; the test run stays at Day 4 ~noon | P1 + P2 |
| R4 | yfinance breakage / rate limits Day 1 | M | **P1's entire Day 1–2 chain (returns → PCA → residuals → pairs_a → zscores_a) hangs on `returns.parquet` landing Day 1** | Download errors or gappy columns Day 1 AM | Raw pulls cached and **committed** on first success (`data/raw/` in git); alternate source: stooq via `pandas-datareader`; fixtures keep P2 and P3 fully unblocked regardless — but P1 escalates at the Day 1 midday sync, not EOD | P1 |
| R5 | Too few triggers / extreme base rate | M | Models can't fit; metrics degenerate | Alarm if train rows < 300 or base rate outside 15–85% at the **Day 3 midday checkpoint** | v1 protocols unchanged: verify label code against the golden file first (extreme rate = suspected bug); widen trigger to 1.75 or extend horizon to 7d — **on validation evidence only, never test**, decided at the checkpoint, logged in `DECISIONS.md`; `class_weight='balanced'`, AUC primary | P2 |
| R6 | Noise test fails Day 3 (binding criteria) | L | Every result untrustworthy | Net-of-cost profit on noise with CI excluding 0, or any model AUC significantly above the mechanical `f_abs_z`-only baseline | **Stop-the-line:** all-hands until root-caused; Track C auto-drops to validation-only; Day 4 AM absorbs the fix; leakage checklist re-walked item by item; test run does not proceed until green | P3 (runs), all (fix) |
| R7 | A member loses a day | M | **In a slice model, each slice stalls alone** — no one else's lane dies with it | Missed sync or missed committed landing | Cross-cover pairs: **P1 covers P3's engine-adjacent steps** (spread module author, manual-trace author) **and P3 covers P1's spread-consuming steps**; **P2's dataset builder is the single point of failure** → mitigated by its golden-tested fixture path (any of the three can run the one-command builder against the golden file) and **P3 is the designated reader of P2's code** from Day 2 | All |
| R8 | Results look too good | M | Fake headline result | Any cell: AUC > 0.60 or Sharpe > 2 (spec 10e) | Smell-test protocol at the Day 3 gate: treat as a bug report until proven otherwise; named investigator; cell excluded until cleared | All |
| R9 | Track C slips | M | 12-cell grid shrinks | P3 behind at the **Day 3 midday checkpoint** | The retained de-scope rule: validation-only, then cut entirely (grid 2×4), logged; **Track C never blocks the Day 4 test run** — no other task waits on it | P3 |
| R10 | Model freeze skipped under time pressure | L | Test-touched-once protocol collapses | Anyone proposing to "just run test now and freeze after" | The guarded runner **physically refuses** without `results/FREEZE.md` and frozen hashes — this guard is never weakened, even to save time on Day 4; that is the point of it | P3 (guard), all (norm) |

### 6.5 Scope-cut ladder (v2)

**Already cut, pre-committed (not on the ladder):** Track D (spec-8.4 one-sentence citation in the lit review, nothing else — no gate, no task, no config constants); the error-analysis deep-dive; report drafting inside Days 1–4.

If we fall behind, cuts happen **in this order**, decided at a sync and logged in `DECISIONS.md`. Cut early and cleanly rather than late and messily; per spec 8.4, cut items become one honest sentence in the report.

1. **Consensus-lite** (P2.11) — already compressed to ~2h; first overboard.
2. **Track C** — first to validation-only, then cut entirely (grid becomes 2×4); per its de-scope rule this never blocks the test run.
3. **F3/F4 figures** (variance-over-time, stability comparison) — the underlying numbers stay in the artifacts.
4. **E2/E3 calibration figures** — keep F6 for E1 (the pre-registered primary model).
5. **Bootstrap draws 1000 → 250** — CIs get slightly rougher; still reported.
6. **Cost-sweep figure** (F7) — keep the breakeven table; drop only the plot.

**The never-cut list (spec Part 8.0, restated):** **purged CV (with embargo), the transaction-cost model, the turnover-matched control (at minimum for Track A × E1), the honest limitations content (the Day-1-seeded `docs/limitations.md`), the noise test, and the test-touched-once protocol.** These survive even if the grid shrinks to Track A × {E0, E1} — because that minimal grid, evaluated honestly, is still the pre-registered primary comparison and a complete, gradeable project. The multiple-testing disclosure is written for the grid actually run: at full scope that is **12 cells**, and the false-positive arithmetic in the limitations material uses 12.

---

## 7. Appendix: tooling notes and practical gotchas

Everything in this appendix was checked against live package indexes and changelogs on **2026-07-31 (the day before the v2 kickoff)**. Pins marked **verify locally** could not be fully confirmed from documentation — run the one-line check given before trusting them. Nothing here changes any contract or config; it exists so nobody burns an evening on API drift.

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

Verified current: **pandas 3.0.5** (3.0.0 landed 2026-01-21), **pyarrow 25.0.0**. numpy is deliberately held at **1.26.4** — not the current 2.5.x — because the team-wide `torch==2.2.2` decision (see 7.5) predates NumPy 2; all other pins verified compatible with 1.26.

- [ ] **Use Python 3.12** (see 7.9) — the newest interpreter with torch 2.2.2 (cp312) wheels on every team platform incl. Intel mac.
- [ ] **pandas 3.0 behavior changes to be aware of** (we pin 3.0.5, not 2.x, because yfinance/sklearn are tested against it by now): Copy-on-Write is always on — **chained assignment silently does nothing**; always write `df.loc[mask, col] = ...`. The default string dtype is the dedicated `str` dtype, not `object` — comparisons with ticker strings are unaffected, but `df.columns.dtype` checks should use `df.columns.map(type)` sparingly and just trust label equality.
- [ ] **Parquet round-trip rules** (bugs here corrupt the interface contracts, so they're worth 10 minutes of tests on Day 1):
  - A `DatetimeIndex` **is** preserved through `to_parquet`/`read_parquet` with the pyarrow engine, including its name. Give every date index the name `"date"` before writing and assert it after reading.
  - **All column names must be `str`.** `pyarrow` refuses non-string column labels. Danger spots: `pd.DataFrame(np.ndarray)` gives integer columns (the PCA loadings path), and any `groupby(...).unstack()` can produce tuple columns. Blanket fix in every writer: `df.columns = df.columns.map(str)`.
  - Timezone-aware indexes round-trip but cause merge misery. Normalize once at ingest: `prices.index = prices.index.tz_localize(None)` (yfinance sometimes returns tz-aware).
  - `bool` columns (`co_clustered`, `enter`) round-trip cleanly; do not store them as 0/1 ints or the schema check in `tests/test_contracts.py` gets ambiguous.
- [ ] One shared helper pair, used by every producer, ends all debate: `contracts.py`'s `write_parquet(df: pd.DataFrame, path: Path) -> None` (asserts str columns + named DatetimeIndex where applicable) and `read_parquet(path: Path) -> pd.DataFrame`. ~1 h, P3, Day 1, alongside the repo skeleton.

### 7.3 scikit-learn

Verified current: **scikit-learn 1.9.0** (2026-06-02; supports Python 3.11–3.14; new hard dependency on `narwhals`).

- [ ] **Pin `scikit-learn==1.9.0`.**
- [ ] **KMeans `n_init`:** the default has been `"auto"` (= 1 for k-means++) since 1.4. `"auto"` with k-means++ does a single init — *not* what the config freeze says. Always pass explicitly, exactly as frozen:

```python
KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=311)
```

- [ ] **Silhouette:** `silhouette_score(X, labels)` where `X` is the *same* formation-window feature matrix given to `fit` (betas for Track A, PC scores for Track B). For Track C's precomputed distance matrix use `silhouette_score(D, labels, metric="precomputed")`. Guard: silhouette is undefined for a clustering with any single distinct label — wrap in a check before comparing across the k range.
- [ ] **LogisticRegression for E1** on 7 dense features / a few hundred rows: `LogisticRegression(penalty="l2", C=C, class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=311)`. `lbfgs` is correct for tiny dense data; never `saga` (needs scaling-sensitive tuning) or `liblinear` (different regularization path). `max_iter=2000` (per P2.8) preempts the convergence warning that otherwise pollutes every log. The C-grid is P2's section; nothing here constrains it.
- [ ] **GDA (E2):** per Section 3 (P1.8), E2 is implemented as ~30 lines of numpy GDA for course alignment; `LinearDiscriminantAnalysis.predict_proba` is the 1e-6 verification reference, and `QuadraticDiscriminantAnalysis` (per-class covariance; `reg_param≈1e-3` if covariances are near-singular — 7 features is small enough that this mostly won't bite) supplies the robustness row.
- [ ] **StandardScaler discipline:** exactly one `scaler.fit(X_train)`; then `transform` on val and test. The fitted scaler is part of the frozen model bundle each model owner hands to the Day 4 test run — persist it with the model, never re-fit downstream. A `Pipeline([("scaler", StandardScaler()), ("clf", ...)])` makes this impossible to get wrong and is the recommended shape.

### 7.4 Rolling OLS: numpy, not statsmodels

The Step 3 betas require, per trading day (~2,300 days), a regression of a 252×40 return block on a 252×m factor block. That is **one multi-target least-squares solve per day**, not 40 separate regressions:

```python
def window_betas(F: np.ndarray, R: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """F: (252, m+1) design matrix — a ones column FIRST (RESIDUAL_INCLUDE_ALPHA=True,
    per P1.3), then the m in-window factor returns reconstructed from THIS window's
    eigenvectors (P1.3's construction — not sliced from the stored factors_a series).
    R: (252, 40) stock returns. Returns (m+1, 40): alphas in row 0, betas below."""
    A = F.T @ F + ridge * np.eye(F.shape[1])
    return np.linalg.solve(A, F.T @ R)
```

- [ ] Total cost: ~2,300 solves of an m×m system (m ≤ 5) — **well under one second for the whole history**. Do not reach for `statsmodels.RollingOLS`, loops over stocks, or anything clever.
- [ ] **Conditioning:** eigenportfolio factors are near-orthogonal by construction, so `A` is well-conditioned; but keep the `ridge` argument (default 0, flip to `1e-8` if `np.linalg.LinAlgError` or wild betas appear) and log whenever it is non-zero. `np.linalg.lstsq(F, R, rcond=None)` is the drop-in alternative if you prefer not to form `F.T @ F`.
- [ ] **statsmodels (`==0.14.6` — latest stable line; check `pip index versions statsmodels` on Day 1, **verify locally** that 0.14.6 imports cleanly against pandas 3.0)** is used in exactly one place: if the report wants a pretty OLS summary table (e.g., a single illustrative beta regression). It never sits in the pipeline hot path.

### 7.5 PyTorch (CPU only)

Team-wide pin: **torch 2.2.2** (the last release with Intel-mac builds — see the single-version decision below). We need CPU wheels only.

- [ ] torch installs **with the env**, ONE version team-wide: **`torch==2.2.2`** — chosen because it is the last release with Intel-mac builds (one teammate's hard ceiling), so every machine runs the identical release. Platform markers in the yml give linux/windows the slim `+cpu` build from `--extra-index-url https://download.pytorch.org/whl/cpu` (the `+cpu` pin is only satisfiable there, so the CUDA-bundled wheel is unreachable) and macs the PyPI wheel (CPU-only by nature). **Consequence, accepted knowingly:** torch 2.2.x predates NumPy 2 and breaks with it, so numpy is pinned `1.26.4` — every other pin was verified to accept numpy>=1.26 (pandas 3.0.5 requires exactly `>=1.26.0`; scipy 1.16 caps at `<2.6`). E3 uses only Linear/ReLU/BCE/Adam — ancient-stable APIs, nothing 2.2 lacks. A teammate with an existing CUDA torch (lab machine, other coursework) may keep it — E3 constructs its model and tensors on `config.TORCH_DEVICE = "cpu"`, never `cuda.is_available()` autodetection, so the installed wheel cannot change any number.

- [ ] **Determinism recipe** (put in `src/contracts.py (seed_everything)`, called by every entry point):

```python
def seed_everything(seed: int = 311) -> None:
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
```

  Caveats: `use_deterministic_algorithms(True)` can raise on ops without deterministic CPU kernels — nothing in a tiny MLP/AE (Linear, ReLU/tanh, MSE, BCE, Adam) triggers this, but if it ever raises, downgrade that one call to `warn_only=True` and note it in DECISIONS.md. CUDA-specific env vars (`CUBLAS_WORKSPACE_CONFIG`) are irrelevant on CPU. DataLoader workers: use `num_workers=0` — data this small needs no workers and it removes a whole seeding category.
- [ ] **Compute reality check, so nobody budgets time for "training":** E3 is 7 features → 8–16 hidden units → 1 output on a few hundred rows: full-batch or minibatch, a few hundred epochs with early stopping = **~1–5 seconds per fit**; a 20-point hyperparameter sweep is under two minutes. (Track D was cut in v2, so E3 is the only torch training in the project.) Compute is never the bottleneck.

### 7.6 Bloomberg terminal export (Track B, one session, Day 2)

Realistic options for a student with one terminal session, ranked:

1. **Excel Add-in (`BDH`/`BDP`)** — the right tool. Zero setup on a terminal PC, and the output is already tabular.
2. Terminal `EXCEL` template builder / drag-and-drop export — fine as a fallback for individual fields.
3. **`blpapi` (Python API)** — overkill for one session: requires a Desktop API entitlement check, a local install, and debugging time we don't have. Skip.

The recipe:

- [ ] **Before the session**, P2 lists the 19 target fields from spec 4B.1 and, at the terminal, runs **`FLDS`** on one ticker to verify each mnemonic actually exists and returns quarterly history (e.g., `PE_RATIO`, `PX_TO_BOOK_RATIO`, `CUR_MKT_CAP`, `TOT_ANALYST_REC`, ... — **verify mnemonics at the terminal**; do not trust any list found online, field names drift).
- [ ] **One workbook, one sheet per field.** Each sheet: tickers across columns, one `BDH` array per ticker:
  `=BDH("AAPL US Equity","PE_RATIO","12/31/2014","12/31/2024","Per=Q","Days=A","Fill=P")`
  (`Per=Q` quarterly; `Fill=P` carries the previous value across empty periods). 40 tickers × ~40 quarters per sheet is far below any practical `BDH` limit; the daily data-usage cap is generous enough for ~30k cells but **do the pull once, not iteratively**.
- [ ] Let all sheets finish calculating (watch the status bar — `BDH` fills asynchronously), then **File → Save As → CSV, one file per sheet**, into `data/raw/bloomberg/{field}.csv`. Commit the CSVs immediately; the terminal session is the only unrepeatable step in the whole project.
- [ ] **Time-box: 2 hours at the terminal.** Any field still fighting back at the deadline (wrong periodicity, entitlement error, mnemonic not found) gets dropped on the spot and the P2.6/P2.7 fallback rule (spec 4B.2: drop sparse columns, median-impute) absorbs it. 12 good fields beat 19 fields and a second terminal trip.
- [ ] Consumes: ticker list from `data/raw/prices.parquet` columns. Produces: `data/raw/bloomberg/*.csv`. Done when: every CSV loads in pandas with a parseable quarterly date column and ≥ 80% non-empty cells.

### 7.7 Matplotlib on WSL2

- [ ] WSL2 has no reliable display; pipeline code must **never call `plt.show()`**. Force the file-only backend once, project-wide: `MPLBACKEND=Agg` exported in the Makefile (belt) and `matplotlib.use("Agg")` at the top of `scripts/figures/plotstyle.py` before any `pyplot` import (suspenders).
- [ ] All figure code goes `fig, ax = plt.subplots(...)` → `fig.savefig(path, dpi=200, bbox_inches="tight")` → `plt.close(fig)` (the `close` matters in loops — Agg leaks figures otherwise).
- [ ] One consistent style, one place — `scripts/figures/plotstyle.py`, imported by every figure script:

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

  P1 creates it Day 2 with the first sanity-check figures; done when every figure in `results/figures/` visibly shares the style.

### 7.8 pytest and the golden-file pattern

Verified current: **pytest 9.1.1**; pin `pytest==9.1.1`.

- [ ] **Goldens are CSVs, not parquet**, stored under `tests/golden/` — they must be human-diffable in a PR (`fixture_zscores.py`'s expected-triggers file is the canonical example). Pattern: compute → `pd.testing.assert_frame_equal(result.reset_index(), pd.read_csv(golden, parse_dates=["trigger_date"]), check_exact=False, atol=1e-9)`. Regenerating a golden requires a reviewer — treat golden diffs like contract changes.
- [ ] **Future-perturbation tests** (the "poke tomorrow, assert today unchanged" leakage tests) must run on a **300-day slice** of the synthetic data, not the full history — a full rolling-PCA pass per test makes the suite minutes-slow and people stop running it. Target: whole suite `< 30 s` so `pytest -q` runs before every push. Mark anything slower `@pytest.mark.slow` and exclude it by default via `addopts = -m "not slow"` in `pyproject.toml`.
- [ ] Seed inside every test via `seed_everything(311)` (7.5) — never rely on import-order side effects.

### 7.9 environment.yml (conda)

Recommended interpreter: **Python 3.12**, provisioned by conda on WSL2 Ubuntu (Miniforge/Miniconda/Anaconda all work; WSL2 needs no special handling). Rationale for 3.12: numpy 2.5 dropped 3.11; 3.13/3.14 are legal for sklearn but the wider ecosystem (yfinance's transitive deps, curl_cffi wheels) is best-tested on 3.12. The pins stay in the **pip section** (not conda packages) because they were verified against PyPI and yfinance/curl_cffi are pip-native — one resolver, no channel drift.

```yaml
# environment.yml — commit at repo init; frozen after Day 1 like the config
name: pair-trading
channels:
  - conda-forge
  - nodefaults
dependencies:
  - python=3.12
  - pip
  - pip:
      # The extra index ADDS PyTorch's CPU wheel host alongside PyPI.
      - --extra-index-url https://download.pytorch.org/whl/cpu
      # --- torch: ONE version team-wide. 2.2.2 is the last release with Intel-mac
      #     builds (Jaskaran's machine is the ceiling). Same release everywhere;
      #     linux/windows get the slim +cpu build, macs get the PyPI wheel, which
      #     is CPU-only by nature. Pipeline computes on config.TORCH_DEVICE="cpu".
      - torch==2.2.2+cpu ; sys_platform == "linux"
      - torch==2.2.2 ; sys_platform == "darwin"
      - torch==2.2.2+cpu ; sys_platform == "win32"
      # --- core numerics (Py 3.12) ---
      - numpy==1.26.4       # deliberately NOT 2.x: torch 2.2.x predates numpy 2 and
      #                       breaks with it; every pin below accepts >=1.26 (verified 2026-08-03)
      - pandas==3.0.5       # unchanged — requires numpy>=1.26, satisfied; CoW + str dtype (see 7.2)
      - pyarrow==25.0.0     # parquet engine for every artifact contract
      - scipy==1.16.0       # accepts numpy>=1.25.2,<2.6 (verified)
      # --- data ---
      - yfinance==1.5.2     # scraper: pin hard, never float (see 7.1)
      - curl_cffi==0.16.0   # required by yfinance>=1.5 (>=0.15); dodges Yahoo blocking
      # --- ML ---
      - scikit-learn==1.9.0 # KMeans/LogReg/LDA-QDA/metrics; accepts numpy>=1.24
      - statsmodels==0.14.6 # report-facing OLS summaries only; accepts numpy>=1.22
      # --- plotting / testing ---
      - matplotlib==3.10.3  # Agg-only usage; accepts numpy>=1.23
      - pytest==9.1.1       # golden-file + leakage test suite

# One command does everything: conda env create -f environment.yml
# Already created it once? conda env update -f environment.yml --prune
# (downgrades numpy in place). A CUDA torch elsewhere is fine for other work —
# the pipeline pins computation to config.TORCH_DEVICE = "cpu".
```

- [ ] Day 1 kickoff includes one command by each person: `conda env create -f environment.yml && conda activate pair-trading`, then `pytest -q` — torch rides along via the `+cpu` pin, no separate step. If pip's resolver rejects any pin (`scipy`/`matplotlib` remain flagged **verify locally**; `curl_cffi` was verified 2026-08-03 — 0.13.0 conflicted with yfinance 1.5.2's >=0.15 requirement and is pinned 0.16.0), the person who hits it fixes the pin, commits, and posts in the channel. The file is frozen after Day 1 like the config.
- [ ] `pip freeze > requirements.lock.txt` (run inside the activated env) after the first successful install, committed — that lockfile, not `environment.yml`, is what the report's reproducibility statement cites.

### 7.10 Repro discipline: Makefile + RUNBOOK

Principle: **after Day 1, no target ever contacts Yahoo or Bloomberg.** The committed raw pulls (`data/raw/*.parquet`, `data/raw/bloomberg/*.csv`) are the roots of the DAG; everything downstream is a deterministic function of them plus `src/config.py` and seed 311.

```make
.PHONY: all data tracka trackb trackc dataset grid noise-test test-run figures test

data:            ## returns.parquet + both fixtures from committed raw pulls (NO network)
	python -m src.data --from-cache
	python -m src.make_synthetic
	python -m src.fixture_zscores
tracka: data     ## Steps 2-6, track a: factors -> clusters -> pairs -> z-scores, one module
	python -m src.representation --track a
trackb: data     ## Track B approach file, then the shared representation machinery
	python -m src.characteristics
	python -m src.representation --track b
trackc: tracka   ## Track C approach file (reads corr_windows.npz), then the shared machinery
	python -m src.partial_corr
	python -m src.representation --track c
dataset: tracka trackb trackc
	python -m src.dataset --tracks a,b,c
grid: dataset    ## E0/E1/E2/E3 x tracks on train+val only (tracks read from config so the Makefile cannot drift from the committed grid)
	python -m src.experiments grid --tracks a,b,c --models e0,e1,e2,e3 --split trainval
noise-test:      ## full pipeline on src/make_synthetic.py output; PASS/FAIL per Part 4
	python -m src.experiments noise
test-run:        ## Day 4 ONLY, witnessed; guarded (--i-am-sure + FREEZE.md + date >= Aug 5 + no results/final/TEST_RUN_COMPLETE marker)
	python -m src.experiments grid --tracks a,b,c --models e0,e1,e2,e3 --split test --i-am-sure
figures: grid
	python -m scripts.figures.make_all
test:
	pytest -q
all: data tracka trackb trackc dataset grid figures
```

- [ ] The runner itself (not the Makefile) writes `results/final/TEST_RUN_COMPLETE` (timestamp + git sha) as the last act of a successful test run, and refuses to start if it exists — mechanical enforcement of "test touched exactly once," durable because `results/final/` is the committed path (Section 2.2).
- [ ] **`RUNBOOK.md`** (repo root, one page, P3 owns, done Day 2 and re-verified Day 4): fresh-clone-to-full-grid in order —

```text
1. git clone <repo> && cd pair_trading
2. conda env create -f environment.yml
3. conda activate pair-trading
4. pytest -q                      # all green before anything else
5. make all                       # raw parquet -> full train+val grid, ~minutes
6. make noise-test                # must find nothing
7. make test-run                  # Day 4, once, all three watching
```

- [ ] Definition of done: a teammate who has never touched the repo runs steps 1–6 on a clean WSL2 machine on Day 3 and gets bit-identical `results/metrics_*` (same seed, same lockfile). That dry-run *is* the reproducibility claim in the report.

Version sources checked 2026-07-31: [yfinance PyPI](https://pypi.org/project/yfinance/) / [changelog](https://github.com/ranaroussi/yfinance/blob/main/CHANGELOG.rst), [pandas 3.0 release notes](https://pandas.pydata.org/docs/whatsnew/v3.0.0.html), [NumPy news](https://numpy.org/news/), [scikit-learn release history](https://scikit-learn.org/stable/whats_new.html), [PyTorch releases](https://github.com/pytorch/pytorch/releases), [pyarrow PyPI](https://pypi.org/project/pyarrow/), [statsmodels releases](https://github.com/statsmodels/statsmodels/releases), [pytest changelog](https://docs.pytest.org/en/stable/changelog.html).

---

## 8. Appendix: spec → plan traceability matrix

Every requirement in `project_spec_v2.md`, mapped to its v2 location (slice section + task), owner, and scheduled day. Cut items still appear, pointing at their disposition (Track D → the spec-8.4 lit-review sentence; the error-analysis deep-dive → cut; prose-only items → the write-up phase per §6.3's banked-inputs list). Use it at the Day 3 and Day 4 syncs as the completeness checklist: any row whose artifact does not exist by its listed day is a named, visible gap — not a silent one.

| Spec item | Plan location | Owner | Day |
|---|---|---|---|
| **Part 3 / Step 0** — 40 stocks, 4 sectors × 10 | §3 P1.1 universe module (proposed 40-ticker table) + §2.8 kickoff ratification | P1 | 1 |
| Step 0 — yfinance download, `auto_adjust=True` confirmed | §3 P1.1 `download_prices` (asserted explicitly) + §7.1 | P1 | 1 |
| Step 0 — drop ticker >2% missing days | §3 P1.1 cleaning check 1 (+ same-sector replacement protocol) | P1 | 1 |
| Step 0 — forward-fill isolated single-day gaps | §3 P1.1 cleaning check 2 | P1 | 1 |
| Step 0 — align to common trading calendar | §3 P1.1 cleaning check 3 | P1 | 1 |
| Step 0 — verify no ±50% return without real event | §3 P1.1 cleaning check 4 (manual-review table) | P1 | 1 |
| Step 0 — verify ~252 rows/year | §3 P1.1 cleaning check 5 | P1 | 1 |
| Step 0 — survivorship-bias acknowledgment | §3 P1.1 disclosure note + §2.6 `docs/limitations.md` Day-1 seed (Part 7 bullets) | P1 | 1, write-up phase |
| **Step 1** — returns, choice stated (log vs simple) | §2.3 `LOG_RETURNS` + §3 P1.1 `compute_returns` | P1 | 1 |
| **Step 2a** — rolling 252d window excluding day t | §3 P1.2 note 1 (in-loop assertion) | P1 | 1 |
| Step 2b — window-local standardization | §3 P1.2 note 2 | P1 | 1 |
| Step 2c — correlation (not covariance) matrix | §3 P1.2 note 3 | P1 | 1 |
| Step 2d — `eigh`, descending sort | §3 P1.2 note 4 | P1 | 1 |
| Step 2e — a-priori component rule, never tuned on test | §2.3 `N_COMPONENTS_CANDIDATES`/`VAR_EXPLAINED_TARGET` + §3 P1.2 `choose_n_components` | P1 | 1 (frozen at kickoff) |
| Step 2f — sign-fix every window | §3 P1.2 note 5 + §2.5 sign-fix property test | P1 | 1 |
| Step 2g — eigenportfolios, inverse-vol weights | §3 P1.2 note 6 (Avellaneda & Lee weighting) | P1 | 1 |
| Step 2 sanity checks — scree, PC1 loadings, PC1-vs-SPY, variance-over-time (2020 spike) | §3 P1.4 (4 scripts, PASS/FAIL) + §6.3 F1–F3 | P1 | 2 (figures 4) |
| **Step 3** — OLS betas trailing window, applied OOS | §3 P1.3 (in-window factor reconstruction; assert window excludes t) | P1 | 2 |
| Step 3 — stale-beta limitation disclosed | §2.6 limitations.md Day-1 seed ("beta drift" Part 7 bullet) | P1 | 1, write-up phase |
| **Step 4** — cluster input, primary option stated | §2.3 config comment (factor-beta vectors) + §3 P1.5 clustering-input decision note | P1 | 1 (decision), 2 |
| Step 4 — "try both" second option (252-length residual series) | §3 P1.5 note: committed skip in DECISIONS.md (Zhang p=0.01 vs 0.785); v1's Option-1 robustness run **cut in v2** | P1 | 1 (decision); robustness run CUT |
| Step 4 — k-means++, n_init=10, empty-cluster handling | §3 P1.5 notes (assert unique labels == k) + §7.3 explicit `n_init=10` warning | P1 | 2 |
| Step 4 — choose k by silhouette/elbow, formation window only | §3 P1.5 (max silhouette, k∈8–13, logged to `kmeans_log_a.csv`) | P1 | 2 |
| Step 4 — label-switching → co-membership matrix | §3 P1.5 `comembership()` (never compare raw labels) | P1 | 2 |
| Step 4 — stability analysis (fraction co-clustered w→w+1) | §3 P1.5 `stability_frame` + §6.3 F4 (A vs B vs C) | P1 | 2 (figure 4) |
| **Step 4B.1** — 19 Bloomberg fields, quarterly pull | §4 P2.6 (session plan, FLDS verification, Monday terminal-access contingency) + §7.6 | P2 | 2 |
| 4B.1 — point-in-time warning (as-reported preferred, else disclosed) | §4 P2.6 step 3 + `FIELDS_USED.md` + limitations | P2 | 2, write-up phase |
| 4B.2 — sentiment merge (19→18) | §4 P2.7 `clean_snapshot` step 1 (+ 15/5→+0.5 unit test) | P2 | 2 |
| 4B.2 — log-transform size columns | §4 P2.7 `clean_snapshot` step 2 | P2 | 2 |
| 4B.2 — z-score every column | §4 P2.7 `clean_snapshot` step 3 | P2 | 2 |
| 4B.2 — our addition: drop sparse columns + median-impute, disclosed | §4 P2.7 (<90% coverage drop + zero-variance drop, logged) + limitations | P2 | 2, write-up phase |
| 4B.3 — PCA on characteristics, fewer components (thin-data caveat) | §4 P2.7 `pca_characteristics` (cumvar ≥ 0.60, cap 5) | P2 | 2 |
| 4B.4 — read and name the components | §4 P2.7 `name_components` → `track_b_components.md` + §6.3 T1 | P2 | 2 (table final 4) |
| 4B.5 — Route B (projection) primary | §4 P2.7 `pca_characteristics` (scores = Route B projection) | P2 | 2 |
| 4B.5 — Route A "if time allows" | §4 P2.7 docstring: deliberately skipped citing Zhang; logged in DECISIONS.md | P2 | 2 (decision logged) |
| 4B.6 — k=10–13 + Zhang group rules (skip 1, take 2–4, split 5+) | §4 P2.7 via P1's shared machinery + §2.3 `K_RANGE["b"]` + §3 P1.6 builder | P2 (P1's machinery) | 2 |
| **Step 5** — pairs within clusters, output schema w/ `source` tag | §3 P1.6 shared `build_pairs` + §2.2 pairs contract (frozen signature) | P1 | 1 (stub+algorithm), 2 (wired) |
| **Step 6a** — spread construction (option chosen) | §2.3 `SPREAD_KIND="simple"` + §3 P1.7 (run/burn-in re-anchoring policy) | P1 | 2 |
| Step 6b — rolling z-score, trailing window-local stats | §2.3 `Z_WINDOW=60` + §3 P1.7 `zscore` (min_periods=60) | P1 | 2 |
| Step 6 — OU s-score simplification stated explicitly | §2.6 limitations.md Day-1 seed (Part 7 bullet) | P1 | 1, write-up phase |
| **Step 7a** — onset-only trigger at \|z\| crossing 2.0 | §4 P2.2 (exact onset semantics, run-aware, re-arm policy, fixture-F6 test) | P2 | 1 |
| Step 7b — label (50% reversion within 5d), parameters stated a priori | §2.3 `TRIGGER_Z`/`REVERSION_FRACTION`/`LABEL_HORIZON` + §4 P2.2 | P2 | 1 (frozen at kickoff) |
| Step 7c — base-rate check before trusting anything | §4 P2.9 + pre-agreed decision table + §6.2 Day 3 midday checkpoint (15–85% band) | P2 | 3 |
| Step 7d — overlap property → motivates purging | §4 P2.2 dis-arm/re-arm policy + §4 P2.3 purge via `horizon_end_date` | P2 | 1–2 |
| **Step 8** — feature 1: \|z\| at trigger | §4 P2.4 `f_abs_z` | P2 | 2 |
| Step 8 — feature 2: spread volatility 60d | §4 P2.4 `f_spread_vol_60d` (Δspread std; deviation from literal spec disclosed) | P2 | 2 |
| Step 8 — feature 3: residual momentum 5d | §4 P2.4 `f_resid_mom_5d` (sign convention defined) | P2 | 2 |
| Step 8 — feature 4: market volatility (PC1, 20d) | §4 P2.4 `f_mkt_vol_20d` (`factors_a` for all tracks) | P2 | 2 |
| Step 8 — feature 5: relative volume 20d | §4 P2.4 `f_rel_volume_20d` | P2 | 2 |
| Step 8 — feature 6: days since last trigger | §4 P2.4 `f_days_since_trigger` (cap 126, no-prior=126) | P2 | 2 |
| Step 8 — feature 7: cluster stability | §4 P2.4 `f_cluster_stability` (default-0 policy stated) | P2 | 2 |
| Step 8 — no future info; standardize on training stats only | §4 P2.4 look-ahead assertion sweep + scaler fit on train inside `EntryModel.fit` (§5 contract) | P2 | 2 |
| **Step 9 E0** — fixed rule, implemented fairly (same engine/costs) | §5 P3.3 `e0_decisions` (same decisions-table path, zero special cases) | P3 | 2 |
| Step 9 E1 — logistic, L2 tuned on val, balanced, τ on val never test | §4 P2.8 + §5 P3.6 pre-registered τ rule | P2 | 3 |
| Step 9 E1 — coefficients reported | §4 P2.8 coefficient tables + §6.3 T3 | P2 | 3–4 |
| Step 9 E2 — GDA (generative comparison) | §3 P1.8 (numpy GDA ≡ sklearn LDA to 1e-6; QDA robustness row) | P1 | 3–4 |
| Step 9 E3 — small MLP, early stopping, weight decay, kept small | §5 P3.6 (7→{8,16}→1, fixed a priori, CPU, seeded) | P3 | 3 |
| Step 9 — bias-variance ladder discussion | §5 P3.6 expected-result framing (E0→E1→E3 ladder); discussion prose unplanned | P3 | 3, write-up phase |
| **Step 10a** — chronological splits, test touched once (spec: test 2023–2025) | §2.3 split config (test 2023–2024; §2.8 states dates frozen/ratified in v1, not reopened) + §4 P2.3 | P2 | 1 (frozen), 2 |
| Step 10b — purging (5d horizon) | §4 P2.3 rule 2 + purge property tests; rows retained and counted | P2 | 2 |
| Step 10c — embargo (10 trading days) | §4 P2.3 rule 3 + tests | P2 | 2 |
| Step 10d — AUC primary; precision/recall at τ; no accuracy | §5 P3.4 `classification_metrics` ("accuracy is never reported") | P3 | 2 |
| Step 10d — calibration / reliability diagram | §5 P3.4 `reliability_diagram` (10 quantile bins) + §6.3 F6 | P3 (owners read own) | 2 (figures 4) |
| Step 10d — strategy metrics: hit rate, mean/cum return, Sharpe, turnover, max drawdown | §2.2 metrics JSON schema + §5 P3.2 engine / P3.4 | P3 | 1–2 |
| Step 10d — bootstrap CIs on everything | §5 P3.4 `bootstrap_ci` (1000 resamples, seed 311; iid caveat in docstring) | P3 | 2 |
| Step 10e — smell test posted and applied | §6.2 Day 3 validation gate (every cell, as a team) + §5 P3.4 ≥0.65 bug rule + risk R8 | all | 3 |
| **Step 11** — costs per transaction, both legs, both ends | §2.3 cost constants + §5 P3.2 (−2c bps booked at entry and exit; cost-arithmetic test) | P3 | 1–2 |
| Step 11 — cost sweep 0–50 bps, one line per strategy | §5 P3.8 + §6.3 F7 | P3 | 4 |
| Step 11 — breakeven cost extraction | §5 P3.8 `breakeven_bps` (analytic + interpolated, agree to 1e-6) | P3 | 4 |
| **Step 12a** — factorial grid runs (3×4 in v2, extensible) | §5 P3.3 `run_cell`/`make_grid_table` + §6.3 T2 | P3 | 3 (val), 4 (test) |
| Step 12a — row/column/interaction analysis (the three questions) | §5 P3.3 row/column marginal means built into T2; written answers unplanned | P3 | 3–4, write-up phase |
| Step 12b — turnover-matched control | §5 P3.9 (per-quarter matching, 1000 seeded draws, percentile + F8); each owner runs own model | P3 machinery; all run | 4 |
| Step 12c — consensus pairs (buckets by selection count) | §4 P2.11 consensus-lite (pair-quarter granularity) + §6.3 F9 | P2 | 4 |
| Step 12d — factor-neutralization question (Han et al. engagement) | Not in v2 Days 1–4 (no task; inputs = T2 row marginals). Prose-only → write-up phase | TBD at write-up | write-up phase |
| Step 12e — coefficient comparison across tracks | §4 P2.10 + §6.3 T3 | P2 | 4 |
| Step 12e — base-rate decay by year | §4 P2.9 + §6.3 F5 | P2 | 3 (figure 4) |
| Step 12e — error analysis on confident false positives | **CUT in v2** (§1.4 delta V6; §4 P2.10 note "explicitly cut, not half-built"; §6.5 pre-committed) — coefficient comparison + calibration commentary carry the 12e load | — | CUT |
| **Part 4 item 1** — PCA trailing windows only | §3 P1.10 item 1 (future-perturbation, two cut dates + written note) | P1 | 3–4 |
| Part 4 item 2 — standardization window-local | §3 P1.10 item 2 (perturbation + grep audit) + §2.5 leakage helper | P1 | 3–4 |
| Part 4 item 3 — betas estimated before application | §3 P1.10 item 3 (perturbation + P1.3 in-loop assertion) | P1 | 3–4 |
| Part 4 item 4 — clustering formation-window only | §3 P1.10 item 4 (perturbation on labels/stability + code-path note) | P1 | 3–4 |
| Part 4 item 5 — labels purged at boundaries | §4 P2.12 item 5 (P2.3 tests + retained `purged` rows, exact counts) | P2 | 2 (tests), 4 (note) |
| Part 4 item 6 — embargo applied | §4 P2.12 item 6 (P2.3 tests + counted `embargo` rows) | P2 | 2 (tests), 4 (note) |
| Part 4 item 7 — test set touched exactly once | §5 P3.3 hard guard (date ≥ 2026-08-05, `FREEZE.md`, `results/final/TEST_RUN_COMPLETE` single-shot marker) + P3.10 freeze + §6.2 witnessed protocol + §7.10 | P3 | 2 (guard), 4 (run) |
| Part 4 item 8 — signal at t, returns at t+1 (hand-traced 3 dates) | §5 P3.2 shift test + §3 P1.9 manual trace (3 regimes, template) + §5 P3.11 item 8 | P1 (trace) + P3 (mechanism) | 1–3 (trace 3) |
| Part 4 item 9 — costs at each transaction | §5 P3.2 cost-arithmetic test (daily −2c booking) + P3.11 item 9 | P3 | 1–2 (note 4) |
| Part 4 item 10 — Bloomberg point-in-time or disclosed | §4 P2.6 (`FIELDS_USED.md`) + P2.12 item 10 (incl. fallback-branch disclosure) | P2 | 2 (note 4) |
| Part 4 item 11 — (Track C) precision matrix in-window, shrinkage documented | §5 P3.5.1/P3.5.2 (lam from config, recorded, not tuned) + P3.11 item 11 | P3 | 3 (note 4) |
| Part 4 item 12 — (Track D) AE trailing-only, schedule + standardization documented | **CUT** — Track D cut in v2 (§1.4 V3); checklist item lapses with the track; disposition = spec-8.4 lit-review sentence | — | CUT |
| Part 4 — noise test (full pipeline on random walks) | §5 P3.7 `make noise-test` + 3 binding criteria (all four models at once; Track C path included) | P3 | 3 |
| Part 4 — manual trace | §3 P1.9 `docs/manual_trace.md`, 3/3 MATCH (non-author of engine) | P1 | 3 |
| Part 4 — checklist published IN the report | §6.2/§6.3 T4 (P1 formats from three audit-note sets; each item signed by verifying owner); report insertion at write-up | P1 + all | 4, write-up phase |
| **Part 5** — honest novelty correction, cite Ekinci + ICBDEIM | Not in v2 Days 1–4 (no lit-review task in the draft). Prose-only → write-up phase | TBD at write-up | write-up phase |
| Part 5 — contribution sentence in report | Not in v2 Days 1–4. Prose-only → write-up phase | TBD at write-up | write-up phase |
| Part 5/7 — pre-registered primary comparison before running anything | §1.4 (Track A × E1 vs Track A × E0) + §2.3 `PRIMARY_COMPARISON` (frozen at kickoff) | all | 1 |
| **Part 6** — expectations calibration (weak results legitimate) | Distributed: §6.2 smell-test bands + "results are what they are" protocol; §5 P3.6 expected-result framing; §4 P2.11 "little overlap is a finding"; discussion prose unplanned | all | 3–4, write-up phase |
| **Part 7** — survivorship bias | §3 P1.1 + §2.6 limitations Day-1 seed → report | P1 | 1, write-up phase |
| Part 7 — only 40 stocks, thin correlation matrices | §2.6 limitations.md Day-1 seed (owner-tagged Part 7 bullet) | P1 | 1, write-up phase |
| Part 7 — Bloomberg not point-in-time | §4 P2.6 → limitations | P2 | 2, write-up phase |
| Part 7 — column-drop/median-impute choice | §4 P2.7 (logged) → limitations | P2 | 2, write-up phase |
| Part 7 — simplified z-score vs OU s-score | §2.6 limitations Day-1 seed | P1 | 1, write-up phase |
| Part 7 — beta drift | §2.6 limitations Day-1 seed | P1 | 1, write-up phase |
| Part 7 — k-means local optima / unstable membership | §6.3 F4 stability figure + §2.6 limitations seed | P1 | 1, 4, write-up phase |
| Part 7 — label-parameter sensitivity | §2.3 frozen label params + §4 P2.9 escalation protocol (val-only changes, logged) + limitations seed | P2 | 1, write-up phase |
| Part 7 — 18×18 from 40 stocks statistically thin | §4 P2.7 cap-at-5 caveat → limitations | P2 | 2, write-up phase |
| Part 7 — (Track C) shrinkage is a modelling choice | §5 P3.5.1 (`PARTIAL_CORR_SHRINKAGE` recorded, not tuned) → limitations | P3 | 3, write-up phase |
| Part 7 — (Track D) coarser retrain schedule, small capacity | **CUT** — lapses with Track D (§1.4 V3); replaced by the spec-8.4 lit-review sentence | — | CUT |
| Part 7 — small trigger sample → wide CIs | §5 P3.4 bootstrap independence caveat (docstring verbatim + limitations bullet) | P3 | 2, write-up phase |
| Part 7 — stylized cost model | §5 P3.2 aggregation-policy disclosure + limitations | P3 | 1–2, write-up phase |
| Part 7 — short-sale constraints / borrow costs ignored | §2.6 limitations Day-1 seed (every Part 7 bullet seeded, owner-tagged) | all (seed); write-up owner TBD | 1, write-up phase |
| Part 7 — multiple-testing statement (expected false positives, emphasize patterns) | §2.3 `PRIMARY_COMPARISON` + §6.5 (disclosure written for grid actually run; arithmetic uses 12 cells) + §2.6 limitations seed | all (no named v2 owner) | 1, write-up phase |
| **Part 8.0** — Track C gate: full grid end-to-end + audits passed | Superseded in v2: Track C **committed** (§1.4 V2, grid 3×4); single de-scope rule at Day 3 midday checkpoint (§6.2, §5 P3.5, R9) — validation-only or cut, never blocks the Day 4 test run | all (P3 owns Track C) | 3 |
| Part 8.0 — Track D gate | **CUT** — Track D pre-committed cut (§1.4 V3, §6.5: "no gate, no task, no config constants"); disposition = spec-8.4 one-sentence citation | — | CUT |
| Part 8.0 — priority: Track C first; never-cut list preserved | §6.5 cut ladder (ordered) + never-cut list restated verbatim | all | 1–4 (standing) |
| 8.1 Track C step 1 — reuse in-window correlation matrix | §3 P1.2 `return_corr` hook (built Day 1) + §5 P3.5 (consumes P1's matrices) | P3 (P1 hook) | 1 (hook), 3 |
| 8.1 step 2 — diagonal shrinkage before inverting | §5 P3.5.1 (`lam=1e-3` from §2.3, reported, not tuned) | P3 | 3 |
| 8.1 step 3 — precision → partial correlation | §5 P3.5.1 (unit tests incl. analytic 3-variable case; stable near-singular inversion) | P3 | 3 |
| 8.1 step 4 — distance + identical k-means machinery | §5 P3.5.2 (same 21d cadence, k-range 8–13, seed 311; precomputed-metric silhouette) | P3 | 3 |
| 8.1 step 5 — same Step-5 rules, `source="track_c"` | §5 P3.5.3 (shared builder; spreads from `residuals_a` via P1; triggers via P2's CLI) | P3 | 3 |
| 8.1 — overlap measurement (extends consensus) | §5 P3.5.4 per-window Jaccard figure (stretch item; feeds P2.11 if time) | P3 | 4 (optional) |
| 8.2 Track D — Baldi–Hornik linear-AE ≡ PCA check FIRST | **CUT** — Track D cut (§1.4 V3, §6.5); disposition = spec-8.4 one-sentence lit-review citation | — | CUT |
| 8.2 — tiny AE, bottleneck matched to PCA m | **CUT** — as above (AE config constants removed from §2.3) | — | CUT |
| 8.2 — monthly retrain on trailing 252d, mismatch → limitations | **CUT** — as above | — | CUT |
| 8.2 — window-local input standardization; strictly OOS residual | **CUT** — as above | — | CUT |
| 8.2 — same downstream, `source="track_d"`; integration by A | **CUT** — as above (`track_d` branch removed from contracts, §2.2) | — | CUT |
| 8.3 — grid/consensus/multiple-testing absorption of extension rows | §5 P3.3 T2 committed at 3×4 + §4 P2.11 buckets by selection count ("exactly spec 8.3's extension") + §6.5 12-cell arithmetic | P3/P2 | 3–4 |
| 8.4 — skip path: one-sentence citations | Track D: §1.4 V3 + §6.5 ("one honest sentence in the report"); any ladder cut likewise §6.5. Sentence itself is prose | all (prose TBD) | write-up phase |
| **Report** — figures the spec calls out (scree, var-over-time 2020, calibration, cost sweep, control histogram, base-rate decay, consensus, stability) | §6.3 inventory F1–F9, T1–T4 with source scripts, owners, due days (report-section ownership deferred to write-up) | per §6.3 table | 2–4 |
| Report — leakage checklist reproduced in full | §6.2/§6.3 T4 (assembled Day 4, signed per item); reproduction in report at write-up | P1 + all | 4, write-up phase |
| Report — contribution sentence + honest positioning | Not in v2 Days 1–4. Prose-only → write-up phase | TBD at write-up | write-up phase |
