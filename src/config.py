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
TAU_RULE: str = "see-plan-sec-5"  # the ONE pre-registered rule lives in Plan Section 5 (P3.6)
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
