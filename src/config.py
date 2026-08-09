"""Every tunable number and setting in the project.

All other code imports its constants from here — nothing re-declares a
window length, threshold, date, or seed locally.
"""
from __future__ import annotations
from pathlib import Path

SEED: int = 311
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

DOWNLOAD_START: str = "2014-01-01"
ANALYSIS_START: str = "2015-01-01"
DATA_END: str = "2025-01-01"
MAX_MISSING_FRAC: float = 0.02
LOG_RETURNS: bool = True
N_TICKERS: int = 40

PCA_WINDOW: int = 252
N_COMPONENTS_CANDIDATES: tuple[int, ...] = (3, 4, 5)
VAR_EXPLAINED_TARGET: float = 0.60
EIGENVECTOR_SIGN_RULE: str = "sum_of_loadings_positive"
BETA_WINDOW: int = 252

RECLUSTER_EVERY: int = 21
TRACK_B_REFRESH: str = "quarterly"
KMEANS_N_INIT: int = 10
K_RANGE: dict[str, range] = {
    "a": range(8, 14), "b": range(10, 14),
}
MAX_WHOLE_CLUSTER: int = 4
SUBGROUP_SIZES: tuple[int, int] = (2, 3)

SPREAD_KIND: str = "simple"
Z_WINDOW: int = 60
SPREAD_POLICY: str = "carry_with_burnin"
SPREAD_WARMUP_DAYS: int = 60

TRIGGER_Z: float = 2.0
REVERSION_FRACTION: float = 0.5
LABEL_HORIZON: int = 5

SPREAD_VOL_WINDOW: int = 60
RESID_MOM_WINDOW: int = 5
MKT_VOL_WINDOW: int = 20
REL_VOLUME_WINDOW: int = 20
FEATURES: tuple[str, ...] = (
    "f_abs_z", "f_spread_vol_60d", "f_resid_mom_5d", "f_mkt_vol_20d",
    "f_rel_volume_20d", "f_days_since_trigger", "f_cluster_stability",
)

TRAIN_START, TRAIN_END = "2015-01-01", "2020-12-31"
VAL_START,   VAL_END   = "2021-01-01", "2022-12-31"
TEST_START,  TEST_END  = "2023-01-01", "2024-12-31"
PURGE_DAYS: int = 5
EMBARGO_DAYS: int = 10

TAU_RULE: str = "see-README.md"
E3_HIDDEN: tuple[int, ...] = (8, 16)
TORCH_DEVICE: str = "cpu"

ENTRY_LAG_DAYS: int = 1
EXIT_Z: float = 0.5
MAX_HOLD_DAYS: int = 5
NOTIONAL_PER_LEG: float = 1.0
COST_GRID_BPS: tuple[int, ...] = (0, 5, 10, 15, 20, 30, 40, 50)
HEADLINE_COST_BPS: int = 10

MODEL_COLORS: dict[str, str] = {"e0": "#2a78d6", "e1": "#eb6834", "e2": "#1baf7a", "e3": "#eda100"}

PRIMARY_COMPARISON: tuple[str, str] = ("track_a__e1", "track_a__e0")
