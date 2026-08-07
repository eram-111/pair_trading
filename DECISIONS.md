# DECISIONS.md — append-only

Format: `YYYY-MM-DD | decision (old -> new where applicable) | initials of all agreeing`
Required for: any config.py change, any artifact-schema change, universe substitutions,
the Bloomberg go/fallback decision, the Track C de-scope decision, each model's τ,
and the model-freeze hashes. Scribe: P1. No silent changes, ever.

2026-08-02 | Plan v2 adopted: vertical slices P1/P2/P3 with a shared built-once spine (engine, dataset builder, metrics) | (ratify at kickoff)
2026-08-02 | Track C promoted to committed third pairing method (grid = 3×4); Track D cut per spec 8.4 | (ratify at kickoff)
2026-08-02 | 4-day build Aug 2–5; results freeze Day 4 EOD; single witnessed test run Day 4 ~noon; write-up Aug 6–7 planned separately | (ratify at kickoff)
2026-08-02 | Repo layout: flat src/ (~16 files, one owner each; models/ the only subpackage); "modular" means people, not files | (ratify at kickoff)
2026-08-02 | Environment: conda (environment.yml, pip-pinned section); torch installed post-create from the CPU wheel index | (ratify at kickoff)
2026-08-02 | Track B: Day-1 stand-in in the loader's exact schema → Monday Bloomberg swap; swap deadline Mon EOD, else price-derived fallback is final | (ratify at kickoff)
2026-08-03 | torch 2.13.0 -> 2.2.2 team-wide (single version; Intel-mac ceiling) + numpy 2.5.1 -> 1.26.4 (torch-2.2 compat; all pins verified vs 1.26) | (ratify at sync)
2026-08-06 | Shared clustering API as-built (representation.py, P3): comembership -> pair_from_labels (now takes explicit tickers list); stability_frame -> pair_stability_table; fit_kmeans_select_k unchanged. Plan §P1.5 updated to match | GE
2026-08-06 | Pair-builder + spread API as-built (representation.py, P3): run_recluster_loop -> cluster_all_windows(feature_table_by_window, k_range); build_pairs takes the labels DataFrame and returns the table (caller writes CSV; pairs_for_window folded in); split_large_cluster takes feature rows, not a distance matrix; build_spread_for_pair/zscore/run_spreads -> one build_spreads(residuals, pairs) -> (spreads, zscores) (caller writes parquet). Plan §2.2/§P1.5/§P1.6/§P1.7 updated | GE
2026-08-07 | src/noise_test.py added, split out of experiments.py (P3): the noise run drives the pipeline directly and shares no grid machinery, so the runner stays evaluation-only; make noise-test now calls python -m src.noise_test. Plan §P3.7 module line updated | GE
2026-08-07 | Runner supports --split val (val rows only) alongside trainval/test: the pre-test model-vs-e0 comparison is judged on val alone (models never fit those rows); trainval grid stays as the integration/status artifact | GE
