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
