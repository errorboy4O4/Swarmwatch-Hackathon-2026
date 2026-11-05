# PLAN.md — draft v0 (pending human decision on approach)

## Approach options
**A. Unsupervised forensics on Swarm Traces (recommended; runs today, data open).**
Build the parent_id graph (61k edges), reconstruct chains/clusters, find structure in payloads (shared templates, redacted-token reuse such as runtime_identifier/destination ids, response↔payload links), recover coordination signals without timestamps or labels. Validate with: article-evidence tagged rows as a small labelled subset, synthetic-injection recall (plant known coordinated templates into held-out rows), and a trivial baseline (exact-duplicate / n-gram clustering) vs. the method. No supervised XGBoost claim.

**B. AI Village (labels/time/agent IDs available) — blocked on gated access.** Closest to the brief's feature set (timing, agent identity). Only viable if access is approved by ~Nov 6. Keep as upside.

**C. Hybrid:** A as primary, B as a second dataset if access arrives.

## Architecture (A)
`src/swarmwatch/` ingest.py (stream jsonl.gz) → graph.py (parent graph, components) → features.py (template hashing, token reuse, text stats) → detect.py (baseline + clustering/optional classifier) → eval.py. `scripts/download_data.py`, `tests/` with small fixture, pinned `requirements.txt`, fixed seeds, `report/` (8-page template + limitations/dual-use appendix).

## Tasks
1. EDA notebook/script on Swarm Traces (graph sizes, template reuse, response linkage). 
2. Baseline + method + synthetic-injection and tag-subset evaluation; ablations.
3. Tests (pytest) + fresh-venv reproduction.
4. Report (template, ≤8 pages) + dual-use appendix; README with prior-work disclosure.
5. Red-team subagent pass Nov 8 AM; freeze 2 PM IST; submit by 6 PM IST.
