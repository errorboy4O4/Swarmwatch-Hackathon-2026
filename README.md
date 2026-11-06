# SwarmWatch — what can a log monitor see in multi-agent swarms?

Submission to the Apart Research **Swarm Alignment Hackathon 2026**, Track 4 (Multi-Agent Oversight and Control).
Report: `report/SwarmWatch_report.pdf` (source `report/report_final.md`; rebuild with `python scripts/make_figures.py && python scripts/build_pdf.py`). Honest summary of results: `STRATEGY.md`.

## What this is
Two small, placebo-controlled studies on real multi-agent logs, with simple baselines and ablations:
1. **Swarm Traces** (reconstructed payloads from the July 2026 Hugging Face incident): attribute rows to masked credential groups (a *proxy* for "same actor") using content, infrastructure tokens and parent/child graph context.
2. **AI Village `swarm-village`** (133 copies of one model, one chat room, 24h): can chat behaviour alone reveal which agents hold governance roles, and how much of that signal survives if the monitor sees only a sample of messages? Roles were assigned at the start of Phase 2, so Phase-1 chat is a built-in placebo.

Main finding (see report for numbers and caveats): the signal is real but modest, a one-feature baseline (how often an agent is @-mentioned) matches XGBoost, burst-timing features add nothing, and the signal degrades gracefully with partial visibility. It is *not* a collusion detector.

## Prior work disclosure
The hackathon judges only work done during the sprint (Nov 6-8, 2026). **Everything in this repository up to the first push was done on Nov 4-5, 2026, before the sprint started**: recon, dataset access, all code, all experiments, the red-team review, and the first draft of the report (git commit dates show this). No code from earlier projects was reused. AI assistance (Claude Code) was used for coding and drafting; every reported number comes from a logged run in `logs/`.

### Sprint-window changes (Nov 6-8)
- **Nov 6:** `scripts/observability.py` + `logs/observability.json`: Swarm Traces attribution at four observability tiers (full log / gateway metadata only / flat log / identifier tokens only), reported as Table A2 in the report appendix. Reuses pre-sprint code and data; its full and gateway tiers reproduce the pre-sprint ablation numbers exactly (0.726 and 0.596 macro-F1).
- **Nov 6:** `src/swarmwatch/roles.py`, `scripts/role_recovery.py`, `logs/role_recovery.json` (+ `tests/test_roles.py`): recovers role changes around the reassignment mistake from chat self-identification (noisy; hand audit of 20 flagged cases: 17 clear, 1 ambiguous, 2 false positives). Reported in the report's Results and Appendix B. The script also writes `logs/role_audit_sample.json` (20 verbatim AI Village message excerpts used for the hand audit); it is git-ignored and not included because the dataset's terms forbid redistribution. The audit labels are recorded by index in the script, and the sample is seeded, so it regenerates identically once you have dataset access.
- **Nov 6:** `src/swarmwatch/evasion.py`, `scripts/evasion.py`, `logs/evasion.json` (+ `tests/test_evasion.py`): simulated evasions of the AI Village log (strip @-mentions, mask role words, clip length) and a white-box vocabulary-deletion adversary vs static and retrained content monitors. Reported in the report's Results and Appendix B (Tables B2, B3). Rewritten logs, not adapting agents.
- **Nov 6:** `scripts/visibility.py`, `logs/visibility.json`: equal-budget comparison of random, time-block and agent-subset visibility on AI Village (Table B4 in the report). Deterministic (rerun identical).
- **Nov 6:** `scripts/ci.py`, `logs/ci.json`: bootstrap 95% intervals for the headline numbers (Table 1); `tests/test_integration.py`: end-to-end smoke test on a synthetic fixture.
Anything else added from Nov 6 onward will be listed here with commit hashes.

## Data (NOT included in this repo)
- Swarm Traces: public. `python scripts/download_swarmtraces.py`
- AI Village: gated on Hugging Face (`aidigestorg/ai-village`, research-use terms). Request access yourself, then
  `HF_TOKEN=<read-only token> python scripts/download_aivillage.py get swarm-village/README.md swarm-village/manifest.json swarm-village/agents.jsonl.gz swarm-village/chat_rooms.jsonl.gz swarm-village/chat_messages.jsonl.gz swarm-village/events.jsonl.gz swarm-village/computer_use_sessions.jsonl.gz swarm-village/agent_goals.jsonl.gz`
  Do not redistribute the data.

## Reproduce
```
python -m venv .venv && .venv/Scripts/activate   # (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python -m pytest -q
python scripts/eda.py > logs/eda.txt
python scripts/run_experiment.py > logs/experiment.json        # Swarm Traces, ~45 min on a laptop
python scripts/run_village.py    > logs/village.json           # AI Village
python scripts/perm_test.py      > logs/perm.json
python scripts/reassign_check.py > logs/reassign.json
python scripts/village_followup.py > logs/village_followup.json   # content baselines, placebo diff, Holm
python scripts/observability.py > logs/observability.json      # Swarm Traces observability tiers (sprint-window)
python scripts/role_recovery.py > logs/role_recovery.json      # role changes from chat self-identification (sprint-window)
python scripts/evasion.py > logs/evasion.json                  # simulated evasions + white-box vocabulary adversary (sprint-window)
python scripts/visibility.py > logs/visibility.json            # random vs targeted visibility at equal budget (sprint-window)
python scripts/ci.py > logs/ci.json                            # bootstrap 95% CIs for headline numbers (sprint-window)
```
Seeds are fixed in the scripts. Logged outputs from our runs are in `logs/`.

## Layout
`src/swarmwatch/` ingest, attribution and village-feature code · `tests/` pytest suite · `scripts/` runnable experiments · `logs/` outputs · `report/` report · `STRATEGY.md`, `PLAN.md` working notes.

## Limitations (short)
Proxy labels on Swarm Traces; 133 agents / 26 positives on AI Village (wide CIs); one village, one model; roles were randomly assigned, so this measures assigned-role effects, not emergent collusion. See report.
