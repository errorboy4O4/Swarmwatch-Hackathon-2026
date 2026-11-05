# SwarmWatch — what can a log monitor see in multi-agent swarms?

Submission to the Apart Research **Swarm Alignment Hackathon 2026**, Track 4 (Multi-Agent Oversight and Control).
Report: `report/SwarmWatch_report.pdf` (source `report/report_final.md`; rebuild with `python scripts/make_figures.py && python scripts/build_pdf.py`). `report/report.md` is an earlier draft. Honest summary of results: `STRATEGY.md`.

## What this is
Two small, placebo-controlled studies on real multi-agent logs, with simple baselines and ablations:
1. **Swarm Traces** (reconstructed payloads from the July 2026 Hugging Face incident): attribute rows to masked credential groups (a *proxy* for "same actor") using content, infrastructure tokens and parent/child graph context.
2. **AI Village `swarm-village`** (133 copies of one model, one chat room, 24h): can chat behaviour alone reveal which agents hold governance roles, and how much of that signal survives if the monitor sees only a sample of messages? Roles were assigned at the start of Phase 2, so Phase-1 chat is a built-in placebo.

Main finding (see report for numbers and caveats): the signal is real but modest, a one-feature baseline (how often an agent is @-mentioned) matches XGBoost, burst-timing features add nothing, and the signal degrades gracefully with partial visibility. It is *not* a collusion detector.

## Prior work disclosure
All code, data processing and analysis were written during the hackathon window (Nov 4–8, 2026; recon and set-up began Nov 4, before the Nov 6 start — disclosed here). No pre-existing project code was reused. AI assistance (Claude Code) was used for coding and drafting; every number comes from a logged run in `logs/`.

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
```
Seeds are fixed in the scripts. Logged outputs from our runs are in `logs/`.

## Layout
`src/swarmwatch/` ingest, attribution and village-feature code · `tests/` pytest suite · `scripts/` runnable experiments · `logs/` outputs · `report/` report · `STRATEGY.md`, `PLAN.md` working notes.

## Limitations (short)
Proxy labels on Swarm Traces; 133 agents / 26 positives on AI Village (wide CIs); one village, one model; roles were randomly assigned, so this measures assigned-role effects, not emergent collusion. See report.
