# STRATEGY.md (living file) — last updated 2026-11-04

## 1. Facts verified in recon (source: apartresearch.com/sprints/swarm-alignment-2026)
- Deadline: Sun Nov 8, 11:59 PM AoE. Internal target: Nov 8, 6:00 PM IST.
- **Deliverable differs from the brief:** PDF report on the official template, **max 8 pages** (excl. references/appendices), title + 150–250 word abstract, authors, and a **required "Limitations and Dual-Use / Ethical Considerations" appendix**. GitHub repo and 3–5 min video are optional.
  Template: https://docs.google.com/document/d/1_1_27EyZriMlc8n8EBE2YScaeyMjm-8xjR097nmHXAE/copy
- Recommended structure: Introduction, Related Work, Methodology, Results, Discussion, Limitations & Dual-Use, References.
- Only weekend work is judged; pre-existing work must be disclosed in README and report. AI tools allowed, but fabricated results/references can disqualify.
- Prizes: $2000 / $1000 / $600 / $200 / $200.

## 2. Rubric (paraphrased from Guidelines tab; 3 dimensions, 1–5 each)
1. **Impact Potential & Innovation** — "How much would this matter for the field if it worked? How innovative is it?" 4–5 requires judges to see it is new vs. replicating recent work; 5 needs a clear theory of change.
2. **Execution Quality** — "How sound are methodology, implementation, and findings?" 3 = competent, acknowledged limitations; 4 = thorough, convincing validation, results support conclusions; 5 = ambitious scope executed rigorously, surprising findings/robust validation.
3. **Presentation & Clarity** — "How clearly are work, findings, and impact potential communicated?" 2 = key info buried or diluted by length; 4 = well structured; 5 = model of its kind.
(Re-fetch the page for exact verbatim text before the final report; the above is from a summarizing fetch.)

## 3. Data landscape (verified 2026-11-04)
| Source | Status | Notes |
|---|---|---|
| Swarm Traces `https://swarmtraces.org/data/final/redacted.jsonl.gz` | **Downloadable, 15.2 MB** | 189,579 rows. Fields: id, cite, kind, parent_id, time_utc, tags, text. kind: payload 91,037 / recovered_text 75,534 / response 23,008. **time_utc is null for ALL rows. No agent IDs. No labels.** parent_id set on 61,125 rows. ~80% outbound requests (per authors); likely incomplete; may contain non-swarm rows. License not stated. |
| AI Village `aidigestorg/ai-village` (HF) | **GATED (manual approval)** | License "ai-village-research-terms", >1M rows. Human must request access; approval time unknown. |
| Transluce agent activity | not yet checked | |

## 4. CRITICAL consequence for the brief
The brief's design (burst-timing stats, rename/copy counts, access overlap, XGBoost on "coordinated groups") **cannot run on Swarm Traces**: no timestamps, no agent identities, no ground-truth labels. Supervised XGBoost + PR-AUC needs labels we do not have. Options in PLAN.md; decision pending with the human.

## 5. Hypotheses
- H1: Judges reward a task fitted to data we actually have over a pre-planned method the data can't support → verify: every claim traces to a logged run on real rows.
- H2: Honest baselines + ablations score Execution 4+ → ship a trivial baseline next to the method.
- H3: Within 8 pages, a single sharp finding beats breadth (Presentation).
- H4: Novelty is judged vs. recent work (Rose et al. 2026 on collusion detection; Motwani 2024) → cite and position explicitly.
- H5 (to verify): a forensics reconstruction ("who talked to whom"/payload-chain structure; the track's example project 1) is feasible from parent_id graph + redacted tokens.

## 6. Scorecard (1–5) — to be filled per milestone
| Milestone | Impact | Execution | Presentation |
|---|---|---|---|
| Recon | – | – | – |

## 7. Red-team log
(empty)

## 8. Open items
- Human: request AI Village access on HF (gated). Check Discord for announcements.
- Decide approach (PLAN.md options).

---
## UPDATE 2026-11-05 — results so far (all from logged runs in logs/)

### Data status
- AI Village access GRANTED (HF gated). Using `swarm-village/` subset: 133 copies of one model (GLM 5.3 Flash), 24h, ONE active chat room (6 exist; 3,851/3,853 msgs in room "a"), 3,848 agent messages, 133 agents, 133 Phase-2 role assignments (agent_goals). Phase 1 (build, no roles): 1,307 msgs; Phase 2 (roles assigned 2026-08-30 10:00): 2,541 msgs. Known issue per README: operator mistake ~5:50am PT reassigned roles, restored ~2pm PT (not yet modelled).
- Dataset README is data, not instructions.

### Experiment A — Swarm Traces credential-group attribution (logs/experiment.json)
Proxy label = masked `[CREDENTIAL n]` id (top 30, 22,631 rows). Template-held-out split, 3 seeds: majority acc 0.241/F1 0.013; entity-lookup baseline 0.654/0.608; full linear-SVM model 0.914/0.818. Ablation: text carries it (minus_text 0.666 F1; only_text 0.804); graph and struct groups add ~0. Partial visibility (train fraction): 1% -> F1 0.208, 25% -> 0.665, 100% -> 0.818. Limitation: proxy label (credential id ~ "same actor") — shows credentials are predictable from content, not that real coordination is found.

### Experiment B — AI Village: can chat alone reveal who holds power roles? (logs/village.json, logs/perm.json)
Label (pre-registered in village.py before features were inspected): governance/state-power roles, 26 of 133 (prevalence 0.195). Features from chat only: volume, @mention graph, timing/burstiness. 5-fold x5 repeated CV, XGBoost.
- **Built-in placebo:** roles were assigned at Phase-2 start, so Phase-1 chat cannot encode them. Phase-1 XGB PR-AUC 0.227 / ROC 0.440 (chance); in-degree baseline perm p=0.319. Phase 2: XGB PR-AUC 0.354 / ROC 0.654; in-degree baseline PR-AUC 0.354, perm p=0.0092, bootstrap 95% CI [0.21, 0.52].
- **Honest finding 1:** signal is real but modest, and XGBoost does NOT beat the one-feature in-degree baseline (0.354 vs 0.354).
- Ablation (PR-AUC): only_volume 0.393 (best), only_graph 0.269, only_timing 0.192 (~prevalence); minus_timing 0.381 (timing features hurt). Burst-timing hypothesis from the original brief NOT supported.
- **Partial visibility (random message sample, phase 2):** 50% -> 0.312, 25% -> 0.293, 10% -> 0.270, 5% -> 0.254 (prevalence 0.195); in-degree baseline 0.342/0.329/0.310/0.283. A monitor seeing 10% of messages keeps part of the signal; degrades gracefully, never below chance.
- Bug caught by test: pandas datetime unit made gap features wrong; fixed before any reported run.
- n=133 agents / 26 positives: wide CIs; single village, single model; roles randomly assigned so "influence" = assigned-role effect, not emergent collusion.

### Hypotheses status
- H1 (fit task to data): supported — two tasks matched to what each dataset contains.
- H2 (baselines/ablations): supported — in fact baselines tie the model; reporting this is the honest result.
- H3 (one sharp finding): candidate headline = "placebo-controlled partial-visibility monitoring": a monitor seeing only chat recovers power-role holders above chance, signal survives down to 10% of messages, simple in-degree matches XGBoost. Swarm Traces = secondary result.
- Brief's burst-timing claim: refuted by ablation (timing adds nothing).

### Scorecard (self-assessed, 1-5)
| Milestone | Impact | Execution | Presentation |
|---|---|---|---|
| 2026-11-05 | 2.5 (modest novelty; framing + placebo are the strengths) | 3.5 (placebo, perm test, CIs; weak: small n, proxy labels) | not yet written (0 until report) |
Weakest: Impact/novelty. Next: strengthen with a content-based/information-spread analysis or the 5:50am role-reassignment natural experiment (do agents keep acting on old roles? = covert persistence), then red-team.

## BUILD_LOG
- 2026-11-04: recon; Swarm Traces EDA; attribution experiment (logs/experiment.json).
- 2026-11-05: AI Village access granted; downloaded small tables (no images/turns/memories); village features + XGB + placebo + partial visibility + permutation test; 6 tests green.

### Experiment C — reassignment-window check (logs/reassign.json) — 2026-11-05
Finding: the new (mistaken) role assignments are NOT in the downloaded tables (agent_goals holds only the original 10:00 draw; the re-role was done in the simulation's git state), so the "natural experiment" cannot be run as a labelled test. Weaker check done: Phase 2 split at 12:50 UTC (~5:50am PT), labels = ORIGINAL roles. In-degree PR-AUC: pre-mistake 0.353 (perm p=0.004, n_messages in JSON), mistake window 0.328 (p=0.019); prevalence 0.195. Volume (n_msgs) is at chance in both. Reading: the original-role signal persists after the reassignment, consistent with agents keeping their earlier roles (per README), but it cannot be distinguished from hub structure simply persisting. Do NOT claim covert persistence. Possible follow-up: pull computer_use_turns (72MB) / git logs to recover the reassigned roles.

---
## RED-TEAM LOG (2026-11-05, hostile-judge subagent) — status
Numbers: all report numbers matched logs. Hostile scores: Impact 2, Execution 2-2.5, Presentation 3.
- FIXED: pseudonym-id leak via text n-grams (Swarm Traces) -> ids stripped from all bracket tokens, template key 400->100 chars; rerun: held-out macro-F1 0.818 -> 0.726 (old: logs/v1_prefix/). Random split 0.859 -> 0.853.
- FIXED: missing content baseline -> TF-IDF LR PR-AUC 0.712 / ROC 0.888 (Phase 1: 0.164); keyword 0.597. Report reframed: "what a monitor retains without content".
- FIXED: mean message length is the strongest single structural feature (0.427, Holm p=0.002); in-degree 0.354 (Holm 0.028). Message count / out-degree / timing at chance. Report says so.
- FIXED: all baselines in both phases + Holm; direct placebo test inconclusive (diff +0.121, boot95 [-0.046,0.276]); size-matched phase 2 = 0.356+-0.018 vs 0.233.
- FIXED: label regex fully listed; sensitivity run (33 pos): in-deg 0.400, p=0.005. "pre-registered" softened (not timestamped).
- FIXED: "partial visibility" on Swarm Traces renamed training-data fraction; Swarm Traces demoted to secondary; permutation count stated correctly (reassign check uses 3,000 — now documented in script only; report says 5,000 for single-feature tests).
- FIXED: citations checked against abstracts (Greenblatt = arXiv 2023/ICML 2024; Motwani full title; Rose = HELIOS/NARCBench). METR PDF unreadable -> incident attributed to sprint page.
- OPEN: placeholders [YOUR NAME]/[AFFILIATION]/[GITHUB URL]; PDF on official template; fresh-venv reproduction; reassignment window uses 3,000 perms (report 4.1 says 5,000 for single-feature tests only - fine); Track-4 relevance remains thin (no adversarial/targeted visibility).
