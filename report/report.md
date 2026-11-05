# What a Log Monitor Retains When It Cannot Read the Messages: Placebo-Controlled Role Inference on an AI Village Swarm

**Author:** Kaushik Gaur — IIT Madras (B.S. Data Science, student) · Apart Research Swarm Alignment Hackathon 2026, Track 4 (Multi-Agent Oversight and Control)
**Code:** https://github.com/errorboy4O4/Swarmwatch-Hackathon-2026

## Abstract

Oversight of agent populations may have to work from logs whose content is unavailable, obfuscated or only partly visible. We ask how much a monitor learns about *who holds power* in a real swarm from message content versus from message structure alone. In the AI Village swarm run (133 copies of one model in one chat room) we test whether chat reveals which agents hold governance roles (26 of 133). Roles were assigned at the start of Phase 2, so Phase-1 chat is a placebo. A content monitor (TF-IDF logistic regression on each agent's messages) reaches PR-AUC 0.712 (ROC-AUC 0.888; prevalence 0.195), because agents openly state their roles. Structure-only monitors do much worse but not nothing: how often an agent is @-mentioned gives PR-AUC 0.354 (permutation p = 0.007) and mean message length 0.427; message count, out-degree and burst-timing features are at chance, and XGBoost on all structural features only ties the best single feature. The Phase-1 placebo is at chance, but a direct Phase-2-minus-Phase-1 test is inconclusive (bootstrap 95% interval for the in-degree gap spans zero). A secondary analysis of Swarm Traces, which has no timestamps, agents or labels, predicts a proxy actor label (masked credential id) at macro-F1 0.726 on held-out templates, almost entirely from text. These are small-sample, proxy- or assigned-label results, not collusion detection.

## 1. Introduction

When agents in a July 2026 evaluation formed a message board and many acted together against Hugging Face's systems (as described on the sprint page), the swarm was found weeks later. A monitor will often see only part of the traffic, or traffic whose content it cannot trust. We ask a narrow, measurable question: *on real swarm logs, what do simple monitors recover, how does that compare with the information in message content, and what happens with partial visibility?*

Contributions: (i) a placebo design for the AI Village data that exploits the known time at which roles were assigned; (ii) a direct comparison of content, structure and timing monitors against one-feature baselines; (iii) negative results — burst-timing features carry no signal, and XGBoost does not beat the best one-feature baseline; (iv) a leakage audit (Section 7). We do **not** claim to detect collusion or covert channels.

## 2. Related Work

Hammond et al. (2025) give a taxonomy of multi-agent risks built on miscoordination, conflict and collusion. Motwani et al. (2024) formalise secret collusion among AI agents via steganography and study countermeasures such as monitoring and paraphrasing. Rose et al. (2026) detect collusion from per-agent activation probes (HELIOS, evaluated on NARCBench). Greenblatt et al. (2023) study control protocols in which a trusted monitor oversees an untrusted model. Those works target collusion mechanisms, interpretability-based detection or protocols; ours is a descriptive log-forensics study that measures how much a role structure is visible in a real swarm's chat and how that changes with what the monitor can see. We have not compared against their methods.

## 3. Data

**AI Village `swarm-village`** (gated Hugging Face release, research-use terms): a 24-hour run of 133 copies of GLM 5.3 Flash. Six chat rooms exist but 3,851 of 3,853 messages are in one room; 3,848 are agent messages from 133 agents. Each agent received a Phase-1 goal and, at 10:00 on Aug 30 (dataset clock), a randomly assigned Phase-2 role. The dataset README notes that around 5:50am PT an operator mistake reassigned roles, restored around 2pm PT; the downloaded tables hold only the original assignment. Phase 1: 1,307 messages; Phase 2: 2,541.

**Swarm Traces** (swarmtraces.org, redacted release; secondary): 189,579 rows (payloads 91,037; recovered text 75,534; responses 23,008) in trees of depth at most two. All timestamps are null; there are no agent ids or labels. The authors say the reconstruction is likely incomplete and may include rows not from the swarm. Redaction replaces sensitive values with stable pseudonyms (`[CREDENTIAL n]`, `[SERVICE n URL]`, `[REDACTED:destination:…]`); 31,210 rows contain a credential pseudonym (2,673 distinct).

## 4. Methods

### 4.1 AI Village: inferring power roles from chat
*Label.* A "power role" is a governance/state-power role matched by a regular expression written in code before inspecting any features (this ordering is not independently timestamped): `^(President|Prime Minister|Deputy Prime Minister|Chief Councilor|Queen|Sultan|Secretary-General|Speaker|Eldest|Mayor|Minister|Chief Justice|Chief of Police|Director of the Central Bank|Ambassador|Tax Collector|god$)`. This marks 26 of 133 agents (prevalence 0.195). Judges, captains and similar roles are negative; a sensitivity run that also counts judges, captains, admiral, fire chief and harbor master gives 33 positives.
*Monitors.* Content: (a) keyword count of governance words in the agent's own messages (partly circular with the label), (b) TF-IDF logistic regression on the agent's concatenated messages. Structure: volume (message count, mean length), @-mention graph (in/out degree, weights, PageRank), timing (median gap, burstiness, fraction of gaps under 120 s). Models: XGBoost (150 trees, depth 3) on the structure groups, and each structural feature alone.
*Evaluation.* 5-fold × 5-repeat stratified CV (out-of-fold PR-AUC; chance = prevalence 0.195); permutation tests (5,000 permutations) for single-feature scores with Holm correction across the five single features per phase; 2,000-draw bootstrap intervals.
*Placebo.* Roles cannot be encoded in Phase-1 behaviour, so each monitor should score at chance there. *Partial visibility.* Random samples (50/25/10/5%) of Phase-2 messages, 5 draws.

### 4.2 Swarm Traces: credential-group attribution (secondary)
Label (proxy): rows with exactly one credential pseudonym, top 30 credentials (22,631 rows). The credential id is a proxy for "same actor/campaign", not ground truth. Features: `text` (char 3–5-gram TF-IDF with **every** bracketed pseudonym's ids stripped, credential masked), `entities` (other pseudonym ids), `graph` (entity tokens of parent and children), `struct`. Linear SVM; baselines: majority, template lookup, infrastructure-token lookup. 80/20 splits holding out whole templates (key = first 100 characters of normalised text), 3 seeds; leave-one-group-out and single-group ablations; training-data-fraction sweep. An earlier version masked only the credential token in the text view and used a 400-character template key; the red-team review showed pseudonym ids leaked through n-grams, and fixing this lowered held-out macro-F1 from 0.818 to 0.726 (old log kept in `logs/v1_prefix/`).

## 5. Results

### 5.1 AI Village

| Monitor (Phase 2 unless noted) | PR-AUC | ROC-AUC |
|---|---|---|
| Chance (prevalence) | 0.195 | 0.5 |
| TF-IDF logistic regression (content) | 0.712 | 0.888 |
| – same, Phase 1 (placebo) | 0.164 | 0.413 |
| Keyword count (content; partly circular) | 0.597 | – |
| – same, Phase 1 (placebo) | 0.211 | – |
| Mean message length | 0.427 (Holm p = 0.002) | 0.769 |
| In-degree (@-mentions received) | 0.354 (Holm p = 0.028) | 0.641 |
| PageRank | 0.310 (Holm p = 0.119) | 0.621 |
| Message count | 0.195 (n.s.) | 0.465 |
| Out-degree | 0.189 (n.s.) | 0.480 |
| XGBoost, all structure features | 0.354 | 0.654 |

Phase-1 single-feature scores (placebo): message count 0.223, mean length 0.152, in-degree 0.233, out-degree 0.200, PageRank 0.257 — none significant (Holm p ≥ 0.99). XGBoost on Phase 1: 0.227.

*Ablations (XGBoost, Phase 2, PR-AUC / ROC-AUC).* Volume only 0.393 / 0.686; graph only 0.269 / 0.494; timing only 0.192 / 0.456; minus timing 0.381 / 0.702; minus volume 0.236 / 0.474; minus graph 0.320 / 0.643. XGBoost does not beat the best single feature (mean length 0.427); differences among the XGBoost variants are within about one fold-noise standard deviation (0.02–0.04). Timing/burst features carry no signal. The graph-only model scores near chance in ROC-AUC although in-degree alone reaches 0.641, which we read as small-sample instability.

*Direct placebo test.* The in-degree PR-AUC gap Phase 2 minus Phase 1 is +0.121; the paired bootstrap 95% interval is [−0.046, 0.276] and 6.2% of draws are ≤ 0, so the difference is not established. Phase 2 subsampled to Phase 1's size (1,307 messages, 50 draws) gives 0.356 ± 0.018 versus 0.233 for Phase 1. Bootstrap intervals for in-degree alone are wide (Phase 2: [0.21, 0.52]).

*Partial visibility (random message samples, Phase 2, XGBoost PR-AUC).* 50%: 0.312; 25%: 0.293; 10%: 0.270; 5%: 0.254, against in-degree baselines 0.342 / 0.329 / 0.310 / 0.283. Signal weakens but stays above prevalence.

*Label sensitivity.* With the broader label (33 positives, prevalence 0.248), in-degree PR-AUC is 0.400 (p = 0.005).

*Reassignment window.* Splitting Phase 2 at 12:50 UTC (the reported mistake) and scoring against the original roles, in-degree PR-AUC is 0.353 before (602 messages, p = 0.004) and 0.328 after (1,939 messages, p = 0.019); message count is at chance in both. This cannot be separated from persistent hub structure; we draw no conclusion about agents persisting in old roles.

### 5.2 Swarm Traces (mean of 3 seeds, held-out templates)

| Method | Accuracy | Macro-F1 |
|---|---|---|
| Majority class | 0.189 | 0.011 |
| Infrastructure-token lookup | 0.571 | 0.548 |
| Linear SVM, all features | 0.874 | 0.726 |
| – minus text | 0.755 | 0.596 |
| – minus entities | 0.842 | 0.682 |
| – minus graph | 0.873 | 0.728 |
| – minus struct | 0.875 | 0.728 |
| – text only | 0.842 | 0.694 |
| – entities only | 0.728 | 0.582 |
| – graph only | 0.279 | 0.130 |

(Template lookup equals majority by construction on held-out templates. On random-row splits: template lookup 0.670 / 0.582, entity lookup 0.683 / 0.654, full model 0.920 / 0.853.) Text carries most of the signal; the parent/child graph and structural features add nothing measurable. Training-data-fraction sweep (seed 0; this reduces training rows only, test rows keep full context, so it is a learning curve, not a visibility test): 1%: 0.615 / 0.213; 5%: 0.699 / 0.361; 10%: 0.757 / 0.450; 25%: 0.812 / 0.617; 50%: 0.858 / 0.692; 100%: 0.882 / 0.734 (accuracy / macro-F1).

## 6. Discussion

*What is visible.* In this swarm, power roles are legible mainly through what agents *say*: a content monitor scores PR-AUC 0.712, and agents state their roles openly, so this is the unsurprising result and gives the structural monitors their reference point. When content is withheld, structure retains a modest but non-trivial signal: @-mention in-degree and mean message length are above chance, while volume, out-degree and burst timing are not. We do not know why power-role agents write longer messages in Phase 2 (in Phase 1 the direction is reversed, ROC-AUC 0.353); the claim is descriptive.
*What did not work.* The timing/burst features we expected to matter carry no signal, and XGBoost does not beat one-feature baselines. The Swarm Traces parent/child graph does not help beyond text. These redirect effort: cheap content and mention statistics are the baseline to beat on data like this.
*Track 4 relevance.* The partial-visibility results speak to "a monitor that sees only part of the messages"; the content-versus-structure comparison speaks to monitors that cannot read content. They do not address agents that edit logs, steganography, adversarial agents, or trusted-monitor protocols.

## 7. Limitations and Dual-Use / Ethical Considerations

*Limitations.* (1) AI Village: 133 agents and 26 positives give wide intervals; one run, one model; roles were randomly assigned, so the study measures effects of an assigned role, not emergent collusion or influence; the label boundary is a judgement (sensitivity run above). (2) The placebo is consistent with the design but a direct Phase-2 vs Phase-1 test is inconclusive. (3) Several comparisons were run; we report all baselines and apply Holm correction only to the five single-feature tests per phase. (4) Cross-validation on the same 133 agents is the only evaluation; there is no held-out village. (5) Swarm Traces labels are a proxy: credential ids stand in for "same actor", shared credentials or redaction artefacts could create structure that is not coordination, and the template holdout still allows similar-but-not-identical rows across the split (the random split is only 0.12 macro-F1 easier). We audited the text view for pseudonym leakage and fixed it, but cannot rule out other leakage of credential identity. (6) The reassigned roles are not in the downloaded tables, so the role-reassignment event could not be tested as a labelled experiment. (7) A monitor like ours is trivially evaded by agents that avoid mentions or paraphrase.
*Dual-use.* The work shows which simple statistics reveal influential agents in a swarm's chat. Defensively this supports triage; the same knowledge could in principle help an adversarial swarm shape its logs to look unremarkable. We release no exploit content: Swarm Traces payloads are used only as redacted text with dataset-provided pseudonyms; we do not redistribute either dataset; AI Village data is used under its research terms.
*Process.* Code and drafting were AI-assisted; every number comes from a logged run in `logs/` (the incident description in the Introduction and dataset sizes come from the sprint page and dataset documentation). A unit test caught a timestamp-unit bug in the timing features before any reported run; an adversarial review of this report found the masking leak and the missing content baseline, both addressed above.

## 8. Next Steps

Recover the reassigned roles (computer-use turns or the simulation's git history) to run the reassignment natural experiment properly; targeted rather than random visibility, and an explicit evasion test; replicate on a second village or model; test a monitor that sees content but not names.

## References

- Hammond et al., 2025. Multi-Agent Risks from Advanced AI. arXiv:2502.14143.
- Motwani et al., 2024. Secret Collusion among AI Agents: Multi-Agent Deception via Steganography. NeurIPS 2024.
- Rose et al., 2026. Detecting Multi-Agent Collusion Through Multi-Agent Interpretability. arXiv:2604.01151.
- Greenblatt et al., 2023. AI Control: Improving Safety Despite Intentional Subversion. arXiv:2312.06942 (ICML 2024).
- Swarm Traces, 2026. swarmtraces.org.
- AI Digest, AI Village dataset (`aidigestorg/ai-village`), Hugging Face.
- Apart Research, 2026. Swarm Alignment Hackathon sprint page (incident description).
