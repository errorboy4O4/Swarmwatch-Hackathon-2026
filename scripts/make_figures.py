"""Figures for the report, read from logs/. Usage: python scripts/make_figures.py"""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
V = json.load(open("logs/village.json"))["results"]; F = json.load(open("logs/village_followup.json"))
PREV = 0.195
s1, s2 = F["phase1_single_feature"], F["phase2_single_feature"]
rows = [("Content: TF-IDF logistic regression", F["phase1_tfidf_logreg"]["pr_auc"], F["phase2_tfidf_logreg"]["pr_auc"]),
        ("Content: governance keyword count", F["phase1_keyword_count"]["pr_auc"], F["phase2_keyword_count"]["pr_auc"]),
        ("Mean message length", s1["mean_len"]["pr_auc"], s2["mean_len"]["pr_auc"]),
        ("@-mentions received (in-degree)", s1["in_deg"]["pr_auc"], s2["in_deg"]["pr_auc"]),
        ("XGBoost, all structure features", V["phase1/full/xgb"]["pr_auc"], V["phase2/full/xgb"]["pr_auc"]),
        ("PageRank on mention graph", s1["pagerank"]["pr_auc"], s2["pagerank"]["pr_auc"]),
        ("Message count", s1["n_msgs"]["pr_auc"], s2["n_msgs"]["pr_auc"]),
        ("@-mentions sent (out-degree)", s1["out_deg"]["pr_auc"], s2["out_deg"]["pr_auc"])]
fig, ax = plt.subplots(figsize=(7.2, 3.9)); h = 0.38
for i, (name, p1, p2) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.barh(y + h / 2, p2, h, color="#1f5fa8", label="Phase 2 (roles assigned)" if i == 0 else None)
    ax.barh(y - h / 2, p1, h, color="#b8b8b8", label="Phase 1 (placebo, no roles yet)" if i == 0 else None)
    ax.text(p2 + 0.008, y + h / 2, f"{p2:.2f}", va="center", fontsize=7.5); ax.text(p1 + 0.008, y - h / 2, f"{p1:.2f}", va="center", fontsize=7.5, color="#555")
ax.axvline(PREV, color="#c0392b", ls="--", lw=1); ax.text(PREV + 0.005, len(rows) - 0.35, "chance = 0.195", color="#c0392b", fontsize=7.5)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8); ax.set_xlabel("PR-AUC for finding the 26 power-role agents (higher is better)", fontsize=8)
ax.set_xlim(0, 0.8); ax.legend(fontsize=7.5, loc="lower right", frameon=False); ax.spines[["top", "right"]].set_visible(False); ax.tick_params(axis="x", labelsize=8)
plt.tight_layout(); plt.savefig("report/figs/fig1_monitors.png", dpi=200); plt.close()
fr = [0.05, 0.10, 0.25, 0.50, 1.0]
xg = [V["phase2/vis0.05/xgb"]["pr_auc"], V["phase2/vis0.1/xgb"]["pr_auc"], V["phase2/vis0.25/xgb"]["pr_auc"], V["phase2/vis0.5/xgb"]["pr_auc"], V["phase2/full/xgb"]["pr_auc"]]
ib = [V["phase2/vis0.05/xgb"]["baseline_in_deg_pr_auc"], V["phase2/vis0.1/xgb"]["baseline_in_deg_pr_auc"], V["phase2/vis0.25/xgb"]["baseline_in_deg_pr_auc"], V["phase2/vis0.5/xgb"]["baseline_in_deg_pr_auc"], V["phase2/full/baseline_in_deg"]["pr_auc"]]
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.plot([f * 100 for f in fr], xg, "o-", color="#1f5fa8", label="XGBoost, structure features"); ax.plot([f * 100 for f in fr], ib, "s-", color="#e08a1e", label="In-degree alone")
ax.axhline(PREV, color="#c0392b", ls="--", lw=1, label="chance = 0.195"); ax.set_xlabel("% of Phase-2 messages the monitor sees (random sample)", fontsize=8); ax.set_ylabel("PR-AUC", fontsize=8)
ax.set_ylim(0.15, 0.4); ax.legend(fontsize=7.5, frameon=False); ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=8)
plt.tight_layout(); plt.savefig("report/figs/fig2_visibility.png", dpi=200); plt.close()
print("ok")
