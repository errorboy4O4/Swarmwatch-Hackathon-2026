"""Sprint-window experiment (Nov 6): recover role changes from chat self-identification around the reassignment mistake,
then test whether a chat-content monitor trained BEFORE the mistake follows the original or the new role afterwards.
Usage: python scripts/role_recovery.py > logs/role_recovery.json   (also writes logs/role_audit_sample.json)"""
import json, sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from swarmwatch.village import load_village, power_label
from swarmwatch.roles import build_title_index, self_titles

ag, cm, roles = load_village(); name = dict(zip(ag.id, ag.name)); agents = list(ag.id)
vmap, keys = build_title_index(sorted(set(roles.values())))
T0, T1, TEND = pd.Timestamp("2026-08-30 12:50"), pd.Timestamp("2026-08-30 21:00"), pd.Timestamp("2026-08-30 10:00")
cm2 = cm[cm.t >= TEND].copy()
rows = []
for _, r in cm2.iterrows():
    for t in sorted(self_titles(name[r.agent_speaker_id], r.content, vmap, keys)):
        rows.append((r.agent_speaker_id, r.t, t, r.content[:200].replace("\n", " ")))
S = pd.DataFrame(rows, columns=["a", "t", "title", "head"]); S["orig"] = S.a.map(roles); S["same"] = S.title == S.orig
S["win"] = np.where(S.t < T0, "pre", np.where(S.t < T1, "mistake", "post"))
out = {"msgs_total_phase2": len(cm2)}
out["msg_level"] = {w: dict(n_title_msgs=int(len(g)), frac_original=float(g.same.mean())) for w, g in S.groupby("win")}
# agent level in the mistake window
g = S[S.win == "mistake"].groupby("a").agg(has_orig=("same", "any"), has_other=("same", lambda x: (~x).any()))
cls = pd.Series("no_self_title", index=agents)
cls[g.index[g.has_orig & ~g.has_other]] = "original_only"; cls[g.index[~g.has_orig & g.has_other]] = "other_only"; cls[g.index[g.has_orig & g.has_other]] = "both"
out["agent_level_mistake_window"] = {k: int(v) for k, v in cls.value_counts().items()}
# audit sample (for manual labelling)
oth = S[(~S.same) & (S.win == "mistake")].drop_duplicates(["a", "title"])
aud = oth.sample(20, random_state=11)
json.dump([dict(agent=name[r.a], original=r.orig, flagged_title=r.title, head=r["head"]) for _, r in aud.iterrows()],
          open("logs/role_audit_sample.json", "w"), indent=1)
# manual audit of the 20 sampled flagged (agent, title) pairs, labelled by reading each head (indices into role_audit_sample.json)
AUDIT = {5: "false_positive", 16: "false_positive", 17: "ambiguous"}  # all other sampled indices: clear self-identification as a different role
lab = [AUDIT.get(i, "clear") for i in range(len(aud))]
out["precision_audit"] = dict(flagged_pairs_in_window=int(len(oth)), sample_n=len(aud), clear=lab.count("clear"),
                              ambiguous=lab.count("ambiguous"), false_positive=lab.count("false_positive"))
# flip analysis: agents who sign ONLY with other roles in the mistake window
changed = cls.index[cls == "other_only"]
new_title = {a: S[(S.a == a) & (S.win == "mistake") & (~S.same)].title.value_counts().index[0] for a in changed}
orig_lab = {a: power_label(roles[a]) for a in agents}; new_lab = {a: power_label(new_title[a]) for a in changed}
flips = [a for a in changed if orig_lab[a] != new_lab[a]]
out["changed_agents"] = int(len(changed)); out["flipped_power_status"] = int(len(flips))
out["flip_direction"] = {"power_to_nonpower": int(sum(orig_lab[a] == 1 for a in flips)), "nonpower_to_power": int(sum(orig_lab[a] == 0 for a in flips))}
# content monitor trained on PRE-mistake text (original labels), scored on mistake-window text of held-out agents
txt = lambda lo, hi: cm2[(cm2.t >= lo) & (cm2.t < hi)].groupby("agent_speaker_id").content.apply(" ".join).reindex(agents).fillna("")
pre_txt, win_txt = txt(TEND, T0), txt(T0, T1)
y = np.array([orig_lab[a] for a in agents]); score = np.zeros((5, len(agents)))
for k, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0).split(np.zeros(len(y)), y)):
    v = TfidfVectorizer(min_df=3, sublinear_tf=True, max_features=20000).fit(pre_txt.iloc[tr])
    m = LogisticRegression(C=5, max_iter=2000).fit(v.transform(pre_txt.iloc[tr]), y[tr])
    score[k // 5, te] = m.predict_proba(v.transform(win_txt.iloc[te]))[:, 1]
sc = pd.Series(score.mean(0), index=agents)
unchanged = [a for a in agents if cls[a] == "original_only"]
def auc(idx, lab): 
    l = np.array([lab[a] for a in idx]); return float(roc_auc_score(l, sc[idx].values)) if 0 < l.sum() < len(l) else None
out["monitor_trained_pre_scored_during"] = dict(
    n_original_only_agents=len(unchanged), auc_vs_original_label_original_only_agents=auc(unchanged, orig_lab),
    n_flipped=len(flips), mean_score_flipped_power_to_nonpower=float(sc[[a for a in flips if orig_lab[a] == 1]].mean()) if any(orig_lab[a] == 1 for a in flips) else None,
    mean_score_flipped_nonpower_to_power=float(sc[[a for a in flips if orig_lab[a] == 0]].mean()) if any(orig_lab[a] == 0 for a in flips) else None,
    mean_score_original_only_power=float(sc[[a for a in unchanged if orig_lab[a] == 1]].mean()),
    mean_score_original_only_nonpower=float(sc[[a for a in unchanged if orig_lab[a] == 0]].mean()))
print(json.dumps(out, indent=1))
