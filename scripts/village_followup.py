"""Follow-up checks after red-team. Usage: python scripts/village_followup.py > logs/village_followup.json"""
import json, re, sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc
from swarmwatch.village import load_village, agent_features, power_label, PHASE2_START

a, cm, roles = load_village(); ags = list(a.id); n = dict(zip(a.id, a.name))
y = np.array([power_label(roles[i]) for i in ags]); rng = np.random.RandomState(0)
P1, P2 = cm[cm.t < PHASE2_START], cm[cm.t >= PHASE2_START]
out = {}

def perm_p(yv, s, k=5000):
    obs = ap(yv, s); null = np.array([ap(rng.permutation(yv), s) for _ in range(k)]); return obs, float((null >= obs).mean())

# 1) all single-feature baselines, both phases, with Holm correction within each phase
for w, m in (("phase1", P1), ("phase2", P2)):
    F = agent_features(m, ags, n, ("volume", "graph", "timing")).fillna(0)
    rows = {}
    for c in ("n_msgs", "mean_len", "in_deg", "out_deg", "pagerank"):
        o, p = perm_p(y, F[c].values.astype(float)); rows[c] = dict(pr_auc=o, roc_auc=float(auc(y, F[c].values)), perm_p=p)
    order = sorted(rows, key=lambda c: rows[c]["perm_p"])
    for i, c in enumerate(order): rows[c]["holm_p"] = min(1.0, rows[c]["perm_p"] * (len(order) - i))
    out[f"{w}_single_feature"] = rows

# 2) content monitors: keyword count (partly circular with the label regex) and TF-IDF logistic regression
KW = re.compile(r"\b(president|prime minister|minister|mayor|councilor|queen|sultan|secretary-general|speaker|chief justice|chief of police|ambassador|tax collector|central bank|god)\b", re.I)
for w, m in (("phase1", P1), ("phase2", P2)):
    txt = m.groupby("agent_speaker_id").content.apply(" ".join).reindex(ags).fillna("")
    kw = txt.map(lambda t: len(KW.findall(t))).values.astype(float)
    o, p = perm_p(y, kw); out[f"{w}_keyword_count"] = dict(pr_auc=o, perm_p=p)
    X = TfidfVectorizer(min_df=3, sublinear_tf=True, max_features=20000).fit_transform(txt)
    oof = np.zeros((5, len(y)))
    for k, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0).split(X, y)):
        oof[k // 5, te] = LogisticRegression(C=5, max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    out[f"{w}_tfidf_logreg"] = dict(pr_auc=float(np.mean([ap(y, o) for o in oof])), roc_auc=float(np.mean([auc(y, o) for o in oof])))

# 3) direct placebo test: is in-degree AP in phase 2 > phase 1? paired bootstrap over agents + size-matched phase 2
s1 = agent_features(P1, ags, n, ("graph",))["in_deg"].values.astype(float)
s2 = agent_features(P2, ags, n, ("graph",))["in_deg"].values.astype(float)
d = []
for _ in range(2000):
    i = rng.randint(0, len(y), len(y))
    if y[i].sum(): d.append(ap(y[i], s2[i]) - ap(y[i], s1[i]))
out["placebo_diff_in_deg"] = dict(ap_diff=float(ap(y, s2) - ap(y, s1)), boot95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], frac_boot_leq0=float((np.array(d) <= 0).mean()))
sm = []
for seed in range(50):
    m = P2.sample(n=len(P1), random_state=seed)
    sm.append(ap(y, agent_features(m, ags, n, ("graph",))["in_deg"].values.astype(float)))
out["phase2_size_matched_to_phase1"] = dict(n_msgs=len(P1), mean_ap=float(np.mean(sm)), sd=float(np.std(sm)), phase1_ap=float(ap(y, s1)))

# 4) label sensitivity: also count judges and captains as power roles
alt = np.array([int(power_label(roles[i]) or bool(re.match(r"^(Judge|Chief Justice|Captain|City Guard Captain|Admiral|Fire Chief|Harbor Master)", roles[i]))) for i in ags])
o, p = perm_p(alt, s2); out["label_sensitivity_in_deg_phase2"] = dict(n_pos=int(alt.sum()), prevalence=float(alt.mean()), pr_auc=o, perm_p=p)
print(json.dumps(out, indent=1))
