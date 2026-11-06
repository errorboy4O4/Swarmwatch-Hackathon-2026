"""Operational view of the monitors: if an overseer can only inspect the k highest-scored agents, how many are power-role agents?
Precision@k and recall@k for the content monitor and the two best single structural features (Phase 2, AI Village).
Out-of-fold content scores use the same 5x5 CV as scripts/ci.py. Usage: python scripts/topk.py > logs/topk.json
Random ranking expectation: precision@k = prevalence; recall@k = k / n_agents."""
import json, sys
sys.path.insert(0, "src")
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from swarmwatch.village import load_village, agent_features, power_label, PHASE2_START

a, cm, roles = load_village(); ags = list(a.id); n = dict(zip(a.id, a.name))
y = np.array([power_label(roles[i]) for i in ags]); P2 = cm[cm.t >= PHASE2_START]
F = agent_features(P2, ags, n, ("volume", "graph", "timing")).fillna(0)
txt = P2.groupby("agent_speaker_id").content.apply(" ".join).reindex(ags).fillna("")
X = TfidfVectorizer(min_df=3, sublinear_tf=True, max_features=20000).fit_transform(txt)
oof = np.zeros((5, len(y)))
for k, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0).split(X, y)):
    oof[k // 5, te] = LogisticRegression(C=5, max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
scores = {"content_tfidf_logreg": oof, "mean_len": F["mean_len"].values[None, :].astype(float), "in_deg": F["in_deg"].values[None, :].astype(float)}
KS = (10, 20, 26, 40)
rng = np.random.RandomState(0)
out = {"n_agents": len(y), "n_positive": int(y.sum()), "prevalence": float(y.mean()), "ks": list(KS), "results": {}}
for name, S in scores.items():
    prec = {k: [] for k in KS}
    for s in S:
        # break ties randomly so constant or tied scores (e.g. in-degree) are not given an arbitrary order advantage
        order = np.lexsort((rng.rand(len(s)), -s))
        for k in KS:
            prec[k].append(float(y[order[:k]].mean()))
    out["results"][name] = {str(k): dict(precision=float(np.mean(v)), expected_positives=float(np.mean(v)) * k) for k, v in prec.items()}
print(json.dumps(out, indent=1))
