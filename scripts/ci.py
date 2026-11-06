"""Bootstrap 95% CIs (resampling agents) for the headline AI Village numbers. Usage: python scripts/ci.py > logs/ci.json
Note: the interval reflects which agents are in the sample; it does not include variation from cross-validation training."""
import json, sys
sys.path.insert(0, "src")
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import average_precision_score as ap
from swarmwatch.village import load_village, agent_features, power_label, PHASE2_START

a, cm, roles = load_village(); ags = list(a.id); n = dict(zip(a.id, a.name))
y = np.array([power_label(roles[i]) for i in ags]); P2 = cm[cm.t >= PHASE2_START]
F = agent_features(P2, ags, n, ("volume", "graph", "timing")).fillna(0)
txt = P2.groupby("agent_speaker_id").content.apply(" ".join).reindex(ags).fillna("")
X = TfidfVectorizer(min_df=3, sublinear_tf=True, max_features=20000).fit_transform(txt)      # same as scripts/village_followup.py
oof = np.zeros((5, len(y)))
for k, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0).split(X, y)):
    oof[k // 5, te] = LogisticRegression(C=5, max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
scores = {"content_tfidf_logreg": oof, "mean_len": F["mean_len"].values[None, :].astype(float), "in_deg": F["in_deg"].values[None, :].astype(float)}
rng = np.random.RandomState(0); out = {"prevalence": float(y.mean()), "n_boot": 2000}
for name, S in scores.items():
    point = float(np.mean([ap(y, s) for s in S])); bs = []
    for _ in range(2000):
        i = rng.randint(0, len(y), len(y))
        if y[i].sum() > 0: bs.append(np.mean([ap(y[i], s[i]) for s in S]))
    out[name] = dict(pr_auc=point, ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
print(json.dumps(out, indent=1))
