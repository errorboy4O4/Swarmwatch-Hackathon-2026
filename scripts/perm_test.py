"""Permutation test + bootstrap CI for the in-degree baseline (phase 2 vs phase 1). Usage: python scripts/perm_test.py > logs/perm.json"""
import json, sys
sys.path.insert(0, "src")
import numpy as np
from sklearn.metrics import average_precision_score as ap
from swarmwatch.village import load_village, agent_features, power_label, PHASE2_START
a, cm, roles = load_village(); ags = list(a.id); n = dict(zip(a.id, a.name))
y = np.array([power_label(roles[i]) for i in ags]); rng = np.random.RandomState(0); out = {}
for w, m in (("phase1", cm[cm.t < PHASE2_START]), ("phase2", cm[cm.t >= PHASE2_START])):
    s = agent_features(m, ags, n, ("graph",))["in_deg"].values.astype(float)
    obs = ap(y, s); null = np.array([ap(rng.permutation(y), s) for _ in range(5000)])
    bs = []
    for _ in range(2000):
        i = rng.randint(0, len(y), len(y))
        if y[i].sum(): bs.append(ap(y[i], s[i]))
    out[w] = dict(ap=obs, prevalence=float(y.mean()), perm_p=float((null >= obs).mean()), boot95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
print(json.dumps(out, indent=1))
