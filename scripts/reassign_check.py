"""Phase-2 split around the ~5:50am PT (12:50 UTC) role-reassignment mistake. Labels = ORIGINAL 10:00 roles.
Usage: python scripts/reassign_check.py > logs/reassign.json"""
import json, sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as ap
from swarmwatch.village import load_village, agent_features, power_label, PHASE2_START
a, cm, roles = load_village(); ags = list(a.id); n = dict(zip(a.id, a.name))
y = np.array([power_label(roles[i]) for i in ags]); rng = np.random.RandomState(0)
T = pd.Timestamp("2026-08-30 12:50:00")
out = {"prevalence": float(y.mean())}
for w, m in (("pre_mistake_10:00-12:50", cm[(cm.t >= PHASE2_START) & (cm.t < T)]), ("mistake_window_12:50+", cm[cm.t >= T])):
    r = {"n_messages": len(m)}
    for col, g in (("in_deg", "graph"), ("n_msgs", "volume")):
        s = agent_features(m, ags, n, (g,))[col].values.astype(float)
        obs = ap(y, s); null = np.array([ap(rng.permutation(y), s) for _ in range(3000)])
        r[col] = dict(pr_auc=float(obs), perm_p=float((null >= obs).mean()))
    out[w] = r
print(json.dumps(out, indent=1))
