"""Sprint-window experiment (Nov 6): random vs targeted partial visibility at an EQUAL message budget (AI Village, Phase 2).
Strategies for which messages the monitor gets to see (budget = fraction of the 2,541 Phase-2 messages):
  random      : uniformly random messages
  time_block  : one contiguous block of messages in time
  agent_subset: ALL messages of randomly chosen agents until the budget is reached (unseen agents have no data)
Monitors are scored on all 133 agents; an agent with no visible messages gets no information.
Usage: python scripts/visibility.py > logs/visibility.json"""
import json, sys
sys.path.insert(0, "src")
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import average_precision_score as ap
from swarmwatch.village import load_village, agent_features, power_label, PHASE2_START

a, cm, roles = load_village(); ags = list(a.id); n = dict(zip(a.id, a.name))
y = np.array([power_label(roles[i]) for i in ags])
P2 = cm[cm.t >= PHASE2_START].sort_values("t").reset_index(drop=True); N = len(P2)


def pick(strategy, frac, seed):
    k = int(round(N * frac)); rng = np.random.RandomState(seed)
    if strategy == "random":
        return P2.iloc[np.sort(rng.choice(N, k, replace=False))]
    if strategy == "time_block":
        s = rng.randint(0, N - k + 1); return P2.iloc[s:s + k]
    order = rng.permutation(ags); chosen, cnt = [], 0                     # agent_subset
    sizes = P2.groupby("agent_speaker_id").size().to_dict()
    for ag in order:
        if cnt >= k: break
        chosen.append(ag); cnt += sizes.get(ag, 0)
    return P2[P2.agent_speaker_id.isin(chosen)]


def content_ap(m):
    txt = m.groupby("agent_speaker_id").content.apply(" ".join).reindex(ags).fillna("")
    sc = []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0).split(np.zeros(len(y)), y):
        v = TfidfVectorizer(min_df=2, sublinear_tf=True, max_features=20000).fit(txt.iloc[tr])
        mdl = LogisticRegression(C=5, max_iter=2000).fit(v.transform(txt.iloc[tr]), y[tr])
        s = np.zeros(len(y)); s[te] = mdl.predict_proba(v.transform(txt.iloc[te]))[:, 1]; sc.append((te, s))
    # pooled out-of-fold AP per repeat
    out = []
    for r in range(2):
        s = np.zeros(len(y))
        for te, sv in sc[r * 5:(r + 1) * 5]: s[te] = sv[te]
        out.append(ap(y, s))
    return float(np.mean(out))


res = {"n_phase2_msgs": N, "prevalence": float(y.mean()), "results": {}}
for frac in (0.10, 0.25):
    for strat in ("random", "time_block", "agent_subset"):
        rows = []
        for seed in range(5):
            m = pick(strat, frac, seed)
            F = agent_features(m, ags, n, ("volume", "graph")).fillna(0)
            rows.append(dict(n_msgs_seen=len(m), agents_seen=int(m.agent_speaker_id.nunique()),
                             in_deg=float(ap(y, F["in_deg"].values.astype(float))),
                             mean_len=float(ap(y, F["mean_len"].values.astype(float))),
                             content=content_ap(m)))
        agg = {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))] for k in rows[0]}
        res["results"][f"{int(frac * 100)}%/{strat}"] = agg
        print(frac, strat, {k: round(v[0], 3) for k, v in agg.items()}, file=sys.stderr)
print(json.dumps(res, indent=1))
