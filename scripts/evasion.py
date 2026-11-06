"""Sprint-window experiment (Nov 6): how much monitor signal survives simulated evasions of the Phase-2 chat log?
Evasions rewrite the log; monitors are re-scored. Defender modes for the content monitor:
  static   : trained on CLEAN text, scored on evaded text (defender unaware)
  adaptive : trained and scored on evaded text (defender knows the evasion)
Usage: python scripts/evasion.py > logs/evasion.json"""
import json, sys
sys.path.insert(0, "src")
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc
from xgboost import XGBClassifier
from swarmwatch.village import load_village, agent_features, power_label, PHASE2_START
from swarmwatch import evasion as E

a, cm, roles = load_village(); ags = list(a.id); n = dict(zip(a.id, a.name))
y = np.array([power_label(roles[i]) for i in ags]); P2 = cm[cm.t >= PHASE2_START]
mask_roles = E.build_role_masker(sorted(set(roles.values())))
COND = {"clean": lambda t: t, "no_mentions": E.strip_mentions, "no_role_words": mask_roles, "clipped_300": E.clip_length,
        "all_three": E.compose(E.strip_mentions, mask_roles, E.clip_length)}
rng = np.random.RandomState(0)

def perm_p(s, k=2000):
    o = ap(y, s); return o, float((np.array([ap(rng.permutation(y), s) for _ in range(k)]) >= o).mean())

def cv_content(clean_txt, evaded_txt, adaptive):
    sc = np.zeros((5, len(y)))
    for k, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0).split(np.zeros(len(y)), y)):
        fit_txt = evaded_txt if adaptive else clean_txt
        v = TfidfVectorizer(min_df=3, sublinear_tf=True, max_features=20000).fit(fit_txt.iloc[tr])
        m = LogisticRegression(C=5, max_iter=2000).fit(v.transform(fit_txt.iloc[tr]), y[tr])
        sc[k // 5, te] = m.predict_proba(v.transform(evaded_txt.iloc[te]))[:, 1]
    return float(np.mean([ap(y, s) for s in sc])), float(np.std([ap(y, s) for s in sc])), float(np.mean([auc(y, s) for s in sc]))

def cv_xgb(X):
    sc = np.zeros((5, len(y)))
    for k, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0).split(X, y)):
        m = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          random_state=0, eval_metric="logloss", verbosity=0).fit(X[tr], y[tr])
        sc[k // 5, te] = m.predict_proba(X[te])[:, 1]
    return float(np.mean([ap(y, s) for s in sc])), float(np.std([ap(y, s) for s in sc])), float(np.mean([auc(y, s) for s in sc]))

clean_txt = P2.groupby("agent_speaker_id").content.apply(" ".join).reindex(ags).fillna("")
out = {"prevalence": float(y.mean()), "conditions": {}}
for cname, fn in COND.items():
    ev = P2.copy(); ev["content"] = ev.content.map(fn)
    ev_txt = ev.groupby("agent_speaker_id").content.apply(" ".join).reindex(ags).fillna("")
    F = agent_features(ev, ags, n, ("volume", "graph", "timing")).fillna(0)
    r = {}
    for col in ("in_deg", "mean_len"):
        o, p = perm_p(F[col].values.astype(float)); r[col] = dict(pr_auc=o, perm_p=p)
    ap_x, sd_x, au_x = cv_xgb(F.values); r["xgb_structure"] = dict(pr_auc=ap_x, sd=sd_x, roc_auc=au_x)
    for mode in ("static", "adaptive"):
        ap_c, sd_c, au_c = cv_content(clean_txt, ev_txt, mode == "adaptive"); r[f"content_{mode}"] = dict(pr_auc=ap_c, sd=sd_c, roc_auc=au_c)
    out["conditions"][cname] = r
    print(cname, {k: round(v["pr_auc"], 3) for k, v in r.items()}, file=sys.stderr)

# ---- white-box vocabulary masking: adversary knows the monitor and deletes the K most power-predictive words ----
import re
v0 = TfidfVectorizer(min_df=3, sublinear_tf=True, max_features=20000); X0 = v0.fit_transform(clean_txt)
coef = LogisticRegression(C=5, max_iter=2000).fit(X0, y).coef_[0]       # worst case: adversary sees ALL labels
vocab = np.array(v0.get_feature_names_out()); order = np.argsort(-coef)
TOK = re.compile(r"\b\w\w+\b")
out["vocab_masking"] = {}
for K in (20, 100, 500, 2000):
    S = set(vocab[order[:K]])
    mk = lambda t, S=S: TOK.sub(lambda m: "" if m.group(0).lower() in S else m.group(0), t)
    ev_txt = clean_txt.map(mk)
    r = {"static": dict(zip(("pr_auc", "sd", "roc_auc"), cv_content(clean_txt, ev_txt, False))),
         "adaptive": dict(zip(("pr_auc", "sd", "roc_auc"), cv_content(clean_txt, ev_txt, True)))}
    out["vocab_masking"][str(K)] = r
    print("vocab K", K, {k: round(v["pr_auc"], 3) for k, v in r.items()}, file=sys.stderr)
print(json.dumps(out, indent=1))
