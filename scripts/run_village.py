"""AI Village experiment. Usage: python scripts/run_village.py > logs/village.json"""
import json, sys
sys.path.insert(0, "src")
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier
from swarmwatch.village import load_village, agent_features, power_label, PHASE2_START

ag, cm, roles = load_village()
agents = list(ag.id); id2name = dict(zip(ag.id, ag.name))
y = np.array([power_label(roles[a]) for a in agents])
print(f"agents {len(agents)} positives {y.sum()} prevalence {y.mean():.3f}", file=sys.stderr)
ALL = ("volume", "graph", "timing")

def cv_scores(X, y, seed=0, repeats=5):
    """Pooled out-of-fold PR-AUC / ROC-AUC per repeat for XGBoost."""
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
    oof = np.zeros((repeats, len(y)))
    for k, (tr, te) in enumerate(rskf.split(X, y)):
        m = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          random_state=seed, eval_metric="logloss", verbosity=0).fit(X[tr], y[tr])
        oof[k // 5, te] = m.predict_proba(X[te])[:, 1]
    ap = [average_precision_score(y, o) for o in oof]; au = [roc_auc_score(y, o) for o in oof]
    return float(np.mean(ap)), float(np.std(ap)), float(np.mean(au))

def baseline(df, col):
    s = df[col].fillna(0).values
    return float(average_precision_score(y, s)), float(roc_auc_score(y, s))

def run(window, p, groups, seed):
    m = cm[cm.t < PHASE2_START] if window == "phase1" else cm[cm.t >= PHASE2_START] if window == "phase2" else cm
    if p < 1: m = m.sample(frac=p, random_state=seed)
    df = agent_features(m, agents, id2name, groups)
    return df, m

out = dict(n_agents=len(agents), n_pos=int(y.sum()), prevalence=float(y.mean()), n_msgs=dict(
    phase1=int((cm.t < PHASE2_START).sum()), phase2=int((cm.t >= PHASE2_START).sum())), results={})
R = out["results"]
for window in ("phase1", "phase2"):               # phase1 = placebo (roles not yet assigned)
    df, _ = run(window, 1.0, ALL, 0)
    ap, sd, au = cv_scores(df.values, y)
    R[f"{window}/full/xgb"] = dict(pr_auc=ap, pr_auc_sd=sd, roc_auc=au)
    for col in ("n_msgs", "in_deg", "pagerank"):
        a, u = baseline(df, col); R[f"{window}/full/baseline_{col}"] = dict(pr_auc=a, roc_auc=u)
    print(window, R[f"{window}/full/xgb"], file=sys.stderr)
for g in ALL:                                      # ablations on phase2
    df, _ = run("phase2", 1.0, tuple(x for x in ALL if x != g), 0)
    ap, sd, au = cv_scores(df.values, y); R[f"phase2/minus_{g}/xgb"] = dict(pr_auc=ap, pr_auc_sd=sd, roc_auc=au)
    df, _ = run("phase2", 1.0, (g,), 0)
    ap, sd, au = cv_scores(df.values, y); R[f"phase2/only_{g}/xgb"] = dict(pr_auc=ap, pr_auc_sd=sd, roc_auc=au)
for p in (0.5, 0.25, 0.1, 0.05):                   # partial visibility, phase2
    aps, aus, bl = [], [], []
    for seed in range(5):
        df, _ = run("phase2", p, ALL, seed)
        ap, _, au = cv_scores(df.values, y, seed=seed, repeats=2)
        aps.append(ap); aus.append(au); bl.append(baseline(df, "in_deg")[0])
    R[f"phase2/vis{p}/xgb"] = dict(pr_auc=float(np.mean(aps)), pr_auc_sd=float(np.std(aps)), roc_auc=float(np.mean(aus)),
                                   baseline_in_deg_pr_auc=float(np.mean(bl)))
    print("vis", p, R[f"phase2/vis{p}/xgb"], file=sys.stderr)
print(json.dumps(out, indent=1))
