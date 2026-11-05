"""Main experiment. Usage: python scripts/run_experiment.py > logs/experiment.json"""
import json, sys, itertools
sys.path.insert(0, "src")
import numpy as np
from swarmwatch.ingest import load_rows
from swarmwatch import attrib as A

TOP_K = 30
rows = load_rows("data/swarmtraces/redacted.jsonl.gz")
tab = A.build_table(rows, top_k=TOP_K)
y = np.array(tab["y"])
out = dict(n_rows=len(tab["ids"]), n_classes=len(set(tab["y"])), top_k=TOP_K, results={})
for split in ("template", "random"):
    for seed in (0, 1, 2):
        tr, te = A.group_split(tab, 0.2, seed=seed, by=split)
        yt, ytr = list(y[te]), list(y[tr])
        r = {"majority": A.score(yt, A.baseline_majority(ytr, len(te))),
             "template_lookup": A.score(yt, A.baseline_template_lookup(tab, tr, te)),
             "entity_lookup": A.score(yt, A.baseline_entity_lookup(tab, tr, te)),
             "full_model": A.score(yt, A.fit_eval(tab, tr, te, seed=seed))}
        for g in A.GROUPS:  # leave-one-group-out
            r[f"minus_{g}"] = A.score(yt, A.fit_eval(tab, tr, te, groups=tuple(x for x in A.GROUPS if x != g), seed=seed))
        for g in A.GROUPS:  # single group
            r[f"only_{g}"] = A.score(yt, A.fit_eval(tab, tr, te, groups=(g,), seed=seed))
        out["results"][f"{split}_seed{seed}"] = r
        print(split, seed, json.dumps(r["full_model"]), file=sys.stderr)
# partial visibility: train on a fraction of training rows (template split, seed 0)
tr, te = A.group_split(tab, 0.2, seed=0, by="template")
rng = np.random.RandomState(0)
pv = {}
for frac in (0.01, 0.05, 0.1, 0.25, 0.5, 1.0):
    sub = np.sort(rng.choice(tr, max(int(len(tr) * frac), 50), replace=False))
    pv[str(frac)] = A.score(list(y[te]), A.fit_eval(tab, sub, te))
out["partial_visibility"] = pv
print(json.dumps(out, indent=1))
