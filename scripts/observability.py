"""Sprint-window experiment (Nov 6): detection vs. observability level on Swarm Traces.
Tiers (what the monitor may see), credential id always masked:
  full      : text + entity tokens + parent/child tokens + row structure
  gateway   : NO text; entity tokens + parent/child tokens + structure (metadata only)
  flat_log  : NO text, NO parent/child links; entity tokens + structure
  tokens    : entity tokens only
NOTE: entity tokens are pseudonym ids extracted from payload text by the dataset's redaction, so 'gateway' is a
metadata-like view, not a real network-egress log.
Usage: python scripts/observability.py > logs/observability.json"""
import json, sys
sys.path.insert(0, "src")
import numpy as np
from swarmwatch.ingest import load_rows
from swarmwatch import attrib as A

TIERS = {"full": A.GROUPS, "gateway": ("entities", "graph", "struct"), "flat_log": ("entities", "struct"), "tokens": ("entities",)}
rows = load_rows("data/swarmtraces/redacted.jsonl.gz")
tab = A.build_table(rows, top_k=30); y = np.array(tab["y"])
out = {"n_rows": len(tab["ids"]), "n_classes": len(set(tab["y"])), "results": {}}
for split in ("template", "random"):
    for seed in (0, 1, 2):
        tr, te = A.group_split(tab, 0.2, seed=seed, by=split); yt = list(y[te])
        r = {"majority": A.score(yt, A.baseline_majority(list(y[tr]), len(te))),
             "entity_lookup": A.score(yt, A.baseline_entity_lookup(tab, tr, te))}
        for name, g in TIERS.items():
            r[name] = A.score(yt, A.fit_eval(tab, tr, te, groups=g, seed=seed))
        out["results"][f"{split}_seed{seed}"] = r
        print(split, seed, {k: v["macro_f1"] for k, v in r.items()}, file=sys.stderr)
print(json.dumps(out, indent=1))
