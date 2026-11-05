"""AI Village 'swarm-village' (133 copies, one chat room): can a monitor see who holds governance roles
from the inter-agent chat alone, and how does that degrade when it sees only a sample of messages?
Roles were assigned at the start of Phase 2, so Phase-1 behaviour is a built-in placebo."""
import gzip, json, re
import numpy as np, pandas as pd, networkx as nx

PHASE2_START = pd.Timestamp("2026-08-30 10:00:00")
# Pre-registered BEFORE looking at any features (see STRATEGY.md): governance / state-power roles.
POWER_RE = re.compile(r"^(President|Prime Minister|Deputy Prime Minister|Chief Councilor|Queen|Sultan|"
                      r"Secretary-General|Speaker|Eldest|Mayor|Minister|Chief Justice|Chief of Police|"
                      r"Director of the Central Bank|Ambassador|Tax Collector|god$)")
MENTION = re.compile(r"@([a-z]+-[a-z]+)")


def read_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return pd.DataFrame([json.loads(l) for l in f])


def load_village(d="data/aivillage/swarm-village"):
    ag, cm, gl = (read_jsonl(f"{d}/{n}.jsonl.gz") for n in ("agents", "chat_messages", "agent_goals"))
    cm = cm[cm.speaker_type == "agent"].copy()
    cm["t"] = pd.to_datetime(cm.created_at)
    cm = cm.sort_values("t").reset_index(drop=True)
    p2 = gl[gl.name.str.startswith("Play your assigned role")].copy()
    p2["role"] = p2.name.str.replace("Play your assigned role: ", "", regex=False).str.split("—").str[0].str.strip()
    roles = dict(zip(p2.agent_id, p2.role))
    return ag, cm, roles


def power_label(role):
    return int(bool(POWER_RE.match(role)))


def agent_features(cm, agents, id2name, groups=("volume", "graph", "timing")):
    """Per-agent features from a (possibly partial) message table. Returns DataFrame indexed by agent id."""
    names = set(id2name.values()); name2id = {v: k for k, v in id2name.items()}
    G = nx.DiGraph(); G.add_nodes_from(agents)
    for _, r in cm.iterrows():
        for m in set(MENTION.findall(r.content)):
            if m in names and name2id[m] != r.agent_speaker_id:
                u, v = r.agent_speaker_id, name2id[m]
                w = G[u][v]["weight"] + 1 if G.has_edge(u, v) else 1
                G.add_edge(u, v, weight=w)
    pr = nx.pagerank(G, weight="weight") if G.number_of_edges() else {a: 1 / len(agents) for a in agents}
    rows = {}
    by = {a: g for a, g in cm.groupby("agent_speaker_id")}
    for a in agents:
        g = by.get(a)
        n = 0 if g is None else len(g)
        f = {}
        f["n_msgs"] = n
        f["mean_len"] = 0.0 if n == 0 else g.content.str.len().mean()
        f["out_deg"] = G.out_degree(a); f["in_deg"] = G.in_degree(a)
        f["in_w"] = G.in_degree(a, weight="weight"); f["out_w"] = G.out_degree(a, weight="weight")
        f["pagerank"] = pr[a]
        t = np.sort((g.t - pd.Timestamp("1970-01-01")).dt.total_seconds().values) if n else np.array([])
        gaps = np.diff(t)
        if len(gaps) >= 2:
            mu, sd = gaps.mean(), gaps.std()
            f["gap_median"] = np.median(gaps); f["burstiness"] = (sd - mu) / (sd + mu + 1e-9)
            f["frac_fast"] = float((gaps < 120).mean())
        else:
            f["gap_median"] = np.nan; f["burstiness"] = np.nan; f["frac_fast"] = np.nan
        rows[a] = f
    df = pd.DataFrame.from_dict(rows, orient="index")
    cols = {"volume": ["n_msgs", "mean_len"], "graph": ["out_deg", "in_deg", "in_w", "out_w", "pagerank"],
            "timing": ["gap_median", "burstiness", "frac_fast"]}
    keep = [c for g in groups for c in cols[g]]
    return df[keep]
