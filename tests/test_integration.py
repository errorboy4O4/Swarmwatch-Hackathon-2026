"""End-to-end smoke test on a tiny synthetic fixture (no real data needed):
chat log -> labels -> per-agent features -> monitors -> out-of-fold PR-AUC; plus the evasion rewrites and role extraction.
The synthetic 'power' agents write governance words and receive more @-mentions, so the pipeline must find them."""
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from swarmwatch.village import agent_features, power_label
from swarmwatch import evasion
from swarmwatch.roles import build_title_index, self_titles


def _fixture(n_agents=40, n_power=8, seed=0):
    rng = np.random.RandomState(seed)
    ids = [f"id{i}" for i in range(n_agents)]
    names = {a: f"agent-{chr(97 + i // 26)}{chr(97 + i % 26)}" for i, a in enumerate(ids)}   # letters only, like real names (e.g. ruby-eta)
    titles = {a: ("Minister of Finance" if i < n_power else "Baker of Old Town") for i, a in enumerate(ids)}
    rows = []
    for k in range(600):
        a = ids[rng.randint(n_agents)]
        is_power = titles[a].startswith("Minister")
        # everyone mentions power agents more often; power agents write about laws and votes
        target = ids[rng.randint(n_power)] if rng.rand() < 0.6 else ids[rng.randint(n_agents)]
        text = ("law vote treasury cabinet " if is_power else "bread flour oven market ") + f"@{names[target]} update {k}"
        rows.append(dict(agent_speaker_id=a, content=text, t=pd.Timestamp("2026-08-30 10:00:00") + pd.Timedelta(minutes=k)))
    return ids, names, titles, pd.DataFrame(rows)


def test_pipeline_end_to_end():
    ids, names, titles, cm = _fixture()
    y = np.array([power_label(titles[a]) for a in ids])
    assert y.sum() == 8
    F = agent_features(cm, ids, names, ("volume", "graph", "timing")).fillna(0)
    assert len(F) == 40 and not F.isna().any().any()
    assert average_precision_score(y, F["in_deg"].values) > 2 * y.mean()          # mention structure finds power agents
    txt = cm.groupby("agent_speaker_id").content.apply(" ".join).reindex(ids).fillna("")
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(4, shuffle=True, random_state=0).split(np.zeros(len(y)), y):
        v = TfidfVectorizer().fit(txt.iloc[tr])
        oof[te] = LogisticRegression(max_iter=1000).fit(v.transform(txt.iloc[tr]), y[tr]).predict_proba(v.transform(txt.iloc[te]))[:, 1]
    assert average_precision_score(y, oof) > 0.9                                   # content monitor finds them


def test_evasions_change_what_the_monitors_see():
    ids, names, titles, cm = _fixture()
    y = np.array([power_label(titles[a]) for a in ids])
    ev = cm.copy(); ev["content"] = ev.content.map(evasion.strip_mentions)
    F = agent_features(ev, ids, names, ("graph",))
    assert (F["in_deg"] == 0).all()                                                # no mentions -> no mention signal
    mask = evasion.build_role_masker(sorted(set(titles.values())))
    assert "minister" not in mask("The Minister of Finance spoke").lower()


def test_role_extraction_on_fixture_signature():
    vmap, keys = build_title_index(["Minister of Finance", "Baker of Old Town"])
    assert self_titles("agent-ab", "FINANCE DESK (agent-ab, Minister of Finance): report", vmap, keys) == {"Minister of Finance"}
