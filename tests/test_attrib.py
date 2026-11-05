from swarmwatch.attrib import build_table, group_split, fit_eval, baseline_majority, score

def _rows():
    rows = []
    for c in (1, 2):
        for k in range(12):
            rows.append(dict(id=f"R{c}_{k}", parent_id=None, kind="payload", time_utc=None, tags="",
                             text=f"fetch([CREDENTIAL {c}], [SERVICE {c} URL 1]) token{c*7} variant{k} " + "ab"*c*5))
    return rows

def test_pipeline_end_to_end():
    tab = build_table(_rows(), top_k=2)
    assert len(tab["ids"]) == 24
    tr, te = group_split(tab, 0.25, seed=0, by="random")
    pred = fit_eval(tab, tr, te, groups=("text", "entities", "struct"))
    y = [tab["y"][j] for j in te]
    assert score(y, pred)["acc"] >= 0.9
    assert len(baseline_majority([tab["y"][j] for j in tr], len(te))) == len(te)

def test_cred_masked():
    from swarmwatch.attrib import row_views
    tab = build_table(_rows(), top_k=2)
    text, own, nb, st = row_views(tab, tab["ids"][0])
    assert "CREDENTIAL" not in text and not any(t.startswith("cred") for t in own)

def test_type_only_strips_all_pseudonym_ids():
    from swarmwatch.attrib import type_only
    t = type_only("a [CREDENTIAL 12] b [REDACTED:secret_key:000123] c [ENCODED BLOB 86810] d [SERVICE 3 URL 1]")
    assert "12" not in t and "000123" not in t and "86810" not in t and "[C]" in t and "secret_key" in t
