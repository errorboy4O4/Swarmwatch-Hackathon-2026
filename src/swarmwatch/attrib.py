"""Credential-group attribution: predict which (masked) credential a row used, from content + infra + graph context.
The credential id is a *proxy* for 'same actor/campaign'; it is masked from all features."""
import re, numpy as np, scipy.sparse as sp
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from .ingest import extract_entities, template_hash

CRED = re.compile(r"\[CREDENTIAL \d+\]")
BRACKET = re.compile(r"\[[^\[\]]*\d[^\[\]]*\]")  # any pseudonym token carrying ids


def type_only(text):
    """Mask credential ids, and strip ids from ALL other bracketed pseudonym tokens (keep their type)."""
    return BRACKET.sub(lambda m: re.sub(r"\d+", "#", m.group(0)), CRED.sub("[C]", text))
GROUPS = ("text", "entities", "graph", "struct")


def build_table(rows, top_k=30, min_rows=1):
    """Rows with exactly one credential id, restricted to the top_k most frequent credentials."""
    by_id = {r["id"]: r for r in rows}
    kids = defaultdict(list)
    for r in rows:
        if r["parent_id"]:
            kids[r["parent_id"]].append(r["id"])
    ents = {r["id"]: extract_entities(r["text"]) for r in rows}
    single = {i: e["cred"][0] for i, e in ents.items() if len(e["cred"]) == 1}
    top = [c for c, n in Counter(single.values()).most_common(top_k) if n >= min_rows]
    ids = [i for i in single if single[i] in top]
    return dict(by_id=by_id, kids=kids, ents=ents, ids=ids, y=[single[i] for i in ids])


def _ent_tokens(e):
    return [f"{k}:{v}" for k, vs in e.items() if k != "cred" for v in vs]


def row_views(tab, i):
    r = tab["by_id"][i]
    text = type_only(r["text"])[:2000]
    own = _ent_tokens(tab["ents"][i])
    nb = []
    if r["parent_id"]:
        nb += ["P" + t for t in _ent_tokens(tab["ents"][r["parent_id"]])]
    for k in tab["kids"].get(i, []):
        nb += ["K" + t for t in _ent_tokens(tab["ents"][k])]
    struct = [f"kind:{r['kind']}", f"haspar:{r['parent_id'] is not None}",
              f"nkids:{min(len(tab['kids'].get(i, [])), 10)}", f"lenb:{int(np.log2(len(r['text']) + 1))}"]
    return text, own, nb, struct


class Featurizer:
    def __init__(self, groups=GROUPS):
        self.groups = groups
        self.tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=200000, sublinear_tf=True)
        self.bags = {g: TfidfVectorizer(analyzer=lambda x: x, min_df=1, sublinear_tf=True) for g in ("entities", "graph", "struct")}

    def _views(self, tab, ids):
        v = [row_views(tab, i) for i in ids]
        return dict(text=[a for a, *_ in v], entities=[b for _, b, _, _ in v],
                    graph=[c for _, _, c, _ in v], struct=[d for *_, d in v])

    def fit_transform(self, tab, ids):
        v = self._views(tab, ids)
        mats = []
        if "text" in self.groups: mats.append(self.tf.fit_transform(v["text"]))
        for g in ("entities", "graph", "struct"):
            if g in self.groups: mats.append(self.bags[g].fit_transform(v[g]))
        return sp.hstack(mats).tocsr()

    def transform(self, tab, ids):
        v = self._views(tab, ids)
        mats = []
        if "text" in self.groups: mats.append(self.tf.transform(v["text"]))
        for g in ("entities", "graph", "struct"):
            if g in self.groups: mats.append(self.bags[g].transform(v[g]))
        return sp.hstack(mats).tocsr()


def group_split(tab, test_frac=0.2, seed=0, by="template"):
    """Hold out whole templates ('template') or random rows ('random')."""
    rng = np.random.RandomState(seed)
    ids = tab["ids"]
    if by == "random":
        mask = rng.rand(len(ids)) < test_frac
    else:
        keys = [template_hash(tab["by_id"][i]["text"]) for i in ids]
        uniq = sorted(set(keys)); rng.shuffle(uniq)
        test_keys = set(uniq[: int(len(uniq) * test_frac)])
        mask = np.array([k in test_keys for k in keys])
    return np.where(~mask)[0], np.where(mask)[0]


def baseline_majority(ytr, nte):
    return [Counter(ytr).most_common(1)[0][0]] * nte


def baseline_template_lookup(tab, tr, te):
    ids, y = tab["ids"], tab["y"]
    maj = Counter(y[j] for j in tr).most_common(1)[0][0]
    by_t = defaultdict(Counter)
    for j in tr: by_t[template_hash(tab["by_id"][ids[j]]["text"])][y[j]] += 1
    return [by_t[template_hash(tab["by_id"][ids[j]]["text"])].most_common(1)[0][0]
            if template_hash(tab["by_id"][ids[j]]["text"]) in by_t else maj for j in te]


def baseline_entity_lookup(tab, tr, te):
    """Predict the credential most often co-occurring with the row's rarest known infra token."""
    ids, y = tab["ids"], tab["y"]
    maj = Counter(y[j] for j in tr).most_common(1)[0][0]
    tok2c = defaultdict(Counter)
    for j in tr:
        for t in _ent_tokens(tab["ents"][ids[j]]): tok2c[t][y[j]] += 1
    out = []
    for j in te:
        toks = [t for t in _ent_tokens(tab["ents"][ids[j]]) if t in tok2c]
        if not toks: out.append(maj); continue
        t = min(toks, key=lambda t: sum(tok2c[t].values()))
        out.append(tok2c[t].most_common(1)[0][0])
    return out


def fit_eval(tab, tr, te, groups=GROUPS, seed=0):
    ids, y = tab["ids"], np.array(tab["y"])
    fz = Featurizer(groups)
    Xtr = fz.fit_transform(tab, [ids[j] for j in tr]); Xte = fz.transform(tab, [ids[j] for j in te])
    clf = LinearSVC(C=0.5, random_state=seed).fit(Xtr, y[tr])
    return clf.predict(Xte)


def score(ytrue, ypred):
    return dict(acc=round(accuracy_score(ytrue, ypred), 4), macro_f1=round(f1_score(ytrue, ypred, average="macro", zero_division=0), 4))
