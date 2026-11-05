"""EDA on Swarm Traces. Usage: python scripts/eda.py > logs/eda.txt"""
import gzip, json, re, collections, hashlib
import pandas as pd

P = "data/swarmtraces/redacted.jsonl.gz"
rows = [json.loads(l) for l in gzip.open(P, "rt", encoding="utf-8")]
df = pd.DataFrame(rows)
df["len"] = df.text.str.len()
print("rows", len(df), df.kind.value_counts().to_dict())
print("id unique", df.id.is_unique, "cite unique", df.cite.is_unique)

# parent graph
ids = set(df.id)
has_p = df.parent_id.notna()
print("with parent", has_p.sum(), "parent exists in data", df.parent_id[has_p].isin(ids).sum())
kind = df.set_index("id").kind
pk = pd.crosstab(df.kind[has_p].values, df.parent_id[has_p].map(kind).values)
print("child kind (rows) x parent kind (cols)\n", pk)
fan = df.parent_id[has_p].value_counts()
print("children per parent: n parents", len(fan), "max", fan.max(), "median", fan.median(),
      "p99", fan.quantile(.99))

# depth
par = dict(zip(df.id, df.parent_id))
def depth(i):
    d = 0
    while par.get(i) is not None and d < 100:
        i = par[i]; d += 1
    return d
dd = collections.Counter(depth(i) for i in df.id)
print("depth dist", sorted(dd.items())[:10])

# templates: normalise redaction ids & digits, hash
def norm(t):
    t = re.sub(r"\[REDACTED:[a-z_]+:\d+\]", "[R]", t)
    t = re.sub(r"\[(CREDENTIAL|SERVICE \d+ URL|[A-Z ]+) \d+\]", r"[\1]", t)
    t = re.sub(r"\d+", "0", t)
    return t[:400]
df["tpl"] = df.text.map(lambda t: hashlib.md5(norm(t).encode()).hexdigest()[:10])
tc = df.tpl.value_counts()
print("distinct templates", len(tc), "top10 sizes", tc.head(10).tolist())
print("rows in templates of size>=10:", int(tc[tc >= 10].sum()), "singletons", int((tc == 1).sum()))
for k in df.kind.unique():
    s = df[df.kind == k]
    print(k, "len median", s.len.median(), "templates", s.tpl.nunique())

# redaction-token reuse (cross-row linkage signal)
tok = collections.Counter()
for t in df.text:
    for m in set(re.findall(r"\[REDACTED:[a-z_]+:\d+\]", t)):
        tok[m] += 1
print("distinct redaction tokens", len(tok), "top", tok.most_common(8))
print("tokens appearing in >=2 rows", sum(1 for v in tok.values() if v >= 2))
print("redaction type counts", collections.Counter(m.split(":")[1] for m in tok).most_common(10))
print("tagged rows", int((df.tags != "").sum()))
