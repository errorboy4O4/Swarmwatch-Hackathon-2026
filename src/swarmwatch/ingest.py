"""Load Swarm Traces rows and extract pseudonymised entity tokens."""
import gzip, json, re, hashlib

ENTITY_PATTERNS = {
    "cred": re.compile(r"\[CREDENTIAL (\d+)\]"),
    "service": re.compile(r"\[SERVICE (\d+) URL"),
    "proxy": re.compile(r"\[PROXY (\d+) URL"),
    "shortener": re.compile(r"\[SHORTENER-(\d+)-HOST\]"),
    "dest": re.compile(r"\[REDACTED:destination:(\d+)\]"),
    "runtime": re.compile(r"\[REDACTED:runtime_identifier:(\d+)\]"),
    "srcid": re.compile(r"\[REDACTED:source_identifier:(\d+)\]"),
}
_NORM = [
    (re.compile(r"\[REDACTED:[a-z_]+:\d+\]"), "[R]"),
    (re.compile(r"\[[A-Z][A-Za-z -]* \d+\]"), "[P]"),
    (re.compile(r"\[(?:SERVICE|PROXY) \d+ URL \d+\]"), "[U]"),
    (re.compile(r"\d+"), "0"),
]


def load_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def extract_entities(text):
    """Return {entity_type: sorted list of ids} for the pseudonym tokens in text."""
    return {k: sorted(set(p.findall(text))) for k, p in ENTITY_PATTERNS.items()}


def template_hash(text, n=100):
    t = text
    for pat, rep in _NORM:
        t = pat.sub(rep, t)
    return hashlib.md5(t[:n].encode("utf-8")).hexdigest()[:12]
