"""Recover the role an agent *signs as* from its chat messages (the reassigned roles are not in the tables).
Heuristic: a role title counts as a self-identification when it appears within a small window around the
speaker's own name in the first 260 characters of the message. Noisy by design; see report for precision audit."""
import re
from collections import defaultdict


def build_title_index(titles):
    """Map lowercase title (and a short head form, e.g. 'Mayor') -> set of full titles. 'god' is excluded (substring noise)."""
    vmap = defaultdict(set)
    for t in titles:
        if t == "god":
            continue
        vmap[t.lower()].add(t)
        head = re.split(r" of | — ", t)[0]
        if len(head) > 8:
            vmap[head.lower()].add(t)
    # keep only unambiguous keys, longest first
    keys = [(v, re.compile(r"\b" + re.escape(v) + r"\b")) for v in sorted(vmap, key=len, reverse=True) if len(vmap[v]) == 1]
    return vmap, keys


def self_titles(name, text, vmap, keys, before=45, after=55, head=260):
    """Set of full role titles found near the speaker's own name in the message head."""
    t = text[:head].lower()
    found = set()
    for m in re.finditer(re.escape(name.lower()), t):
        win = t[max(0, m.start() - before): m.end() + after]
        for v, rx in keys:
            if rx.search(win):
                found |= vmap[v]
    return found
