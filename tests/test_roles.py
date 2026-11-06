from swarmwatch.roles import build_title_index, self_titles

TITLES = ["Mayor of Old Town", "Fire Chief", "Baker of Old Town", "god"]

def test_self_title_near_own_name():
    vmap, keys = build_title_index(TITLES)
    assert self_titles("red-alpha", "FIRE CHIEF desk (red-alpha, day 3) - report", vmap, keys) == {"Fire Chief"}

def test_other_agents_title_far_from_name_not_counted():
    vmap, keys = build_title_index(TITLES)
    text = "red-alpha reporting. " + "x" * 120 + " the Fire Chief said"
    assert self_titles("red-alpha", text, vmap, keys) == set()

def test_god_excluded_and_unrelated_text_empty():
    vmap, keys = build_title_index(TITLES)
    assert self_titles("red-alpha", "red-alpha, godspeed to all, god willing", vmap, keys) == set()
