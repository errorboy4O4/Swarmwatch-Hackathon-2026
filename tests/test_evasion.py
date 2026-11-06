from swarmwatch.evasion import strip_mentions, build_role_masker, clip_length, compose

def test_strip_mentions():
    assert "@" not in strip_mentions("hi @red-alpha and @blue-beta, ok")

def test_role_masker_masks_titles_and_gov_words_only():
    m = build_role_masker(["Mayor of Old Town", "Baker of Old Town"])
    out = m("The Mayor of Old Town and the prime minister baked bread; Baker said hi")
    assert "mayor" not in out.lower() and "prime minister" not in out.lower() and "baker" not in out.lower() and "bread" in out

def test_clip_and_compose():
    assert len(clip_length("x" * 1000)) == 300
    f = compose(strip_mentions, lambda t: t.upper())
    assert f("hi @a-b yo") == "HI  YO"
