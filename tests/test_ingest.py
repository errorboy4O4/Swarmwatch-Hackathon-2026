from swarmwatch.ingest import extract_entities, template_hash

def test_extract_entities():
    t = "x [CREDENTIAL 12] [CREDENTIAL 12] [SERVICE 3 URL 1] https://[REDACTED:destination:000058]"
    e = extract_entities(t)
    assert e["cred"] == ["12"] and e["service"] == ["3"] and e["dest"] == ["000058"]
    assert e["proxy"] == []

def test_template_hash_ignores_ids():
    a = "fetch([CREDENTIAL 1],[REDACTED:destination:000001]) 123"
    b = "fetch([CREDENTIAL 99],[REDACTED:destination:000777]) 456"
    c = "different"
    assert template_hash(a) == template_hash(b) != template_hash(c)
