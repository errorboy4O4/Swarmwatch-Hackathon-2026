import pandas as pd
from swarmwatch.village import power_label, agent_features

def test_power_label():
    assert power_label("Mayor of Port Meridian") == 1 and power_label("god") == 1
    assert power_label("Baker of Old Town") == 0 and power_label("Judge of the Civic Court") == 0

def test_agent_features_graph_and_timing():
    id2name = {"a": "red-alpha", "b": "blue-alpha"}
    cm = pd.DataFrame(dict(agent_speaker_id=["a", "a", "a", "b"], content=["hi @blue-alpha", "x", "y", "z"],
                           t=pd.to_datetime(["2026-08-30 10:00:00", "2026-08-30 10:00:10", "2026-08-30 10:10:00", "2026-08-30 10:20:00"])))
    f = agent_features(cm, ["a", "b"], id2name)
    assert f.loc["b", "in_deg"] == 1 and f.loc["a", "out_deg"] == 1 and f.loc["a", "n_msgs"] == 3
    assert f.loc["a", "frac_fast"] == 0.5
