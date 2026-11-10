import numpy as np

from env.t2_stage import T2StageEnv
from experiments.t2 import run_episode


def test_audit_does_not_change_a_development_episode():
    plain = T2StageEnv(openings=[294300], shaping=False, obs_profile="cut_priv")
    audit = T2StageEnv(openings=[294300], shaping=False, obs_profile="cut_priv")
    plain.certify = audit.certify = True
    audit.audit_logging = True
    observations = [[], []]

    def action(index):
        def decide(env, obs):
            observations[index].append(obs.copy())
            return np.zeros(2)
        return decide

    a = run_episode(plain, 294300, action(0))
    b = run_episode(audit, 294300, action(1))
    np.testing.assert_array_equal(observations[0], observations[1])
    assert (a["event"], a["return"], a["time_s"]) == (b["event"], b["return"], b["time_s"])
    assert {"budget_m_s", "terminal", "min_margins", "cert_wall_s_total", "cert_decisions"} <= b.keys()
    assert len(a["decisions"]) == len(b["decisions"])
    for old, new in zip(a["decisions"], b["decisions"]):
        assert {"t_start", "remaining_s", "remaining_dv", "phase_rad", "time_to_pass_s",
                "lat_err_deg", "macro_end", "true_mode_event_t"} <= new.keys()
        for key, value in old.items():
            if key == "cert":
                assert {k: new[key][k] for k in value} == value
                assert len(new[key]["cands"]) == 9
            else:
                assert new[key] == value


def test_candidate_audit_reads_cached_rows(monkeypatch):
    env = T2StageEnv(openings=[294300], shaping=False, obs_profile="cut_priv")
    env.audit_logging = True
    env.reset(options={"opening": 294300})
    cache = env._opportunity_eval()
    original = env.cert_prober.probe_intent
    calls = []

    def count(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(env.cert_prober, "probe_intent", count)
    _, audit = env._certify(np.zeros(2))
    assert len(calls) == 1
    for logged, row in zip(audit["cands"], cache["rows"], strict=True):
        assert {key: logged[key] for key in ("event", "kind", "t", "dv", "certified")} == {
            key: row[key] for key in ("event", "kind", "t", "dv", "certified")}
        assert logged["binding"] == (cache["t"] + row["t"] + cache["period"] > cache["latest"])
