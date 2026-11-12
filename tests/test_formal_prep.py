"""Registered settings for the formal experiment and its development prescreen."""
import itertools

import pytest

from experiments import final


@pytest.mark.parametrize("name,arm,geometry,certify,critic", [
    ("SAC_CERT_LOOP", "e_sac", True, True, 24),
    ("ECC_CERT_LOOP_NOGEO", "e_priv", False, True, 48),
    ("TD3_C", "e_td3", True, False, 24),
])
def test_formal_configuration_manifest(name, arm, geometry, certify, critic):
    final.verify_manifest({
        "arm": arm, "obs_profile": "cut_priv", "window": True, "geometry": geometry,
        "actor_view": 24, "critic_view": critic, "certify": certify,
    }, name)


def test_formal_openings_disjoint_except_paired_resource_condition():
    stages = final.registry()["protocol"]["stages"]
    formal = stages["formal"]
    resource = formal["conditions"].get("resource")
    if resource is not None:
        assert resource["openings"] == formal["conditions"]["economy"]["openings"]
    condition_blocks = [set(final.block(formal["conditions"][key]["openings"]))
                        for key in ("urgent", "economy", "natural")]
    assert all(len(group) == 96 for group in condition_blocks)
    validation = set(final.block(formal["validation"]["openings"]))
    prescreen = set(final.block(stages["formal_prescreen"]["eval_openings"]))
    assert len(prescreen) == 48 and max(prescreen) < 300000
    for lhs, rhs in itertools.combinations([*condition_blocks, validation, prescreen], 2):
        assert lhs.isdisjoint(rhs)
    protected = set().union(*condition_blocks, validation, prescreen)
    assert set(stages) == {"formal", "formal_prescreen"}
    reserved = set(final.block(final.registry()["protocol"]["reserved"]["dev_eval"]))
    assert protected.isdisjoint(reserved)


def test_formal_registry_evaluation_matrix():
    stage = final.stage_of("formal")
    assert stage["steps"] == 5000 and stage["seeds"] == [20, 21, 22]
    assert len(stage["trained"]) == 6
    labels = set(stage["evaluations"])
    assert len(labels) == len(stage["evaluations"])
    assert len(labels) == 18 * len(stage["conditions"]) + 21
    assert stage["conditions"]["resource"]["budget_m_s"] == 2.0
    assert set(final.registry()["trained"]) == set(stage["trained"])


def test_smoke_runs_may_not_use_formal_seeds(tmp_path):
    from argparse import Namespace
    with pytest.raises(SystemExit, match="formal seed"):
        final.cmd_train(Namespace(stage="formal", config="ECC_CERT_LOOP", seed=20, output=str(tmp_path / "x"),
                                  workers=1, smoke_steps=4))
