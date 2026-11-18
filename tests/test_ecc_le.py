"""Long-arrival entry candidate of the opportunity coordination (``docs/ecc2/METHOD_FINAL.md``)."""


def test_long_entry_is_off_by_default_and_only_due_at_the_last_pass():
    from env.t2_stage import STAGE_S, T2StageEnv
    env = T2StageEnv.__new__(T2StageEnv)          # the rule only reads the cached opportunity evaluation and the clock
    assert T2StageEnv.__init__.__defaults__ is not None
    env._opp = {'period': 150.0, 'latest': 120.0}

    class _Clock:
        time_s = 40.0
    env.env = _Clock()
    no_event = {'own_event': 0, 'intervened': False}
    assert env._long_entry_due(no_event, {'time_to_pass_s': 10.0})                  # pass imminent, last before 120 s
    assert not env._long_entry_due(no_event, {'time_to_pass_s': STAGE_S + 5.0})     # pass not within this stage
    assert not env._long_entry_due({'own_event': 1, 'intervened': False}, {'time_to_pass_s': 10.0})
    assert not env._long_entry_due({'own_event': 0, 'intervened': True}, {'time_to_pass_s': 10.0})
    env._opp = {'period': 150.0, 'latest': 400.0}
    assert not env._long_entry_due(no_event, {'time_to_pass_s': 10.0})              # another pass remains
