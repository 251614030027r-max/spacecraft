"""Pre-registered plateau stop rule of the development watcher."""

from __future__ import annotations

from experiments.t1_dev_eval import plateau_reached


def _s(pairs):
    return {n: {"clean_capture": c, "mean_discounted_return": r} for n, (c, r) in pairs.items()}


def test_never_stops_at_or_before_the_minimum():
    assert not plateau_reached(_s({0: (11, 0.0), 5000: (11, 0.0), 10000: (11, 0.0), 15000: (11, 0.0), 20000: (11, 0.0)}))


def test_two_non_improving_marks_after_the_minimum_stop():
    base = {0: (5, -50.0), 5000: (8, -10.0), 10000: (12, 20.0), 15000: (13, 30.0), 20000: (13, 31.0)}
    assert not plateau_reached(_s(base | {25000: (13, 32.0)}))
    assert plateau_reached(_s(base | {25000: (13, 32.0), 30000: (12, 33.0)}))


def test_a_return_or_capture_improvement_resets_the_count():
    base = {0: (5, -50.0), 5000: (8, -10.0), 10000: (12, 20.0), 15000: (13, 30.0), 20000: (13, 31.0)}
    assert not plateau_reached(_s(base | {25000: (13, 31.0), 30000: (13, 33.5), 35000: (13, 34.0)}))
    assert not plateau_reached(_s(base | {25000: (13, 31.0), 30000: (14, 20.0), 35000: (13, 22.0)}))
    assert plateau_reached(_s(base | {25000: (13, 31.0), 30000: (14, 20.0), 35000: (13, 22.0), 40000: (14, 32.0)}))
