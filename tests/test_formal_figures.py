from experiments.formal_figures import select_cases


def test_case_selection_uses_smallest_registered_opening():
    yes, no = {"event": "success"}, {"event": "timeout"}
    ecc = {3: yes, 2: yes, 1: yes}
    sac = {1: no, 2: yes, 3: yes}
    pure = {3: yes, 2: yes}
    nocert = {2: yes, 3: no}
    cases = select_cases(ecc, sac, pure, nocert)
    assert cases["physical"]["opening"] == 2
    assert cases["mechanism"]["opening"] == 3
    fallback = select_cases(ecc, sac, {}, {})
    assert fallback["physical"]["opening"] == fallback["mechanism"]["opening"] == 1
