"""python -B screen_equiv.py ROOT OUT_DIR: fixed policies pure / gstar / rally / c_10_0 / b_10 on 294300 (urgent) and
294301 (economy, budget 2.0) with audit, in ROOT's code."""
import inspect, os, sys
from dataclasses import replace
from pathlib import Path


def main():
    root, out = sys.argv[1], Path(sys.argv[2])
    sys.path.insert(0, root); os.chdir(root)
    from experiments import t2
    try:
        from env.t2_stage import mission_from_name as mission_of
    except ImportError:
        mission_of = t2.mission_profile
    for policy in ('pure', 'gstar', 'rally', 'c_10_0', 'b_10'):
        for op, name, budget in ((294300, 'urgent', None), (294301, 'economy', 2.0)):
            m = mission_of(name)
            if budget is not None:
                m = replace(m, budget_m_s=budget)
            d = out / name; d.mkdir(parents=True, exist_ok=True)
            if 'mission' in inspect.signature(t2.screen).parameters:
                t2.screen_job(dict(opening=op, label=policy, out=str(d), window=True, mission=m, audit=True, condition=name))
            else:
                t2.screen_job((op, policy, str(d), True, m, None, True, name))


if __name__ == '__main__':
    main()
