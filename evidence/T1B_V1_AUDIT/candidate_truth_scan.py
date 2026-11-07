"""Truth value of each first-review candidate on a fixed development opening; continuation = keep the live plan
(earliest when there is no plan), i.e. the frozen V1 teacher. Offline diagnostic, no learning."""
import json, sys, time
import numpy as np
from env.t1_b_task import MissionDistribution
from env.t1_b_feasible_v1 import FeasibleArrivalEnv, bin_center
from env.t1_n_review import N_DUE_INDEX
prof = json.load(open('evidence/T1B_V1/profile_mission_v3.json', encoding='utf-8-sig'))
def run(opening, j0, out):
    env = FeasibleArrivalEnv(openings=[opening], mission=MissionDistribution(**prof)); obs, info = env.reset()
    first = True; done = False; disc = 0.; k = 0; t0 = time.perf_counter(); choices0 = None; dv = 0.
    while not done:
        z = obs; n = int((z[49:52] > .5).sum()); a = 0.
        if z[N_DUE_INDEX] > .5 and n >= 2:
            if first:
                if j0 >= n: json.dump({'opening': opening, 'j0': j0, 'skipped': True, 'n': n}, open(out, 'w')); return
                a = bin_center(j0, n); choices0 = (z[46:46+n]*200).tolist(); first = False
            else:
                choices = z[46:46+n]*200
                j = int(np.argmin(abs(choices - z[28]*200))) if z[27] > .5 else 0
                a = bin_center(j, n)
        obs, r, term, trunc, i = env.step(np.array([a], np.float32)); done = term or trunc
        disc += .999**k * r; k += 1
    json.dump({'opening': opening, 'j0': j0, 'choices0': choices0, 'event': i['event'], 'discounted_return': disc,
               'time_s': env.env.time_s, 'delta_v': env.env._delta_v, 'mission': env.mission, 'wall_s': time.perf_counter()-t0},
              open(out, 'w'), default=float)
if __name__ == '__main__':
    run(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
