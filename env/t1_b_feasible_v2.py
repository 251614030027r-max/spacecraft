"""T1-B V2 interface: the V1 task, verified arrival support, G*, MPC and truth unchanged.

Only the paired MPC alternative changes. V1 always paired the farthest other candidate, so
while a live plan was kept the "delay" side was never paired (2021 valid V1 pairs: 1668 were
"advance 15 s" vs "keep"). V2 draws the alternative uniformly among the other verified
candidates with a fixed hash of (opening, physical review index): no outcome, rank or policy input.
"""
import hashlib

from env.t1_b_feasible_v1 import FeasibleArrivalEnv

VERSION = 't1b-feasible-induced-categorical-sac-v2'


class FeasibleArrivalEnvV2(FeasibleArrivalEnv):
    ALTERNATIVE_RULE = 'uniform among the other verified candidates by sha256(opening, review index); no outcome/rank oracle'

    def alternative_index(self, choices, idx):
        others = [j for j in range(len(choices)) if j != idx]
        key = f'{int(self.current_opening)}:{int(self._review_index)}'.encode()
        return others[int.from_bytes(hashlib.sha256(key).digest()[:4], 'little') % len(others)]
