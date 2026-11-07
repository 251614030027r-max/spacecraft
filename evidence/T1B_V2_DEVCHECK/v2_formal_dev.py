"""Sandbox development check with the formal V2 hyper-parameters; only the teacher file differs
(regenerated here with the same rule because the V1 pickle lives on the lower machine)."""
import sys
from pathlib import Path

if __name__ == '__main__':
    import experiments.t1b_v2 as d
    expert = sys.argv[1]; d.EXPERT_SHA = d.sha(expert)
    d.train(Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), 0, expert, smoke=False)
