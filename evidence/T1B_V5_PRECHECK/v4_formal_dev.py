import sys
from pathlib import Path
if __name__ == '__main__':
    import experiments.t1b_v2 as d
    d.train(Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), 0, sys.argv[1], smoke=False, interface='v4')
