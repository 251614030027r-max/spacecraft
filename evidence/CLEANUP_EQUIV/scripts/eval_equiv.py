"""python -B eval_equiv.py ROOT RUN_DIR OUT_DIR [OPENINGS]: evaluate RUN_DIR/final.zip (urgent, audit) in ROOT's code on
OPENINGS (comma-separated; default 294300,294301)."""
import inspect, json, os, sys
from pathlib import Path


def main():
    root, run, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    sys.path.insert(0, root); os.chdir(root)
    from experiments import t2
    out.mkdir(parents=True, exist_ok=True)
    ck = str(run / 'final.zip')
    ops = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else '294300,294301').split(',')]
    for op in ops:
        if 'mission' in inspect.signature(t2.evaluate).parameters:
            t2.eval_job(dict(opening=op, checkpoint=ck, label='E', out=str(out), window=None, mission='urgent', certify=None,
                             audit=True, condition='urgent'))
        else:
            t2.eval_job((op, ck, 'E', str(out), None, 'urgent', None, None, True, 'urgent'))


if __name__ == '__main__':
    main()
