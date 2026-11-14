#!/bin/bash
set -u
PY=$SCRATCH/venv/bin/python
cd $SCRATCH/u2eq/old && PYTHONPATH=$SCRATCH/u2eq/old $PY -B -m experiments.final train --stage dryrun --config EACH --seed 10 --workers 1 --output $SCRATCH/u2eq/old_EACH > $SCRATCH/u2eq/old_EACH.log 2>&1 &
cd /home/user/spacecraft && PYTHONPATH=/home/user/spacecraft $PY -B -m experiments.final train --stage dryrun --config EACH --seed 10 --workers 1 --output $SCRATCH/u2eq/new_EACH > $SCRATCH/u2eq/new_EACH.log 2>&1 &
cd $SCRATCH/u2eq/old && PYTHONPATH=$SCRATCH/u2eq/old $PY -B -m experiments.final train --stage dryrun --config SAC --seed 10 --workers 1 --output $SCRATCH/u2eq/old_SAC > $SCRATCH/u2eq/old_SAC.log 2>&1 &
cd /home/user/spacecraft && PYTHONPATH=/home/user/spacecraft $PY -B -m experiments.final train --stage dryrun --config SAC --seed 10 --workers 1 --output $SCRATCH/u2eq/new_SAC > $SCRATCH/u2eq/new_SAC.log 2>&1 &
wait
echo done
