"""Run the frozen scientific CLI with explicitly authorized logging-only observer."""
import sys
from pathlib import Path
if __name__=='__main__':
 sys.path.insert(0,'D:/py/DRL2')
 import logging_adapter
 args=sys.argv[1:];out=Path(args[args.index('--output')+1]);steps=int(args[args.index('--steps')+1])
 # Native train requires output directory to be absent; keep precreated logs separately until make_model.
 original_install=logging_adapter.install
 # Delay observer construction until native train has created out; native make_model remains the only model creation.
 import train.t2_mve_sac as tm
 old=tm.make_model
 observer=[]
 def make(*a,**kw):
  tm.make_model=old
  obs=original_install(out,steps);observer.append(obs)
  return tm.make_model(*a,**kw)
 tm.make_model=make
 from experiments.t2 import main
 sys.argv=['experiments.t2',*args]
 try:main()
 finally:
  if observer:observer[0].close()
