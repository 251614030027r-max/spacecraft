"""One startup check after OS launch; no training changes."""
import json, math
from pathlib import Path
import runtime as a
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
def main():
 a.frozen();out={'jst':a.now(),'science_SHA':a.SHA,'seeds':{},'restarted_for_scheduling':False}
 proof=a.read(a.R/'记录/NO_HOT_OBSERVER_CHANGE_VERIFIED.json');assert all(v['unchanged'] for v in proof['observers'].values())
 for s in (0,1):
  root=a.R/f'B/TRAIN_e_mve_s{s}';m=a.read(root/'manifest.json');assert m['workers']==2 and m['steps']==4000 and m['seed']==s and m['git_head']==a.SHA
  rows=[json.loads(x) for x in (root/'live/progress.jsonl').read_text(encoding='utf-8').splitlines()];assert len(rows)>1 and any(y['steps']>x['steps'] for x,y in zip(rows,rows[1:]))
  e=EventAccumulator(str(root/'live/tensorboard'));e.Reload();assert 'progress/steps' in e.Tags()['scalars']
  for t in e.Tags()['scalars']:
   v=e.Scalars(t);assert all(math.isfinite(x.value) for x in v) and all(y.step>=x.step for x,y in zip(v,v[1:]))
  assert not (root/'live/LOGGER_ERROR.json').exists()
  out['seeds'][str(s)]={'latest':rows[-1],'snapshots':len(rows),'event_tags':e.Tags()['scalars'],'event_file_bytes':sum(p.stat().st_size for p in (root/'live/tensorboard').glob('events.out*')),'independent_output':str(root)}
 a.dump(a.R/'记录/FIRST_LIVE_GROWTH_VERIFIED.json',out)
 print(json.dumps({'status':'both_running_progress_and_events_verified','jst':a.now(),'steps':{s:x['latest']['steps'] for s,x in out['seeds'].items()}},ensure_ascii=False))
if __name__=='__main__':main()
