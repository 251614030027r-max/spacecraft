"""Read-only 30-minute throughput estimate. Never changes training or its observer."""
import datetime as dt, json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def main():
 out={'timestamp':dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(),'window_seconds':1800,'stability_definition':'At least 30 min; >=128 accepted decisions; optimizer active; adjacent 15-min rates within 25%. Operational ETA qualification only, never a scientific gate.','seeds':{}}
 for s in (0,1):
  p=R/f'B/TRAIN_e_mve_s{s}/live/progress.jsonl'
  if not p.exists():out['seeds'][str(s)]={'stable':False,'reason':'no progress yet'};continue
  rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
  end=rows[-1];now=end['wall_s'];start=min(rows,key=lambda x:abs(x['wall_s']-max(0,now-1800)));mid=min(rows,key=lambda x:abs(x['wall_s']-max(0,now-900)))
  duration=end['wall_s']-start['wall_s'];delta=end['steps']-start['steps'];rate=delta/duration if duration>0 else 0
  r1=(mid['steps']-start['steps'])/max(1,mid['wall_s']-start['wall_s']);r2=(end['steps']-mid['steps'])/max(1,end['wall_s']-mid['wall_s'])
  stable=duration>=1770 and delta>=128 and start['updates']>0 and min(r1,r2)>0 and abs(r1-r2)/max(r1,r2)<=.25
  out['seeds'][str(s)]={'stable':stable,'latest_record':end['jst'],'completed_steps':end['steps'],'completed_updates':end['updates'],'completed_episodes':end['episodes'],'observed_window_seconds':duration,'observed_decisions':delta,'observed_decisions_per_s':rate,'half_window_rates':[r1,r2],'remaining_seconds_estimate':(4000-end['steps'])/rate if stable else None,'reason':None if stable else 'insufficient or changing throughput; no qualified ETA'}
 qualified=all(x['stable'] for x in out['seeds'].values())
 out['both_seeds_stable']=qualified
 out['training_remaining_seconds_estimate']=max(x['remaining_seconds_estimate'] for x in out['seeds'].values()) if qualified else None
 out['completion_basis']='slower seed; estimate is separate from completed counters'
 (R/'记录/THROUGHPUT_ESTIMATE.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
