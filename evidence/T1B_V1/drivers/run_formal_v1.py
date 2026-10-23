"""Continue only after saved smoke verification; no rerun of baseline/expert."""
import run_pipeline as q
import json,os,hashlib,subprocess
from pathlib import Path
ROOT=q.ROOT
def main():
    old=json.loads((ROOT/'STATE.json').read_text(encoding='utf-8-sig'))
    assert old['phase']=='technical_failure_read_stderr' and not old.get('active')
    check=json.loads((ROOT/'记录/ACTOR_SMOKE_FROZEN_CHECK.json').read_text(encoding='utf-8'));assert check['passed']
    q.dump(ROOT/'记录/PRETRAIN_PIPELINE_STATE_PRESERVED.json',old)
    q.STATE.clear();q.STATE.update(old);q.STATE['prior_pretraining_error']=q.STATE.pop('error',None);q.STATE.update(pid=os.getpid(),phase='freeze_after_pretrain_logging_correction',training_active=False)
    q.dump(ROOT/'记录/SMOKE_RECEIPT.json',{'passed':True,'correction':'Probe logging phase alias corrected before formal training; actual200-step updates/weights/Replay independently checked without learning','evidence':check,'scientific_effectiveness_claim':False})
    paths=['env/t1_b_feasible_v1.py','train/t1_b_feasible_sac.py','experiments/t1b_repair_v1.py']
    q.git('add','--',*paths);q.git('commit','-m','T1B V1 verified arrival support and exact induced SAC with shared expert MPC coupling')
    head=q.git('rev-parse','HEAD');assert not q.git('status','--porcelain')
    q.dump(ROOT/'记录/SCIENCE_FREEZE.json',{'science_commit':head,'base_frozen':'27e8095cd3f5e53b069b083b4df581fb1f0ecbf4','new_modules':{f:hashlib.sha256(Path('D:/py/DRL2',f).read_bytes()).hexdigest() for f in paths},'profile_sha256':hashlib.sha256((ROOT.parent/'通宵研究_20261019/记录/profile_mission_v3.json').read_bytes()).hexdigest(),'expert_sha256':json.loads((ROOT/'结果/expert/EXPERT_RECEIPT.json').read_text())['sha256'],'common_formal_pretrain_correction':'Sparse probes include gradient_step3 as well as0; optimization formulas and all physical code unchanged.'})
    q.STATE['science_commit']=head;q.STATE['training_active']=True
    q.runbatch('matched_seed0_training_5k',[(a,['train','--arm',a,'--steps','5000','--output',str(ROOT/f'结果/TRAIN_{a}_seed0')]) for a in ['e_sac','e_tcpf']])
    q.STATE['training_active']=False;q.save()
    for mark in [2000,5000]:q.runbatch(f'evaluate_predeclared_{mark}',[(f'eval_{a}_{mark}',['eval','--checkpoint',str(ROOT/f'结果/TRAIN_{a}_seed0/checkpoints/step_{mark}.zip'),'--label',f'{a}_rl_{mark}','--output',str(ROOT/f'结果/eval_{a}_{mark}')]) for a in ['e_sac','e_tcpf']])
    q.STATE['phase']='matched_5k_complete_readout_required';q.STATE['next_action']='Read paired5k scientific outcomes; publish V1 evidence; conditional10k only under explicit documented growth requirement.';q.save()
if __name__=='__main__':
    try:main()
    except Exception as e:q.STATE['phase']='technical_failure_read_stderr';q.STATE['error']=repr(e);q.STATE['training_active']=False;q.save();raise
