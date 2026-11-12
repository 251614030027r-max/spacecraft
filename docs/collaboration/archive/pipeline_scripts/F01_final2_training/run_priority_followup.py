"""Additive supervisor: never stop current training/evaluation or rewrite science."""
import os, sys, subprocess, time, json, shutil, zipfile
from pathlib import Path
import run_formal_after_training as f
sys.path.insert(0,str(f.MAIN))
from experiments.v3_stopping import gate_verdict

STATE=f.REC/'PRIORITY_EXECUTION.json'
S={'phase':'priority_followup','workers':{},'events':{},'no_processes_stopped':True,
   'authorization':'2026-10-09 user: 271000 first; retain 262461; replay after learned',
   'science_commit':f.COMMIT}
expected=list(range(271000,271048));active={}
COLLAB=Path(r'C:\Users\35884\Documents\Spacecraft\过程文件\协作\Git工作树')
PACKAGE=f.REC.parent/'交付/271000_early_20261009'
def save():
    S.update(supervisor_pid=os.getpid(),updated_at=f.now())
    tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(S,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,STATE)
def ready(dest,seeds=expected):return all((dest/f'seed_{s}.json').is_file() for s in seeds)
def spawn(name,module,args):
    f.frozen()
    with (f.REC/f'priority_{name}.stdout.log').open('xb') as out,(f.REC/f'priority_{name}.stderr.log').open('xb') as err:
        p=subprocess.Popen([str(f.PY),'-u','-B','-m',module,*map(str,args)],cwd=f.MAIN,env=f.ENV,stdout=out,stderr=err,creationflags=0x08004000)
    S['workers'][name]={'pid':p.pid,'started_at':f.now(),'exit_code':None};active[name]=p;save()
def rows(dest,row,seed=None):
    out={s:f.read(dest/f'seed_{s}.json') for s in expected}
    for s,r in out.items():
        assert r['seed']==s and r['row']==row and r['code_commit']==f.COMMIT and not r['code_dirty']
        assert r.get('max_decisions') is None
        if seed:assert r['model_sha256']==f.sha(f.MAIN/f'logs/final2_{seed}/final_model.zip')
    return out
def publish():
    PACKAGE.mkdir(parents=True,exist_ok=False)
    pure=rows(f.E/'pure','pure');models={}
    for seed in (262460,262462):models[str(seed)]=gate_verdict(pure,rows(f.E/'stopping'/str(seed),'stopping',seed))
    passed=sum(v['passes'] for v in models.values())
    result={'scope':'271000 early official per-model gates, two of three models; not final three-model readout',
            'science_commit':f.COMMIT,'model_count':2,'remaining_model':262461,'models':models,
            'passing_models':passed,'three_model_verdict':'PENDING',
            'threshold_already_met':passed>=2,'function':'experiments.v3_stopping.gate_verdict',
            'coordination_gain':'pending learned rows','problems':[]}
    (PACKAGE/'readout_gates_271000.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    for row,seed in [('pure',None),('stopping',262460),('stopping',262462)]:
        dest=PACKAGE/'raw'/row/(str(seed) if seed else '');dest.mkdir(parents=True)
        source=f.E/row/(str(seed) if seed else '')
        for s in expected:shutil.copy2(source/f'seed_{s}.json',dest/f'seed_{s}.json')
    for seed in (262460,262462):
        dest=PACKAGE/'training'/str(seed);dest.mkdir(parents=True)
        for name in ('manifest.json','train.monitor.csv'):shutil.copy2(f.MAIN/f'logs/final2_{seed}'/name,dest/name)
    report='# 271000提前交付：官方单模型门槛\n\n固定代码f2f8169，工况w2.36_r15，60k最终模型262460/262462，271000–271047各48开局；Pure及两条stopping共144个原始结果随包提供。不等272000，262461训练与评估照常保留。\n\n门槛：干净完成数大于Pure、违规不多于Pure、毁掉Pure成功开局≤2；三个模型至少两个通过。直接调用冻结官方gate_verdict，无新门槛。当前是两模型阶段判定，完整readout需learned及262461；不以缺少第三模型的结果宣布最终失败。\n\n'
    for model,v in models.items():report+=f"- {model}：完成{v['clean_completions']}；违规{v['violation_episodes']}；救回{len(v['rescued'])}；毁掉{len(v['destroyed'])}；通过{v['passes']}。\n"
    report+='\n本次检验无保真问题；模型SHA见manifest及状态。协调增益待learned齐全后补。分析定义与完整源文件见readout_gates_271000.json、raw/、training/。\n'
    (PACKAGE/'README.md').write_text(report,encoding='utf-8')
    hashes={p.relative_to(PACKAGE).as_posix():f.sha(p) for p in PACKAGE.rglob('*') if p.is_file()}
    (PACKAGE/'FILES_SHA256.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf-8')
    assert all(f.sha(PACKAGE/n)==v for n,v in hashes.items())
    S['events']['early_gate_delivery']={'path':str(PACKAGE),'time':f.now(),'passing_models':passed}
    save()
    try:
        def git(*args):return subprocess.check_output(['git','-C',str(COLLAB),*args],stderr=subprocess.STDOUT,encoding='utf-8')
        assert not git('status','--porcelain'),'Collaboration tree dirty; retain local delivery'
        git('pull','--ff-only','origin','collab/spacecraft')
        rel='docs/collaboration/lower/handoffs/final_rerun_271000_early_20261009'
        shutil.copytree(PACKAGE,COLLAB/rel)
        (COLLAB/rel/'.gitattributes').write_text('* -text\n',encoding='utf-8')
        git('add','--',rel);git('commit','-m','docs: early 271000 official gates and independent raw evidence')
        git('push','origin','collab/spacecraft')
        S['events']['early_gate_delivery']['git_commit']=git('rev-parse','HEAD').strip();save()
    except Exception as e:S['events']['early_gate_delivery']['publish_error']=repr(e);save()
def run():
    assert not STATE.exists(),'Existing coordinator; do not duplicate'
    f.frozen();save()
    while True:
        f.frozen()
        for name,p in list(active.items()):
            code=p.poll()
            if code is None:continue
            S['workers'][name]['exit_code']=code;save()
            assert code==0,f'{name} failed; preserve evidence'
            del active[name]
        m=f.read(f.MAIN/'logs/final2_262461/manifest.json')
        if m['status']=='completed' and not f.alive(f.TRAIN_PIDS[262461]) and 'seed461_started' not in S['events']:
            assert m['actual_outer_decisions']==60000
            with zipfile.ZipFile(f.MAIN/'logs/final2_262461/final_model.zip') as z:assert z.testzip() is None
            for row in ('stopping','learned'):
                spawn(f'{row}_262461','experiments.v3_stopping',['evaluate','--run-dir','logs/final2_262461','--row',row,'--seeds','271000-271047,272000-272047','--output-dir',f.E/row/'262461'])
            S['events']['seed461_started']=f.now();save()
        if 'early_gate_delivery' not in S['events'] and all(ready(d) for d in (f.E/'pure',f.E/'stopping/262460',f.E/'stopping/262462')):publish()
        for seed in (262460,262462,262461):
            name=f'replay_271_{seed}'
            if name not in S['workers'] and ready(f.E/'learned'/str(seed)) and ready(f.E/'stopping'/str(seed)):
                rows(f.E/'learned'/str(seed),'learned',seed);rows(f.E/'stopping'/str(seed),'stopping',seed)
                spawn(name,'experiments.stopping_replay',['replay','--run-dir',f'logs/final2_{seed}','--seeds','271000-271047','--formal-dir',f.E/'stopping'/str(seed),'--counterfactual','--output-dir',f.R/str(seed)])
            if name in S['workers'] and S['workers'][name]['exit_code']==0:
                assert ready(f.R/str(seed)) and all(f.read(f.R/str(seed)/f'seed_{s}.json').get('fidelity_ok') is True for s in expected),'Replay fidelity failed'
        for seeds,label in [((262460,262462),'two'),((262460,262461,262462),'three')]:
            name=f'readout_271_{label}'
            if name not in S['workers'] and ready(f.E/'pure') and all(ready(f.E/row/str(s)) for s in seeds for row in ('learned','stopping')):
                args=['readout','--pure',f.E/'pure']
                for row in ('stopping','learned'):
                    for s in seeds:args += ['--'+row,f'{s}={f.E/row/str(s)}']
                args+=['--seeds','271000-271047','--output',f.E/f'readout_271000_{label}.json']
                spawn(name,'experiments.v3_stopping',args)
            if name in S['workers'] and S['workers'][name]['exit_code']==0:
                r=f.read(f.E/f'readout_271000_{label}.json');assert r['verdict']!='FIDELITY_FAIL' and not r['problems']
                S['events'][name]={'path':str(f.E/f'readout_271000_{label}.json'),'scope':label,'verdict':r['verdict']};save()
        full=list(range(271000,271048))+list(range(272000,272048))
        all_eval=all(ready(f.E/row,full) for row in ('pure','nominal')) and all(ready(f.E/row/str(s),full) for row in ('learned','stopping') for s in (262460,262461,262462))
        if all_eval and not active:
            S['phase']='271000_priority_completed_272000_evidence_ready';save();return
        save();time.sleep(300)
if __name__=='__main__':
    try:run()
    except Exception as e:S.update(phase='fault_preserve_evidence',error=repr(e));save();raise
