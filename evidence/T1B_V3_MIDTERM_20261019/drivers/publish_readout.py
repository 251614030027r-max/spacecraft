"""Independent V3 evidence/readout. Frozen-model audit only; no training or true rollouts."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys,json,csv,hashlib,zipfile,shutil,statistics,datetime
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'D:/py/DRL2')
from train.t1_b_v2_sac import FeasibleSupportSACV2
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1];SPACE=ROOT.parents[1];REPO=Path('D:/py/DRL2');FROZEN='e33a0dd7b0bd0d73b7d3ad16561f1dd494d242fa';ARMS=('e_sac','e_tcpf');MARKS=(2000,5000,10000,20000)
def read(p):
    try:return json.loads(Path(p).read_text(encoding='utf-8-sig'))
    except (FileNotFoundError,json.JSONDecodeError):return {}
def rows(p):
    if not Path(p).exists():return []
    return [json.loads(x) for x in Path(p).read_bytes().splitlines(keepends=True) if x.endswith(b'\n')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def csvfile(p,data,fields):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
def summarize_cf(csvpath,modelpath):
    # Recover a complete CSV whose previous summary failed, without replacing it.
    with csvpath.open(encoding='utf-8-sig',newline='') as f:data=list(csv.DictReader(f))
    rr=[{k:float(v) for k,v in r.items()} for r in data]
    return {'n':len(rr),'model_sha256':sha(modelpath),'median_abs_immediate':statistics.median(abs(x['immediate_difference']) for x in rr),'median_abs_bootstrap':statistics.median(abs(x['bootstrap_difference']) for x in rr),'bootstrap_dominates_fraction':statistics.mean(int(abs(x['bootstrap_difference'])>abs(x['immediate_difference'])) for x in rr),'bootstrap_sign_agrees_Y_fraction':statistics.mean(int(np.sign(x['bootstrap_difference'])==np.sign(x['Y'])) for x in rr),'both_successors_committed_fraction':statistics.mean(int(x['alt_has_plan']>.5 and x['fact_has_plan']>.5) for x in rr),'scope':'Existing pairs at first audit snapshot, frozen target critic recomputation; not the original per-update labels.'}
def main():
    state=read(ROOT/'STATE.json');terminal=(ROOT/'记录/PIPELINE_EXIT.json').exists();build=ROOT/'交付构建';build.mkdir(exist_ok=True);raw={};evals=[];models=[];gates={};coupling={};rankings={};pairs={};table=[];decomp=[]
    status=('STOPPED' if state.get('phase','').startswith('stopped') else 'COMPLETE' if terminal else 'RUNNING')
    def evidence(p,n=None):
        if Path(p).is_file():raw[n or Path(p).relative_to(ROOT).as_posix()]=Path(p)
    for folder,pattern in [('输入','*.md'),('脚本','*'),('记录','*.json')]:
        for p in (ROOT/folder).glob(pattern):
            if p.is_file():evidence(p)
    for n in ('pytest.log','contract.log','expert.log','pipeline.err.log'):evidence(ROOT/'记录'/n)
    evidence(ROOT/'contract/CONTRACT.json');evidence(ROOT/'expert/EXPERT_RECEIPT.json')
    truth={};missions={}
    for folder,pattern in [('T1B_V1_AUDIT/candidate_truth_scan','r_*.json'),('T1B_V2_PILOT/scan_292120','scan_*.json')]:
        for p in (REPO/'evidence'/folder).glob(pattern):
            r=read(p);evidence(p,'frozen_truth/'+folder.split('/')[-1]+'/'+p.name)
            if not r.get('skipped') and 'discounted_return' in r:
                op=r['opening'];j=r.get('j0',r.get('first_index'));truth.setdefault(op,{})[j]=r['discounted_return'];missions[op]=r['mission']
    for seed in range(3):
        for arm in ARMS:
            path=ROOT/f'TRAIN_{arm}_s{seed}'
            if not path.exists():continue
            key=f'{arm}_s{seed}';h=read(path/'heartbeat.json');fin=read(path/'finished.json');m=read(ROOT/f'M_{key}.json');gates[key]=m or {'status':'PENDING'}
            table.append({'run':key,'steps':fin.get('steps',h.get('steps')),'critic_updates':fin.get('critic_updates',h.get('critic_updates')),'actor_updates':fin.get('actor_updates',h.get('actor_updates')),'M':m.get('pass','PENDING')})
            evidence(ROOT/f'M_{key}.json')
            learning=read(path/'learning_audit.json') or [];probes=[r.get('applied_gradient_share',0) for r in learning if r.get('phase')=='on'];prefix=rows(path/'model_prefix_releases.jsonl');valid=[r for r in prefix if r.get('gate',{}).get('weight',0)>0];directions={'earlier':0,'later':0,'same':0}
            for r in valid:
                p=r['payload'];d=p['alternative']['tau']-p['factual']['tau'];directions['earlier' if d<0 else 'later' if d>0 else 'same']+=1
            coupling[key]={'closed_pairs':len(prefix),'valid_pairs':len(valid),'directions':directions,'on_probes_including_zero':len(probes),'median_applied_gradient_share':statistics.median(probes) if probes else None,'delta_calibration':read(path/'delta_calibration.json'),'feedback_audit':read(path/'feedback_audit.json'),'label_decomposition':{},'caution':'Labels below are recomputed using each frozen checkpoint target critic on existing released pairs; exact historical per-update targets were not logged.'}
            for n in ('manifest.json','finished.json','heartbeat.json','learning_audit.json','feedback_audit.json','delta_calibration.json'):
                evidence(path/n,f'training/{key}/{n}')
            if terminal:
                for n in ('decisions.jsonl','model_prefix_releases.jsonl'):evidence(path/n,f'training/{key}/{n}')
            for p in (path/'checkpoints').glob('*.zip'):
                try:
                    with zipfile.ZipFile(p) as z:
                        if z.testzip() is not None:continue
                    models.append({'run':key,'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'clock':'Frozen RecordV1 callback checkpoint; saved before replay insertion and updates of current vector step, not a new final model.'})
                except (OSError,zipfile.BadZipFile):continue
            for mark in MARKS:
                audit=read(ROOT/f'audit_{key}_{mark}.json');evidence(ROOT/f'audit_{key}_{mark}.json')
                ss=audit.get('models',{}).get(str(mark),{}).get('per_state',[])
                if ss:
                    scored=[s for s in ss if len(truth.get(s['opening'],{}))==len(s['p'])];correct=sum(s['Q_argmax']==max(truth[s['opening']],key=truth[s['opening']].get) for s in scored);dist={}
                    for s in ss[:8]:
                        typ='urgent' if missions[s['opening']].get('urgent') else 'economy';b=dist.setdefault(typ,{});j=str(s['selected_index']);b[j]=b.get(j,0)+1
                    rankings[f'{key}_{mark}']={'ranked_states':len(scored),'Q_truth_best_agreement_count':correct,'Q_truth_best_agreement_fraction':correct/len(scored) if scored else None,'first8_choice_distribution_by_mission':dist,'first16_choices':{str(s['opening']):s['selected_index'] for s in ss},'truth_caution':'Frozen candidate truth uses fixed V1 teacher continuation. V3 commits, but later decisions after plan loss can still differ; not an exact policy-Q claim.'}
                for p in (ROOT/f'eval_{key}_{mark}').glob('*.json'):
                    r=read(p)
                    if 'opening' not in r:continue
                    d=r.get('first_decision') or {};evals.append({'seed':seed,'arm':arm,'checkpoint':mark,'block':'292140–147' if r['opening']>=292140 else '292120–127','opening':r['opening'],'urgent':r['mission']['urgent'],'D_s':r['mission']['deadline_s'],'B_m_s':r['mission']['budget_m_s'],'event':r['event'],'capture':int(r['event']=='success'),'discounted_return':r['discounted_return'],'first_index':d.get('index'),'first_candidates_s':json.dumps(d.get('tau_candidates')),'time_s':r['time_s'],'delta_v':r['delta_v'],'source':str(p)});evidence(p)
                if terminal:
                    for p in (ROOT/f'eval_{key}_{mark}').glob('*.trace.jsonl'):evidence(p)
                modelpath=path/f'checkpoints/step_{mark}.zip';cfout=ROOT/f'CF_LABEL_{key}_{mark}.csv'
                if arm=='e_tcpf' and valid and ss and (terminal or not cfout.exists() or not (ROOT/f'CF_LABEL_{key}_{mark}.json').exists()):
                    if not cfout.exists():
                        model=FeasibleSupportSACV2.load(modelpath,device='cpu',custom_objects={'learning_rate':1e-4,'lr_schedule':lambda _:1e-4})
                        alt=torch.tensor(np.array([r['payload']['alternative']['observation'] for r in valid],np.float32));fact=torch.tensor(np.array([r['payload']['factual']['observation'] for r in valid],np.float32))
                        with torch.no_grad():va=model.target_value(alt).flatten().numpy();vf=model.target_value(fact).flatten().numpy()
                        rr=[]
                        for k,r in enumerate(valid):
                            a=r['payload']['alternative'];f=r['payload']['factual'];im=a['return']-f['return'];boot=a['discount']*(1-a['done'])*float(va[k])-f['discount']*(1-f['done'])*float(vf[k]);rr.append({'seed':seed,'checkpoint':mark,'slot':r['slot'],'env':r['env'],'add_index':r['add_index'],'gate_weight':r['gate']['weight'],'actual_return':r['actual_return'],'predicted_factual_return':f['return'],'alternative_return':a['return'],'immediate_difference':im,'bootstrap_difference':boot,'Y':im+boot,'alternative_tau':a['tau'],'factual_tau':f['tau'],'V_alt':float(va[k]),'V_fact':float(vf[k]),'alt_has_plan':a['observation'][27],'fact_has_plan':f['observation'][27]})
                        csvfile(cfout,rr,list(rr[0]))
                    if not (ROOT/f'CF_LABEL_{key}_{mark}.json').exists():dump(ROOT/f'CF_LABEL_{key}_{mark}.json',summarize_cf(cfout,modelpath))
                    coupling[key]['label_decomposition'][str(mark)]=read(ROOT/f'CF_LABEL_{key}_{mark}.json');evidence(cfout);evidence(ROOT/f'CF_LABEL_{key}_{mark}.json')
    for seed in range(3):
        for mark in MARKS:
            for block in ('292140–147','292120–127'):
                a={r['opening']:r for r in evals if r['seed']==seed and r['arm']=='e_sac' and r['checkpoint']==mark and r['block']==block};b={r['opening']:r for r in evals if r['seed']==seed and r['arm']=='e_tcpf' and r['checkpoint']==mark and r['block']==block};common=sorted(set(a)&set(b))
                if common:pairs[f's{seed}_{mark}_{block}']={'completed_pairs':len(common),'first_choice_difference_count':sum(a[o]['first_index']!=b[o]['first_index'] for o in common),'return_differences_TCPF_minus_SAC':{str(o):b[o]['discounted_return']-a[o]['discounted_return'] for o in common},'mean_paired_difference':statistics.mean(b[o]['discounted_return']-a[o]['discounted_return'] for o in common)}
    assessment={'state':state,'M':gates,'L':read(ROOT/'L_JUDGMENT.json') or {'status':'PENDING'},'rankings_and_state_dependence':rankings,'paired':pairs,'coupling':coupling,'all_three_research_goals_claimed':False,'cross_seed_inference':'Report exact matched development seeds. No formal test-set or statistical success is presumed.'}
    dump(build/'ASSESSMENT.json',assessment);dump(build/'POLICY_AND_GATE_AUDIT.json',{'M':gates,'L':assessment['L'],'rankings':rankings,'paired':pairs});dump(build/'MPC_COUPLING_AUDIT.json',coupling);dump(build/'MODEL_SHA256.json',models)
    csvfile(build/'SAME_TASK_EVAL.csv',sorted(evals,key=lambda r:(r['seed'],r['checkpoint'],r['arm'],r['opening'])),list(evals[0]) if evals else ['seed','arm','checkpoint','opening','event','discounted_return'])
    stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat();test=read(ROOT/'记录/PYTEST_EXIT.json');contract=read(ROOT/'contract/CONTRACT.json');expert=read(ROOT/'expert/EXPERT_RECEIPT.json')
    admission=f"# V3 ADMISSION\n\n冻结 {FROZEN}。pytest退出码 {test.get('exit_code','PENDING')}（指定13项，原始pytest.log随附）；CONTRACT.passed={contract.get('passed','PENDING')}；原初值动作差={contract.get('init_weights_max_abs_action_diff','PENDING')}。V3专家重新采集 {expert.get('transitions','PENDING')}真实步，interface={expert.get('interface','PENDING')}，SHA256={expert.get('sha256','PENDING')}；所有臂和种子共用该文件，无BC。\n\nM以每臂2k首次检查的累计日志为准：无在用计划时决策、已完成回合平均1–4次决策、TCPF有效配对包含更早/更晚、两臂原8首审至少4个熵>.05；一项失败停止两臂，不调参重跑。结果见POLICY_AND_GATE_AUDIT.json。\n"
    (build/'ADMISSION.md').write_text(admission,encoding='utf-8')
    text=f"# T1_B_V3_ROUND_20261019 实测交付\n\n{status}；时间 {stamp}；阶段 {state.get('phase')}。\n\n"+admission.split('\n\n',1)[1]+f"\nL：seed0 20k原块平均原折扣回报>36.54且至少一个首选非最早；通过才自动运行seed1、2两臂各20k。当前判定：{json.dumps(assessment['L'],ensure_ascii=False)}。停止原因：{json.dumps(state.get('stop_reason'),ensure_ascii=False)}。\n\n|运行|最后持久化步数|critic更新|Actor更新|M|\n|---|---:|---:|---:|---|\n"
    for r in table:text+='|'+ '|'.join(str(r[k]) for k in ('run','steps','critic_updates','actor_updates','M'))+'|\n'
    text+='\n强制停止且没有finished.json时，上表只是最后持久化可核验时钟，不补造未保存的优化次数。\n\n|seed|检查点|臂|开发块|完成/8|捕获|平均原折扣回报|\n|---|---|---|---|---:|---:|---:|\n'
    for seed in range(3):
        for mark in MARKS:
            for arm in ARMS:
                for block in ('292140–147','292120–127'):
                    rr=[r for r in evals if r['seed']==seed and r['checkpoint']==mark and r['arm']==arm and r['block']==block]
                    if rr:text+=f"|{seed}|{mark}|{arm}|{block}|{len(rr)}/8|{sum(r['capture'] for r in rr)}|{statistics.mean(r['discounted_return'] for r in rr):.9f}{'（部分，非整块）' if len(rr)<8 else ''}|\n"
    text+='\n必要基线：原块总是最早7/8、约36.54，固定教师后续的最佳首选8/8、约60.87；扩展总是最早6/8、约16.38，最佳首选6/8、约18.70，扩展不是L门槛。真值逐例原件与固定源码路径、SHA在本轮ZIP中。历史V1两臂原块2k/5k/10k同为7/8、35.582057，扩展10k6/8、15.505363；V2 seed0 2k E-SAC两块7/8、36.754445及6/8、16.176671，E-TCPF为7/8、35.582057及6/8、15.505363；V2因E-SAC5k A1失败停止。cold0k两块4/8、−34.446131与6/8、17.132155；强全程G★两块7/8，37.436581与39.961325。不同接口/后续规则的对照不冒充严格同策略Q。\n\n状态相关首选、Q与真值排序一致率、经济/急迫型分布、两臂逐开局差见POLICY_AND_GATE_AUDIT.json；标签即时/自举分解见MPC_COUPLING_AUDIT.json及逐样本CF_LABEL表。分解使用已存在配对和冻结检查点target critic复算，不伪称历史每次更新标签。训练日志的有效配对、探针梯度份额和同状态策略差用于核实机制，不能单独证明任务收益。\n\n所有任务物理、D/B、奖励、安全、G★、MPC与学习参数均为冻结e33a0dd。M通过不等于L通过，L通过不等于MPC独立收益成立；独立贡献须看同条件三种子消融，负结果与缺项完整保留。原始科学文件留在过程目录，当前ZIP不包含旧ZIP或模型大文件；模型逐个SHA列于MODEL_SHA256.json，远端证据按执行单另行普通推送。\n'
    (build/'RESULT.md').write_text(text,encoding='utf-8');(ROOT/'RESULT.md').write_text(text,encoding='utf-8');(build/'TRAINING_STATUS.md').write_text('# V3实际运行状态\n\n'+status+'\n\n'+json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    for n in ('ASSESSMENT.json','POLICY_AND_GATE_AUDIT.json','MPC_COUPLING_AUDIT.json'):evidence(build/n,'readout/'+n)
    # Snapshot bytes before hash and ZIP; active JSON sources may be atomically replaced.
    frozen={n:p.read_bytes() for n,p in raw.items()};manifest={n:{'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)} for n,b in frozen.items()};archive=build/'RAW_EVIDENCE_V3.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for n,b in frozen.items():z.writestr(n,b)
        z.writestr('MANIFEST.json',json.dumps({'scientific_commit':FROZEN,'files':manifest},ensure_ascii=False,indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for n,h in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==h['sha256']
    latest=SPACE/'上层交付/最新';history=SPACE/'上层交付/历史';marker=ROOT/'记录/FIRST_PUBLICATION.json'
    if not marker.exists():
        moved=[];assert latest.resolve().is_relative_to(SPACE.resolve()) and history.resolve().is_relative_to(SPACE.resolve())
        for p in list(latest.iterdir()):
            assert p.is_file();dest=history/f'BEFORE_T1B_V3_20261019_{p.name}';assert not dest.exists();h=sha(p);shutil.move(str(p),dest);assert sha(dest)==h;moved.append({'source':str(p),'archive':str(dest),'sha256':h})
        dump(marker,{'archived':moved,'jst':stamp})
    names=['RESULT.md','ADMISSION.md','SAME_TASK_EVAL.csv','ASSESSMENT.json','POLICY_AND_GATE_AUDIT.json','MPC_COUPLING_AUDIT.json','MODEL_SHA256.json','TRAINING_STATUS.md','RAW_EVIDENCE_V3.zip']
    for n in names:shutil.copy2(build/n,latest/n)
    receipt={'jst':stamp,'scientific_commit':FROZEN,'status':status,'phase':state.get('phase'),'zip_all_entries_sha_verified':True,'eval_rows':len(evals),'artifacts':{n:sha(latest/n) for n in names}}
    dump(latest/'DELIVERY_RECEIPT.json',receipt);dump(ROOT/'记录/DELIVERY_RECEIPT.json',receipt);(latest/'README.md').write_text('# T1-B V3 当前交付\n\n'+status+'。先读RESULT.md和ADMISSION.md，逐例结果及哈希随附；旧V2已保全归档。\n',encoding='utf-8');print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':
    import msvcrt,time
    # Serialize native watcher and manual repair publications without touching training.
    with (ROOT/'记录/PUBLICATION.lock').open('a+b') as lock:
        lock.seek(0,2)
        if lock.tell()==0:lock.write(b'0');lock.flush()
        while True:
            try:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);break
            except OSError:time.sleep(5)
        try:main()
        finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
