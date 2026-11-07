"""Independent V3 evidence/readout. Frozen-model audit only; no training or true rollouts."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys,json,csv,hashlib,zipfile,shutil,statistics,datetime
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'D:/py/DRL2')
from train.t1_b_v4_sac import FeasibleSupportSACV4 as FeasibleSupportSACV2
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1];SPACE=ROOT.parents[1];REPO=Path('D:/py/DRL2');FROZEN='8684cca3d3c3a2997f3ed3f2c106c247b94e20e4';ARMS=('e_sac','e_tcpf');MARKS=(2000,5000,10000,20000)
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
    for n in ('Q_POLICY_SCAN_2K_5K.csv','BEHAVIOUR_COVERAGE_1K.csv','REPLAY_COVERAGE_5K.json','HOST_CRASH_REPORT.md','HOST_CRASH_VERIFICATION_20261020.json'):
        evidence(ROOT/'记录'/n,'review/'+n)
    evidence(ROOT/'contract/CONTRACT.json');evidence(ROOT.parent/'T1B_V3_ROUND_20261019/expert/EXPERT_RECEIPT.json','expert/EXPERT_RECEIPT.json')
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
            key=f'{arm}_s{seed}';h=read(path/'heartbeat.json');fin=read(path/'finished.json');m=read(ROOT/f'M_{key}.json');gates[key]=m or {'status':'PENDING' if seed==0 else 'NOT_GATED: V4 gate is seed0 only'}
            table.append({'run':key,'steps':fin.get('steps',h.get('steps')),'critic_updates':fin.get('critic_updates',h.get('critic_updates')),'actor_updates':fin.get('actor_updates',h.get('actor_updates')),'M':m.get('pass','PENDING' if seed==0 else 'NOT_GATED')})
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
                    scored=[s for s in ss if len(s['p'])==3 and len(truth.get(s['opening'],{}))==3];correct=sum(s['Q_argmax']==max(truth[s['opening']],key=truth[s['opening']].get) for s in scored);dist={}
                    for s in [x for x in ss if len(x['p'])==3]:
                        typ='urgent' if missions[s['opening']].get('urgent') else 'economy';b=dist.setdefault(typ,{});j=str(s['selected_index']);b[j]=b.get(j,0)+1
                    rankings[f'{key}_{mark}']={'ranked_states':len(scored),'Q_truth_best_agreement_count':correct,'Q_truth_best_agreement_fraction':correct/len(scored) if scored else None,'three_candidate_choice_distribution_by_mission':dist,'first16_choices':{str(s['opening']):s['selected_index'] for s in ss},'truth_caution':'Frozen candidate truth uses fixed V1 teacher continuation. V3 commits, but later decisions after plan loss can still differ; not an exact policy-Q claim.'}
                for p in (ROOT/f'eval_{key}_{mark}').glob('*.json'):
                    r=read(p)
                    if 'opening' not in r:continue
                    d=r.get('first_decision') or {};evals.append({'seed':seed,'arm':arm,'checkpoint':mark,'block':'292140–147' if r['opening']>=292140 else '292120–127','opening':r['opening'],'urgent':r['mission']['urgent'],'D_s':r['mission']['deadline_s'],'B_m_s':r['mission']['budget_m_s'],'event':r['event'],'capture':int(r['event']=='success'),'discounted_return':r['discounted_return'],'first_index':d.get('index'),'first_candidates_s':json.dumps(d.get('tau_candidates')),'time_s':r['time_s'],'delta_v':r['delta_v'],'source':str(p)});evidence(p)
                if terminal:
                    for p in (ROOT/f'eval_{key}_{mark}').glob('*.trace.jsonl'):evidence(p)
                modelpath=path/f'checkpoints/step_{mark}.zip';cfout=ROOT/f'CF_LABEL_{key}_{mark}.csv'
                if arm=='e_tcpf' and valid and ss and (terminal or not cfout.exists() or not (ROOT/f'CF_LABEL_{key}_{mark}.json').exists()):
                    if not cfout.exists():
                        model=FeasibleSupportSACV2.load(modelpath,device='cpu',custom_objects={'learning_rate':1e-4,'lr_schedule':lambda _:1e-4,'buffer_size':4})
                        alt=torch.tensor(np.array([r['payload']['alternative']['observation'] for r in valid],np.float32));fact=torch.tensor(np.array([r['payload']['factual']['observation'] for r in valid],np.float32))
                        with torch.no_grad():va=model.target_value(alt).flatten().numpy();vf=model.target_value(fact).flatten().numpy()
                        rr=[]
                        for k,r in enumerate(valid):
                            a=r['payload']['alternative'];f=r['payload']['factual'];im=a['return']-f['return'];boot=a['discount']*(1-a['done'])*float(va[k])-f['discount']*(1-f['done'])*float(vf[k]);rr.append({'seed':seed,'checkpoint':mark,'cf_slot':int(r.get('cf_slot',0)),'slot':r['slot'],'env':r['env'],'add_index':r['add_index'],'gate_weight':r['gate']['weight'],'actual_return':r['actual_return'],'predicted_factual_return':f['return'],'alternative_return':a['return'],'immediate_difference':im,'bootstrap_difference':boot,'Y':im+boot,'alternative_tau':a['tau'],'factual_tau':f['tau'],'V_alt':float(va[k]),'V_fact':float(vf[k]),'alt_has_plan':a['observation'][27],'fact_has_plan':f['observation'][27]})
                        csvfile(cfout,rr,list(rr[0]))
                    if not (ROOT/f'CF_LABEL_{key}_{mark}.json').exists():dump(ROOT/f'CF_LABEL_{key}_{mark}.json',summarize_cf(cfout,modelpath))
                    coupling[key]['label_decomposition'][str(mark)]=read(ROOT/f'CF_LABEL_{key}_{mark}.json');evidence(cfout);evidence(ROOT/f'CF_LABEL_{key}_{mark}.json')
    for seed in range(3):
        for mark in MARKS:
            for block in ('292140–147','292120–127'):
                a={r['opening']:r for r in evals if r['seed']==seed and r['arm']=='e_sac' and r['checkpoint']==mark and r['block']==block};b={r['opening']:r for r in evals if r['seed']==seed and r['arm']=='e_tcpf' and r['checkpoint']==mark and r['block']==block};common=sorted(set(a)&set(b))
                if common:pairs[f's{seed}_{mark}_{block}']={'completed_pairs':len(common),'first_choice_difference_count':sum(a[o]['first_index']!=b[o]['first_index'] for o in common),'return_differences_TCPF_minus_SAC':{str(o):b[o]['discounted_return']-a[o]['discounted_return'] for o in common},'mean_paired_difference':statistics.mean(b[o]['discounted_return']-a[o]['discounted_return'] for o in common)}
    for key,c in coupling.items():
        pp=rows(ROOT/f'TRAIN_{key}/model_prefix_releases.jsonl');vv=[x for x in pp if x.get('gate',{}).get('weight',0)>0]
        c['valid_by_cf_slot']={str(k):sum(x.get('cf_slot')==k for x in vv) for k in (0,1)}
    for key,r in rankings.items():
        ss=read(ROOT/f'audit_{key}.json').get('models',{}).get(key.rsplit('_',1)[1],{}).get('per_state',[])
        slopes=[{'opening':x['opening'],'Q_per_candidate_step':np.diff(x['Q_min']).tolist(),'mean_Q_per_candidate_step':float(np.diff(x['Q_min']).mean()),'urgent':missions[x['opening']]['urgent']} for x in ss if len(x['p'])==3]
        r['Q_slopes_15_states']=slopes;r['mean_Q_step']=statistics.mean(x['mean_Q_per_candidate_step'] for x in slopes) if slopes else None
    aggregate={}
    for mark in MARKS:
        for block in ('292140–147','292120–127'):
            pp={str(s):pairs.get(f's{s}_{mark}_{block}') for s in range(3)}
            for s,p in pp.items():
                if p:
                    a=[r for r in evals if r['seed']==int(s) and r['arm']=='e_sac' and r['checkpoint']==mark and r['block']==block];b=[r for r in evals if r['seed']==int(s) and r['arm']=='e_tcpf' and r['checkpoint']==mark and r['block']==block]
                    p['capture_difference_TCPF_minus_SAC']=sum(r['capture'] for r in b)-sum(r['capture'] for r in a)
                    ra=rankings.get(f'e_sac_s{s}_{mark}',{});rb=rankings.get(f'e_tcpf_s{s}_{mark}',{})
                    p['truth_agreement_count_difference_15states']=rb.get('Q_truth_best_agreement_count',0)-ra.get('Q_truth_best_agreement_count',0)
            full=all(p and p['completed_pairs']==8 for p in pp.values());aggregate[f'{mark}_{block}']={'seedwise':pp,'all_three_complete':full,'mean_return_difference_3seeds':statistics.mean(p['mean_paired_difference'] for p in pp.values()) if full else None,'mean_capture_difference_3seeds':statistics.mean(p['capture_difference_TCPF_minus_SAC'] for p in pp.values()) if full else None}
    zero=[]
    for p in (ROOT/'eval_shared_0k').glob('*.json'):
        z=read(p)
        if 'opening' in z:zero.append(z);evidence(p)
    if terminal:
        for p in (ROOT/'eval_shared_0k').glob('*.trace.jsonl'):evidence(p)
    dump(build/'THREE_SEED_COMPARISON.json',aggregate);dump(build/'SHARED_0K.json',zero)
    assessment={'state':state,'M':gates,'scope':'Six runs authorized from start; only seed0 M gate; no V3 L gate','rankings_and_state_dependence':rankings,'paired':pairs,'three_seed_comparison':aggregate,'shared_0k':zero,'coupling':coupling,'all_three_research_goals_claimed':False,'cross_seed_inference':'Report exact matched development seeds. No formal test-set or statistical success is presumed.'}
    dump(build/'ASSESSMENT.json',assessment);dump(build/'POLICY_AND_GATE_AUDIT.json',{'M':gates,'scope':assessment['scope'],'rankings':rankings,'paired':pairs});dump(build/'MPC_COUPLING_AUDIT.json',coupling);dump(build/'MODEL_SHA256.json',models)
    csvfile(build/'SAME_TASK_EVAL.csv',sorted(evals,key=lambda r:(r['seed'],r['checkpoint'],r['arm'],r['opening'])),list(evals[0]) if evals else ['seed','arm','checkpoint','opening','event','discounted_return'])
    stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat();contract=read(ROOT/'contract/CONTRACT.json');expert=read(ROOT.parent/'T1B_V3_ROUND_20261019/expert/EXPERT_RECEIPT.json')
    admission=f"# V4 ADMISSION\n\n冻结{FROZEN}。指定14项测试退出码{read(ROOT/'记录/PYTEST_EXIT.json').get('exit_code','PENDING')}；合同passed={contract.get('passed')}，epsilon={contract.get('behaviour_epsilon')}，alpha={contract.get('candidate_alpha')}，初始动作差={contract.get('init_weights_max_abs_action_diff')}。共享V3专家1000步，SHA={expert.get('sha256')}，没有重采。\n\n仅seed0在2k检查M：学习决策无在用计划；500步后非最早≥30%；TCPF cf_slot0和1各自有效配对>0。任一失败停止全部6运行，不调参重跑；没有V3的20k L准入条件，不继承V2熵/覆盖门槛。\n"
    (build/'ADMISSION.md').write_text(admission,encoding='utf-8')
    text=f"# T1-B V4资源故障后技术restart1实测交付\n\n{status}；{stamp}；阶段{state.get('phase')}。\n\n"+admission.split('\n\n',1)[1]+f"\n停止原因：{json.dumps(state.get('stop_reason'),ensure_ascii=False)}。\n\n|运行|最后持久化步数|critic更新|Actor更新|M|\n|---|---:|---:|---:|---|\n"
    for r in table:text+='|'+'|'.join(str(r[k]) for k in ('run','steps','critic_updates','actor_updates','M'))+'|\n'
    text+='\n无finished.json时，时钟仅为最后持久化值，不造精确终止步数。\n\n|seed|步数|臂|块|完成/8|捕获|平均原折扣回报|首选非最早|无学习首选|\n|---|---|---|---|---:|---:|---:|---:|---:|\n'
    for seed in range(3):
        for mark in MARKS:
            for arm in ARMS:
                for block in ('292140–147','292120–127'):
                    rr=[r for r in evals if r['seed']==seed and r['checkpoint']==mark and r['arm']==arm and r['block']==block]
                    if rr:text+=f"|{seed}|{mark}|{arm}|{block}|{len(rr)}/8|{sum(r['capture'] for r in rr)}|{statistics.mean(r['discounted_return'] for r in rr):.9f}{'（部分）' if len(rr)<8 else ''}|{sum(r['first_index'] is not None and r['first_index']!=0 for r in rr)}|{sum(r['first_index'] is None for r in rr)}|\n"
    text+='\n共享0k使用seed0 E-SAC原step0模型在相同V4接口确定性执行；全部运行同一INIT SHA、初始网络与确定性评估，行为epsilon只用于训练，不用旧接口0k代替。16例逐原件在SHARED_0K.json与ZIP，未完成部分明确保留，不冒充整块。\n'
    for block,ops in [('main',range(292140,292148)),('expanded',range(292120,292128))]:
        z=[r for r in zero if r['opening'] in ops]
        if z:text+=f"\n0k {block}: {len(z)}/8已完成，捕获{sum(r['event']=='success' for r in z)}，平均折扣回报{statistics.mean(r['discounted_return'] for r in z):.9f}（不足8仅部分均值）。\n"
    text+='\n必要对照：主块总是最早7/8、约36.54，固定教师后续最佳首选8/8、约60.87；扩展6/8、约16.38与6/8、约18.70。V3本机2k M通过，10k两臂主块均7/8、36.538164，扩展6/8、16.220946，主块首选均最早，10k原8熵达标0/8；按用户新V4命令终止于最后持久化14400/12300，V3不做20k L、不扩种子。V3中期科学b2a65c5、collab b8e96e8，原始失败证据完整保留。\n\n15个三候选状态的Q真值排序一致数、每档Q斜率、经济型/急迫型首选分布见POLICY_AND_GATE_AUDIT；各seed各检查点TCPF−SAC回报/捕获/一致数差与三种子均值见THREE_SEED_COMPARISON，三种子未齐时均值为null。所有开局保留，不挑成功样本。固定教师后续真值仍不等于当前SAC严格Q，尤其计划丢失后策略可能不同。\n\nMPC各cf_slot有效配对、全部on探针含0的梯度份额中位数、Y即时/自举分解见MPC_COUPLING_AUDIT和CF_LABEL逐样本CSV。分解使用已留存配对与冻结模型target critic复算，非历史每次训练更新原标签。模型/梯度/技术通过不能单独证明科研要求；只有同条件三种子结果才能评价独立收益，负结果与缺项如实交付。\n\n物理、任务D/B、奖励、安全、G★、MPC、网络、初值、温度、epsilon与权重均保持冻结8684cca。原六同时运行已因WinError1455终止并保留；本技术restart1按seed顺序两臂配对，各4环境，共8环境，BLAS/OMP单线程；按2/1/0运行分配4/8/12条只读小Replay评估，每对完成后销毁评估池再开始下一seed。所有训练/评估Hidden，未授权扩步或增加种子。原始文件不移动；当前ZIP仅含本轮必要原始证据，模型独立保全并列SHA，Git按执行单交付。\n'
    supplement=ROOT/'记录/FIVE_K_REVIEW.md'
    if supplement.is_file():
        text=text.split('\n\n',1)[0]+'\n\n'+supplement.read_text(encoding='utf-8-sig')+'\n\n---\n\n'+text.split('\n\n',1)[1]
        evidence(supplement,'readout/FIVE_K_REVIEW.md')
    (build/'RESULT.md').write_text(text,encoding='utf-8');(ROOT/'RESULT.md').write_text(text,encoding='utf-8');(build/'TRAINING_STATUS.md').write_text('# V4 actual state\n\n'+status+'\n\n'+json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    for n in ('ASSESSMENT.json','POLICY_AND_GATE_AUDIT.json','MPC_COUPLING_AUDIT.json','THREE_SEED_COMPARISON.json','SHARED_0K.json'):evidence(build/n,'readout/'+n)
    # Snapshot bytes before hash and ZIP; active JSON sources may be atomically replaced.
    frozen={n:p.read_bytes() for n,p in raw.items()};manifest={n:{'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)} for n,b in frozen.items()};archive=build/'RAW_EVIDENCE_V4.zip'
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
            assert p.is_file();dest=history/f'BEFORE_V4_RESOURCE_RESTART1_20261020_{p.name}';assert not dest.exists();h=sha(p);shutil.move(str(p),dest);assert sha(dest)==h;moved.append({'source':str(p),'archive':str(dest),'sha256':h})
        dump(marker,{'archived':moved,'jst':stamp})
    names=['RESULT.md','ADMISSION.md','SAME_TASK_EVAL.csv','ASSESSMENT.json','POLICY_AND_GATE_AUDIT.json','MPC_COUPLING_AUDIT.json','MODEL_SHA256.json','TRAINING_STATUS.md','THREE_SEED_COMPARISON.json','SHARED_0K.json','RAW_EVIDENCE_V4.zip']
    for n in names:shutil.copy2(build/n,latest/n)
    receipt={'jst':stamp,'scientific_commit':FROZEN,'status':status,'phase':state.get('phase'),'zip_all_entries_sha_verified':True,'eval_rows':len(evals),'artifacts':{n:sha(latest/n) for n in names}}
    dump(latest/'DELIVERY_RECEIPT.json',receipt);dump(ROOT/'记录/DELIVERY_RECEIPT.json',receipt);(latest/'README.md').write_text('# T1-B V4 当前交付\n\n'+status+'。先读RESULT.md和ADMISSION.md，逐例结果及哈希随附；旧V3已保全归档。\n',encoding='utf-8');print(json.dumps(receipt,ensure_ascii=False))
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

