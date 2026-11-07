"""Independent actual V2 readout, including incomplete/failure markers. No training."""
import json,csv,hashlib,zipfile,shutil,statistics
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];SPACE=ROOT.parents[1];REPO=Path('D:/py/DRL2');V1=ROOT.parent/'T1B学习修复V1_20261019'
FROZEN='a1698acefd0f3f4cf7298438bd842ef63ae0aee0'
def read(p):
    try:return json.loads(Path(p).read_text(encoding='utf-8-sig'))
    except (FileNotFoundError,json.JSONDecodeError):return {}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    stage=read(ROOT/'STATE.json');terminal=(ROOT/'记录/PIPELINE_EXIT.json').exists();failed=stage.get('phase','').startswith('stopped_')
    build=ROOT/'交付构建';build.mkdir(exist_ok=True);raw={};rows=[];models=[];gates={};coupling={};table=[]
    status='STOPPED / 按门槛或技术失败停止，未调参重跑' if failed else ('COMPLETE / 全部开发试跑完成，不自动扩种子' if terminal else 'RUNNING / 开发试跑中，未完成不宣称通过')
    def evidence(file,name=None):
        file=Path(file)
        if not file.is_file():return
        raw[name or file.relative_to(ROOT).as_posix()]=file
    for p in (ROOT/'输入').glob('*.md'):evidence(p)
    for p in (ROOT/'记录').glob('*.json'):evidence(p)
    for name in ['pytest.log','contract.log']:evidence(ROOT/'记录'/name)
    evidence(ROOT/'contract/CONTRACT.json')
    for p in (ROOT/'脚本').glob('*.py'):evidence(p)
    for p in (ROOT/'scan_292120').glob('scan_*.json'):evidence(p)
    for p in (REPO/'evidence/T1B_V1_AUDIT/candidate_truth_scan').glob('r_*.json'):evidence(p,'upper_original_truth/'+p.name)
    for name in ['README.md','cf_label_and_coverage.json','policy_saturation_audit.json']:
        evidence(REPO/'evidence/T1B_V1_AUDIT'/name,'upper_V1_audit/'+name)
    evidence(REPO/'evidence/T1B_V1/profile_mission_v3.json','profile_mission_v3.json')
    for name in ['RESULT_V1.md','SAME_TASK_EVAL.csv']:evidence(REPO/'evidence/T1B_V1'/name,'reference_V1/'+name)
    for arm in ['e_sac','e_tcpf']:
        path=ROOT/f'TRAIN_{arm}_s0';h=read(path/'heartbeat.json');finish=read(path/'finished.json');cov=read(ROOT/f'coverage_{arm}.json');learn=read(path/'learning_audit.json') or []
        complete=[b for b in cov.get('bins',[]) if b['complete']];a2=('PASS so far' if cov.get('A2_pass_complete_windows') else 'FAIL') if complete else 'PENDING: no complete post-warmup1k window'
        gates[arm]={'A2':cov,'A1':{}}
        for mark in [2000,5000,10000]:
            audit=read(ROOT/f'audit_{arm}_{mark}.json');evidence(ROOT/f'audit_{arm}_{mark}.json')
            if audit:gates[arm]['A1'][str(mark)]=audit
            for p in (ROOT/f'eval_{arm}_{mark}').glob('*.json'):
                r=read(p)
                if 'opening' not in r:continue
                block='292140–147' if r['opening']>=292140 else '292120–127'
                rows.append({'arm':arm,'checkpoint':mark,'block':block,'opening':r['opening'],'D_s':r['mission']['deadline_s'],'B_m_s':r['mission']['budget_m_s'],'event':r['event'],'capture':int(r['event']=='success'),'discounted_return':r['discounted_return'],'time_s':r['time_s'],'delta_v':r['delta_v'],'first_index':(r.get('first_decision') or {}).get('index'),'source':str(p),'source_sha256':sha(p)});evidence(p)
            if terminal:
                for p in (ROOT/f'eval_{arm}_{mark}').glob('*.trace.jsonl'):evidence(p)
        probes=[v['applied_gradient_share'] for v in learn if v.get('phase')=='on'];feedback=read(path/'feedback_audit.json');directions={'advance':0,'delay':0,'equal':0};valid_pairs=0
        log=path/'model_prefix_releases.jsonl'
        if log.exists():
            for line in log.read_bytes().splitlines(keepends=True):
                if not line.endswith(b'\n'):continue
                r=json.loads(line)
                if r['gate']['weight']<=0:continue
                valid_pairs+=1;s=r['payload']['sampling'];choices=s['population_tau_s'];delta=choices[s['alternative_index']]-choices[s['selected_index']];directions['delay' if delta>0 else 'advance' if delta<0 else 'equal']+=1
        median=statistics.median(probes) if probes else None
        coupling[arm]={'manifest':read(path/'manifest.json'),'heartbeat':h,'finished':finish,'feedback_audit':feedback,'valid_logged_pairs':valid_pairs,'valid_pair_directions':directions,'applied_gradient_share_median_all_on_probes':median,'on_probe_count':len(probes),'B1_pass_so_far':bool(arm=='e_tcpf' and valid_pairs>0 and directions['advance']>0 and directions['delay']>0 and median is not None and .02-1e-9<=median<=.1+1e-9),'learning_probes':learn}
        row=finish or h
        table.append(f"|{arm}|{row.get('steps','未启动')}|{row.get('critic_updates','—')}|{row.get('actor_updates','—')}|{a2}|")
        evidence(ROOT/f'coverage_{arm}.json')
        for name in ['manifest.json','heartbeat.json','finished.json','learning_audit.json','feedback_audit.json','delta_calibration.json']:
            p=path/name
            if p.exists():
                snap=build/f'{arm}_{name}';shutil.copy2(p,snap);raw[f'training/{arm}/{name}']=snap
        if terminal:
            for name in ['decisions.jsonl','model_prefix_releases.jsonl']:
                evidence(path/name,f'training/{arm}/{name}')
        for p in (path/'checkpoints').glob('*.zip'):
            models.append({'arm':arm,'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'clock':'predeclared callback, before current4 transitions/gradient updates'})
    truth={}
    for file in [*(REPO/'evidence/T1B_V1_AUDIT/candidate_truth_scan').glob('r_*.json'),*(ROOT/'scan_292120').glob('scan_*.json')]:
        r=read(file)
        if r.get('skipped') or 'discounted_return' not in r:continue
        op=r['opening'];j=r.get('first_index',r.get('j0'));truth.setdefault(op,{})[j]=r['discounted_return']
    paired={};b2={}
    for mark in [2000,5000,10000]:
        for block in ['292140–147','292120–127']:
            a={r['opening']:r for r in rows if r['arm']=='e_sac' and r['checkpoint']==mark and r['block']==block};b={r['opening']:r for r in rows if r['arm']=='e_tcpf' and r['checkpoint']==mark and r['block']==block}
            if len(a)==len(b)==8:paired[f'{mark}_{block}']={'paired_return_differences':{str(op):b[op]['discounted_return']-a[op]['discounted_return'] for op in sorted(a)},'first_choice_difference_states':sum(a[op]['first_index']!=b[op]['first_index'] for op in a)}
        audits={a:read(ROOT/f'audit_{a}_{mark}.json') for a in ['e_sac','e_tcpf']}
        if all(audits.values()):
            states={a:list(d['models'].values())[-1]['per_state'] for a,d in audits.items()};detail={}
            for arm,ss in states.items():
                scored=[s for s in ss if len(truth.get(s['opening'],{}))==len(s['p'])];correct=sum(s['Q_argmax']==max(truth[s['opening']],key=truth[s['opening']].get) for s in scored)
                detail[arm]={'ranked_states':len(scored),'Q_best_first_choice_truth_agreement':correct/len(scored) if scored else None,'selected_first_index':{str(s['opening']):s['selected_index'] for s in ss}}
            b2[str(mark)]={'first16_choice_difference_states':sum(x['selected_index']!=y['selected_index'] for x,y in zip(states['e_sac'],states['e_tcpf'])),'arms':detail,'caution':'Candidate truth uses fixed subsequent V1 teacher, not strict current-policy Q; no absolute-Q truth claim.'}
    final10={}
    for arm in ['e_sac','e_tcpf']:
        final10[arm]={}
        for block,baseline in [('292140–147',36.53539449952668),('292120–127',16.215477761063436)]:
            group=[r for r in rows if r['arm']==arm and r['checkpoint']==10000 and r['block']==block]
            if len(group)==8:final10[arm][block]={'captures':sum(r['capture'] for r in group),'mean_return':statistics.mean(r['discounted_return'] for r in group),'fixed_always_earliest_baseline':baseline,'first_non_earliest':sum(r['first_index']!=0 for r in group)}
    a3=any(len(g)==2 and all(v['mean_return']>=v['fixed_always_earliest_baseline'] for v in g.values()) and any(v['first_non_earliest'] for v in g.values()) for g in final10.values()) if any(len(g)==2 for g in final10.values()) else None
    assessment={'state':stage,'A1_A2':gates,'A3_pass':a3,'10k':final10,'B1':coupling,'B2':b2,'paired_results':paired,'trajectory_comparison':read(ROOT/'B2_TRAJECTORY_AUDIT.json'),'all_three_research_goals_claimed':False,'no_automatic_multi_seed':True}
    dump(build/'POLICY_AND_GATE_AUDIT.json',{'gates':gates,'A3':a3,'B2':b2,'paired':paired});dump(build/'MPC_COUPLING_AUDIT.json',coupling);dump(build/'MODEL_SHA256.json',models);dump(build/'ASSESSMENT.json',assessment)
    fields=list(rows[0]) if rows else ['arm','checkpoint','block','opening','event','discounted_return']
    with (build/'SAME_TASK_EVAL.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(sorted(rows,key=lambda r:(r['checkpoint'],r['arm'],r['opening'])))
    timestamp=datetime.now(timezone(timedelta(hours=9))).isoformat()
    text=f'''# T1_B_V2_PILOT_20261019 实测交付\n\n{status}。时间{timestamp}。科学冻结{FROZEN}；seed0，E-SAC/E-TCPF各10k开发试跑，无多种子GO。\n\n必要检查：pytest12/12通过（记录/pytest.log）；CONTRACT.passed=true，初值动作最大差0，alpha=.5、epsilon=.10；共享专家完整SHA bf3c24f9f8db7eb156a5ab42835f6faeefee2332efbd59ba7c35ed733f5ffe80，没有重采专家或BC。固定任务/奖励/D/B/G★/MPC/网络/训练器保持上层冻结版本。\n\n|臂|真实全局步|critic更新|Actor更新|A2|\n|---|---:|---:|---:|---|\n'''+ '\n'.join(table)
    text+=f"\n\n阶段{stage.get('phase')}；停止原因{json.dumps(stage.get('stop_reason'),ensure_ascii=False)}。原生子进程、命令、门槛预登记和结果计数见ASSESSMENT。A1原8首审状态至少4个熵>.05，Actor梯度至少0k的1%；2k/5k任一臂失败即停。A2暖身结束后完整1k窗口非最早≥5%，执行窗口已在启动前固定501–1500、1501–2500等；尾部不足1k只报告，不冒充完整窗口。没有完整窗口/检查点时为PENDING而非通过。\n\n"
    if failed:text+='\n停止时步数若只有heartbeat/完整日志而没有finished.json，仅为最后持久化可核验值；强制停止后未保存的精确优化次数不补造。\n\n'
    text+='|检查点|臂|开发块|完成/8|捕获|平均原折扣回报|\n|---|---|---|---:|---:|---:|\n'
    for mark in [2000,5000,10000]:
        for arm in ['e_sac','e_tcpf']:
            for block in ['292140–147','292120–127']:
                group=[r for r in rows if r['arm']==arm and r['checkpoint']==mark and r['block']==block]
                if group:text+=f"|{mark}|{arm}|{block}|{len(group)}/8|{sum(r['capture'] for r in group)}|{statistics.mean(r['discounted_return'] for r in group):.9f}{'（部分）' if len(group)<8 else ''}|\n"
    text+=f'''\nA3当前判定：{a3 if a3 is not None else 'PENDING'}；按两块总是最早原回报36.535394500 /16.215477761比较，并要求某开局首次非最早，不用挑选成功回合fuel替代全样本回报。B1需有效pair>0、提前/推迟两侧均出现、on阶段全部探针（含0）的反馈梯度占比中位2–10%；B2只如实报告选择差、与真值排序一致率、逐例配对回报差，不设方向门槛。详细数值见POLICY_AND_GATE_AUDIT与MPC_COUPLING_AUDIT。\n\n必要历史对照：V1两臂2k/5k/10k原块均7/8、35.582056910，扩展10k均6/8、15.505362613；原始cold两块4/8、−34.446131473与6/8、17.132155286；强全程G★均7/8，回报37.436580681与39.961325235。不能把接口/专家收益计作RL或MPC独立收益。原块上层24候选真值采用固定V1教师后续，最佳首次8/8、约60.87，最早7/8、约36.50；扩展24例本机真值已完成{stage.get('completed_scan',0)}/24，全部候选/失败/skip均保留，不把这种教师后续回报称为当前SAC严格Q。\n\n最终科学要求未由工程检查或非零梯度自动成立；本轮为单seed开发判断，不称跨seed统计成功。A1/A2失败不调参重跑，A3失败不扩种子，技术失败只允许按单核对原始stderr后从零_restart1，禁止续训。当前无多种子或新的正式训练授权。原始科学产物不搬动，来源{ROOT}；RAW_EVIDENCE_V2.zip每项SHA核验，未完成项明确缺失，不生成替代数据。'''
    (build/'RESULT.md').write_text(text,encoding='utf-8');(ROOT/'RESULT.md').write_text(text,encoding='utf-8')
    (build/'TRAINING_STATUS.md').write_text('# V2实际训练状态\n\n'+status+'\n\n'+'\n'.join(table)+'\n\n'+json.dumps(stage,ensure_ascii=False,indent=2),encoding='utf-8')
    for name in ['POLICY_AND_GATE_AUDIT.json','MPC_COUPLING_AUDIT.json','ASSESSMENT.json']:raw['readout/'+name]=build/name
    for name in ['B2_TRAJECTORY_AUDIT.json','B2_TRAJECTORY_COMPARISON.csv']:evidence(ROOT/name)
    # Freeze bytes before hashing: coverage/heartbeat JSON may be atomically
    # replaced by the active queue. Never move or zip a changing source directly.
    snapshots={}
    for n,p in raw.items():
        snap=build/'raw_snapshots'/n;snap.parent.mkdir(parents=True,exist_ok=True);snap.write_bytes(p.read_bytes());snapshots[n]=snap
    raw=snapshots
    manifest={n:{'sha256':sha(p),'bytes':p.stat().st_size} for n,p in raw.items()};archive=build/'RAW_EVIDENCE_V2.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for n,p in raw.items():z.write(p,n)
        z.writestr('MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for n,h in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==h['sha256']
    latest=SPACE/'上层交付/最新';history=SPACE/'上层交付/历史';marker=ROOT/'记录/FIRST_PUBLICATION.json'
    if not marker.exists():
        moved=[]
        assert latest.resolve().is_relative_to(SPACE.resolve()) and history.resolve().is_relative_to(SPACE.resolve())
        for p in list(latest.iterdir()):
            assert p.is_file();target=history/f'BEFORE_T1B_V2_20261019_{p.name}';assert not target.exists();before=sha(p);shutil.move(str(p),target);assert sha(target)==before;moved.append({'file':p.name,'archive':str(target),'sha256':before})
        dump(marker,{'moved':moved,'timestamp_jst':timestamp})
    names=['RESULT.md','SAME_TASK_EVAL.csv','POLICY_AND_GATE_AUDIT.json','MPC_COUPLING_AUDIT.json','TRAINING_STATUS.md','MODEL_SHA256.json','ASSESSMENT.json','RAW_EVIDENCE_V2.zip']
    for name in names:shutil.copy2(build/name,latest/name)
    receipt={'timestamp_jst':timestamp,'status':status,'scientific_commit':FROZEN,'phase':stage.get('phase'),'raw_manifest_items':len(manifest),'zip_verified':True,'artifacts':{n:{'sha256':sha(latest/n),'bytes':(latest/n).stat().st_size} for n in names}}
    dump(latest/'DELIVERY_RECEIPT.json',receipt);dump(ROOT/'记录/DELIVERY_RECEIPT.json',receipt)
    (latest/'README.md').write_text('# T1-B V2当前实测交付\n\n'+status+'。先读RESULT.md和POLICY_AND_GATE_AUDIT.json。旧V1交付已完整保全归档；不把未完成或失败补造为通过。全部证据SHA见DELIVERY_RECEIPT.json。\n',encoding='utf-8')
    print(json.dumps({'status':status,'phase':stage.get('phase'),'eval_rows':len(rows),'raw_items':len(manifest),'A3':a3},ensure_ascii=False))
if __name__=='__main__':main()
