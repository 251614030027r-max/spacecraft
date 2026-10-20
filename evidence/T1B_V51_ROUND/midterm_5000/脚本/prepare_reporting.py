"""Adapt reporting only. No changes to scientific modules or active training."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT.parent/'T1B_V4_ROUND_20261019_restart1'
src=(OLD/'脚本/publish_readout.py').read_text(encoding='utf-8')
src=src.replace('train.t1_b_v4_sac','train.t1_b_v5_sac').replace('FeasibleSupportSACV4','FeasibleSupportSACV5').replace('8684cca3d3c3a2997f3ed3f2c106c247b94e20e4','b6f5c8a22728b6ca3970d6201525cd4047cba8d8')
start=src.index('    stamp=datetime.datetime.now(');end=src.index("    for n in ('ASSESSMENT.json'",start)
src=src[:start]+'''    stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
    admission=f"# V5.1 ADMISSION\\n\\nFrozen {FROZEN}; local specified16 tests and contract passed. epsilon=.5, alpha=.5; full-horizon feedback share10% calibrated/20% cap in E-TCPF; shared original V3 expert1000 transitions, SHA9c91cb76f10999ca36aecf4172bb092e4e52de9fcfbf62402a76306e701e1d0f. No new expert or BC. Seed0 2k M: no live-plan learning, post500 non-earliest>=30%, both CF slots admitted>0; any failure stops all.\\n"
    (build/'ADMISSION.md').write_text(admission,encoding='utf-8')
    text=f"# T1-B V5.1 actual research readout\\n\\n{status}; {stamp}; scientific freeze {FROZEN}.\\n\\n"
    text+="用户授权seed0–2两臂各20k；预留一个物理核（逻辑CPU0/1）、BelowNormal。当前物理/提交内存仅按完整双臂配对准入，容量不足时种子排队；不是六运行均已启动。每臂仍4环境，算法/奖励/安全/任务/Gstar/MPC/初值/步数全冻结。实际活动运行与排队种子见STATE/TRAINING_STATUS及RESOURCE_ADMISSION。\\n\\n"
    text+="|运行|持久化真实步|critic更新|Actor更新|M|\\n|---|---:|---:|---:|---|\\n"
    for r in table:text+='|'+'|'.join(str(r[k]) for k in ('run','steps','critic_updates','actor_updates','M'))+'|\\n'
    text+="\\n|seed|checkpoint|arm|block|complete|captures|mean original discounted return|non-earliest first|no learning first|\\n|---|---:|---|---|---:|---:|---:|---:|---:|\\n"
    for seed in range(3):
        for mark in MARKS:
            for arm in ARMS:
                for block in ('292140–147','292120–127'):
                    rr=[r for r in evals if r['seed']==seed and r['checkpoint']==mark and r['arm']==arm and r['block']==block]
                    if rr:text+=f"|{seed}|{mark}|{arm}|{block}|{len(rr)}/8|{sum(r['capture'] for r in rr)}|{statistics.mean(r['discounted_return'] for r in rr):.9f}{' PARTIAL' if len(rr)<8 else ''}|{sum(r['first_index'] is not None and r['first_index']!=0 for r in rr)}|{sum(r['first_index'] is None for r in rr)}|\\n"
    for block,ops in [('main',range(292140,292148)),('expanded',range(292120,292128))]:
        zz=[r for r in zero if r['opening'] in ops]
        if zz:text+=f"\\nV5.1 common0k {block}: {len(zz)}/8 completed, captures{sum(r['event']=='success' for r in zz)}, mean={statistics.mean(r['discounted_return'] for r in zz):.9f}; partial blocks are not full estimates.\\n"
    text+="\\n必要历史背景：V4同任务seed0主块0k6/8、13.715624173，2k和5k两臂均7/8、36.538164132；扩展0k6/8、18.001071055，2k/5k两臂均6/8、16.220946479，逐例相同。V4完整失败证据提交2c7283b，不把V4模型当V5.1模型。固定教师后续真值主块最早7/8约36.54、最佳首选8/8约60.87；扩展6/8约16.38和6/8约18.70。固定教师后续真值不等于当前SAC严格Q；强Gstar历史主块7/8、37.436580681，扩展7/8、39.961325235仅作背景而非同条件消融。\\n\\n逐状态Q1/Q2/概率/梯度/熵，15个三候选初态Q最佳与真值最佳一致数、经济/急迫首选分布及每档Q斜率见POLICY_AND_GATE_AUDIT。每seed、每stage、同开局TCPF−SAC和三seed均值见THREE_SEED_COMPARISON；只有三个seed全齐才生成三seed均值。技术/M/非零梯度不等于科研成功。\\n\\nFull-horizon MPC配对原始payload、真实区间回报与事实模型回报误差、拒绝原因、两槽接受率及是否两侧终止见MPC_FULL_HORIZON_AUDIT。门控实际冻结为同终止事件、终止步差<=10、回报误差<=5；当两侧终止时反馈Y为模型成对回报差，不含critic自举。CF_LABEL文件仍是冻结检查点对持久化配对的复算，不冒充历史每次更新原标签。开发初态的模型排序验证原始证据在科学b6f5c8a的evidence/T1B_V5_PRECHECK，最佳一致11/15、两两方向31/38；它是预检而非本轮训练独立增益。\\n"
    full={}
    for key in coupling:
        pp=rows(ROOT/f'TRAIN_{key}/model_prefix_releases.jsonl');out=[];reasons={}
        for r in pp:
            f=r['payload']['factual'];a=r['payload']['alternative'];g=r['gate'];both=bool(f['done'] and a['done'])
            for reason in g.get('reasons',[]):reasons[reason]=reasons.get(reason,0)+1
            out.append({'slot':r['slot'],'env':r['env'],'cf_slot':r.get('cf_slot'),'add_index':r['add_index'],'actual_return':r['actual_return'],'actual_event':r.get('actual_event'),'actual_steps':r.get('actual_steps'),'factual_return':f['return'],'factual_event':f.get('event'),'factual_steps':f['steps'],'alternative_return':a['return'],'alternative_event':a.get('event'),'both_model_branches_terminal':both,'independent_paired_Y':a['return']-f['return'] if both else None,'absolute_factual_return_error':abs(f['return']-r['actual_return']),'gate_weight':g.get('weight',0),'reasons':json.dumps(g.get('reasons',[]))})
        errors=[r['absolute_factual_return_error'] for r in out];valid=[r for r in out if r['gate_weight']>0]
        full[key]={'total_pairs':len(out),'valid_pairs':len(valid),'rejection_reasons':reasons,'valid_by_slot':{str(j):sum(r['cf_slot']==j for r in valid) for j in (0,1)},'valid_both_terminal':sum(r['both_model_branches_terminal'] for r in valid),'absolute_factual_return_error_median':statistics.median(errors) if errors else None,'absolute_factual_return_error_max':max(errors) if errors else None,'scope':'Actual persistent released training pairs, not new real counterfactual evaluations'}
        if out:csvfile(build/f'FULL_HORIZON_PAIRS_{key}.csv',out,list(out[0]));evidence(build/f'FULL_HORIZON_PAIRS_{key}.csv','readout/FULL_HORIZON_PAIRS_'+key+'.csv')
    dump(build/'MPC_FULL_HORIZON_AUDIT.json',full);evidence(build/'MPC_FULL_HORIZON_AUDIT.json','readout/MPC_FULL_HORIZON_AUDIT.json')
    if state.get('stop_reason'):text+='\\n停止原因：'+json.dumps(state['stop_reason'],ensure_ascii=False)+'\\n'
    (build/'RESULT.md').write_text(text,encoding='utf-8');(ROOT/'RESULT.md').write_text(text,encoding='utf-8');(build/'TRAINING_STATUS.md').write_text('# V5.1 actual state\\n\\n'+status+'\\n\\n'+json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
''' +src[end:]
src=src.replace('RAW_EVIDENCE_V4.zip','RAW_EVIDENCE_V51.zip').replace('BEFORE_V4_RESOURCE_RESTART1_20261020_','BEFORE_V51_ROUND_20261020_').replace('# T1-B V4 当前交付','# T1-B V5.1 当前交付').replace('旧V3已保全归档','旧V4已保全归档')
src=src.replace("'SHARED_0K.json','RAW_EVIDENCE_V51.zip'","'SHARED_0K.json','MPC_FULL_HORIZON_AUDIT.json','RAW_EVIDENCE_V51.zip'")
(ROOT/'脚本/publish_readout.py').write_text(src,encoding='utf-8')
compile(src,str(ROOT/'脚本/publish_readout.py'),'exec')
