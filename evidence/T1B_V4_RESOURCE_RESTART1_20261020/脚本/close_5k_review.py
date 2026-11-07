"""Close an interrupted round after the user's bounded frozen-5k evaluation."""
import json,hashlib,statistics,datetime,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SPACE=ROOT.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    receipt=read(ROOT/'记录/FINISH_5K_EXIT.json');assert receipt['total_5k_episodes']==32
    stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
    coverage=read(ROOT/'记录/REPLAY_COVERAGE_5K.json')
    text='''本轮结论：V4冻结8684cca的两臂seed0、5k同任务配对评估已补齐；训练仍停止，未启动20k。两块各8固定开局全部保留，不挑选成功例；原E-SAC16例逐文件SHA复用，新补TCPF16例。末尾原始5k模型SHA、逐候选Q/概率、行为覆盖和CF标签分解均随交付。主机崩溃影响训练长度，不等于已证明继续训练可改善。

|阶段|臂|块|捕获|均值原折扣回报|首选非最早|
|---|---|---|---:|---:|---:|
'''
    means={}
    for mark in (2000,5000):
        for arm in ('e_sac','e_tcpf'):
            label=f'{arm}_s0_{mark}';rr=[read(ROOT/f'eval_{label}/{label}_{o}.json') for o in [*range(292140,292148),*range(292120,292128)]]
            for block,lo in [('main',292140),('expanded',292120)]:
                z=[r for r in rr if lo<=r['opening']<lo+8];assert len(z)==8
                ret=statistics.mean(r['discounted_return'] for r in z);means[f'{arm}_{mark}_{block}']=ret
                text+=f"|{mark}|{arm}|{block}|{sum(r['event']=='success' for r in z)}/8|{ret:.9f}|{sum(r.get('first_decision') is not None and r['first_decision']['index']!=0 for r in z)}|\n"
    for block in ('main','expanded'):
        d=means[f'e_tcpf_5000_{block}']-means[f'e_sac_5000_{block}'];text+=f'\n5k {block}配对E-TCPF−E-SAC平均回报差={d:.9f}。\n'
    text+='''
同接口0k主块6/8、13.715624173；扩展6/8、18.001071055。强G★历史对照主块7/8、37.436580681，扩展7/8、39.961325235（历史方法/接口不同，仅作背景，不替代同任务对照）。固定首次候选真值使用V1教师后续规则：主块总是最早7/8、约36.54，按状态最佳8/8、约60.87；扩展约16.38与18.70。该规则的回报不是当前SAC严格Q，不能直接用绝对误差归因critic。

|臂|步数|15个三候选状态中Q选非最早|Actor选非最早|每延后一档平均Q变化|原8熵>0.05|原8Actor梯度范数|
|---|---:|---:|---:|---:|---:|---:|
'''
    for arm in ('e_sac','e_tcpf'):
        for mark in (2000,5000):
            a=read(ROOT/f'audit_{arm}_s0_{mark}.json');m=a['models'][str(mark)];ss=[s for s in m['per_state'] if len(s['p'])==3]
            slope=statistics.mean((s['Q_min'][2]-s['Q_min'][0])/2 for s in ss)
            text+=f"|{arm}|{mark}|{sum(s['Q_argmax']!=0 for s in ss)}/15|{sum(s['selected_index']!=0 for s in ss)}/15|{slope:.6f}|{a['original8_entropy_gt_005']}/8|{m['actor_grad_norm']:.9f}|\n"
    text+='\n覆盖（实际完整JSONL前缀，500<step≤5000；在线与共享专家分开）：\n'
    for arm,c in coverage.items():text+=f"\n{arm}：非最早{c['post500_to5k_logged_non_earliest']}/{c['post500_to5k_logged_decisions']}={100*c['non_earliest_fraction']:.2f}%；完整行为日志到{c['logged_last_complete_step']}，心跳仅到{c['heartbeat_last_step']}；日志/心跳均不是对应权重可恢复时钟。\n"
    text+='''
机制与解释边界：两臂2k M均通过。E-TCPF最后持久化248个释放配对中193个有效，槽0/1分别110/83，提前54/推迟139；反馈梯度份额中位0.10（含零探针）。冻结5k标签复算193对：即时差绝对中位0.195534、自举差2.035713，95.34%由自举项量级主导，97.41%标签符号与自举项一致。以上为冻结target critic复算，不是历史每次更新的原标签。实际监督进入critic、保留Actor梯度，不等于产生独立任务收益。

目前不能把问题仅归为训练不够：两臂15/15三候选初态的Q仍把最早排第一，E-SAC的延后负斜率从2k到5k加陡且熵下降；E-TCPF使斜率更平且保持熵，但确定性首选仍未改变。尽管在线非最早覆盖约40%，总决策量仍小；覆盖数量、函数外推与短窗反馈自举信息不足尚无法从单seed5k独立区分。应由上层据逐状态原始表判断学习层是否需改；不得把再跑20k称为已证实修复。本轮没有修改任务/安全/奖励/G★/MPC/Actor/critic，没有新训练。

训练未完成与当前资产：01:59系统蓝屏重启，两臂心跳6200/5100，日志最后完整向量记录6240/5144；两臂最后权重皆5k。10k/20k以及seed1/2均缺失。原Replay/RNG只保留本地并列SHA，不能将callback保存模型的pre-update相位与Replay/优化器/RNG盲目拼接续跑。崩溃事件、当前双显卡OK、用户授权缺项评估回执随附。旧HOST_CRASH_REPORT中的TCPF5k缺项为10:35历史事实，本轮完成后以本文和FINISH_5K_EXIT为准。
'''
    (ROOT/'记录/FIVE_K_REVIEW.md').write_text(text,encoding='utf-8')
    local=[]
    for arm in ('e_sac','e_tcpf'):
        for name in ('replay_latest.pkl','rng_latest.pkl'):
            p=ROOT/f'TRAIN_{arm}_s0/{name}'
            if p.exists():local.append({'arm':arm,'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'scope':'Preserved local asset; matching resume clocks not certified by this evaluation'})
    dump(ROOT/'记录/LOCAL_RESTART_ASSET_SHA256.json',local)
    state=read(ROOT/'STATE.json');state.update(phase='stopped_host_reboot_5k_review_complete',training_active=False,checkpoint_evaluation_active=False,completed_evaluations=64,active_evaluations=[],review_completed_jst=stamp,user_latest_scope='Complete matched frozen5k evaluation and push key data only; no20k training started')
    dump(ROOT/'STATE.json',state)
    dump(ROOT/'记录/PIPELINE_EXIT.json',{'phase':state['phase'],'all_science_workers_closed':True,'training_complete':False,'planned_three_seed20k_complete':False,'matched5k_readout_complete':True,'origin':'Observational closeout after host reboot and separately authorized missing frozen5k evaluations; not native training completion','jst':stamp})
    for name in ('FIVE_K_REVIEW.md','Q_POLICY_SCAN_2K_5K.csv','BEHAVIOUR_COVERAGE_1K.csv','REPLAY_COVERAGE_5K.json'):
        shutil.copy2(ROOT/'记录'/name,SPACE/'上层交付/最新'/name)
    print(json.dumps({'phase':state['phase'],'means':means},ensure_ascii=False))
if __name__=='__main__':main()
