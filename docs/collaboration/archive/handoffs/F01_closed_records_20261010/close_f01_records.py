"""Publish existing F01 records only; missing results stay missing."""
from pathlib import Path
import json,hashlib,shutil,csv
B=Path(r'C:\Users\35884\Documents\Spacecraft');T=B/'过程文件/整理_20261010';C=B/'过程文件/协作/Git工作树/docs/collaboration'
M=Path(r'D:\py\DRL2');E=M/'eval/final2';D=C/'lower/handoffs/F01_closed_records_20261010';D.mkdir(parents=True,exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(p,q):q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(p)==sha(q)
def write(p,s):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf-8')
r=json.loads((T/'记录/readout_271000_three_closed.json').read_text(encoding='utf-8'))
assert r['verdict']=='METHOD_DOES_NOT_HOLD' and not r['problems']
copy(T/'记录/readout_271000_three_closed.json',D/'readout_271000_three.json')
missing={};count=0
rows=['pure','nominal']+[f'{row}/{s}' for row in ('stopping','learned') for s in (262460,262461,262462)]
for row in rows:
    for block in (271000,272000):
        absent=[]
        for seed in range(block,block+48):
            p=E/'formal'/row/f'seed_{seed}.json'
            if not p.exists():absent.append(seed);continue
            dest=D/('raw_271000_complete' if block==271000 else 'raw_272000_incomplete')/row/p.name
            copy(p,dest);count+=1
        if block==271000:assert not absent,(row,absent)
        missing[f'{row}/{block}']=absent
replays={}
for s in (262460,262461,262462):
    found=list((E/'replay'/str(s)).glob('seed_*.json'));replays[str(s)]=len(found)
    for p in found:copy(p,D/'replay_incomplete'/str(s)/p.name)
    for n in ('manifest.json','train.monitor.csv'):
        copy(M/f'logs/final2_{s}'/n,D/'training'/str(s)/n)
models={str(s):sha(M/f'logs/final2_{s}/final_model.zip') for s in (262460,262461,262462)}
data={'closure':'USER_CANCELLED_REMAINING_RECORD_WORK','science_commit':'f2f8169acd580ea96a578d45022dd8952447eca5','regime':'w2.36_r15','evaluation_files':count,'expected_evaluation_files':768,'formal_271000_complete':384,'formal_271000_official_rows':336,'nominal_271000_descriptive':48,'missing_evaluation_files':sum(map(len,missing.values())),'missing_by_row_block':missing,'replay_completed_file_counts':replays,'model_sha256':models,'evaluation_restart':False,'new_simulations':False,'partial_block_verdict':'NOT_COMPUTED','partial_replay_error_statistics':'NOT_COMPUTED','git_asset_recovery':'Historical code/docs/evidence provided by cleaned science commit and archive tags; missing F01 executions were never substituted with other blocks'}
write(D/'RECORD_MANIFEST.json',json.dumps(data,ensure_ascii=False,indent=2))
table='|模型|Pure|learned|stopping|协调增益|stopping违规|毁掉Pure成功|过门|\n|---|---:|---:|---:|---:|---:|---:|---|\n'
for s,v in r['models'].items():
    n=v['clean_completions'];table+=f"|{s}|{n['pure']}|{n['learned']}|{n['hybrid']}|+{n['hybrid']-n['learned']}|{v['violation_episodes']['hybrid']}|{len(v['destroyed'])}|{'是' if v['passes'] else '否'}|\n"
report=f'''# F01实验记录收口：不再补跑

最新用户授权：直接整理DRL2与Spacecraft，将剩余评估收为记录，进入下一阶段整体审查。没有启动新的评估、重放或训练。此前进程无报错消失，原因未知；原状态及中断现场完整保留。

## 完整271000官方读数

冻结科学代码f2f8169，工况w2.36_r15，三个模型均60k且code_dirty=false。同块Pure及三模型stopping/learned七组336回合完整，另外nominal48回合仅描述。用冻结experiments.v3_stopping readout重新整理已有数据：**METHOD_DOES_NOT_HOLD，problems=[]**；仅一个模型过门，未达到至少2/3。无新仿真，未改变门槛。

{table}

门槛：干净完成高于Pure、违规不多于Pure、毁掉Pure成功≤2。协调增益是同一模型stopping较learned多完成多少，不等于优于Pure；所有learned行违规均3，stopping分别3/3/0。完整官方原读数附readout_271000_three.json，失败列表与效率判据直接查该文件，不另改解释。历史nominal35个共同成功开局比Pure中位慢86.1s、多耗Δv1.33m/s；nominal48/48仅本轮历史描述，按共用交接下一阶段不再单列该基线。

## 不完整数据与缺失

现有评估{count}/768；271000完整384回合，272000仅{count-384}/384，缺{data['missing_evaluation_files']}回合。缺失编号逐项见RECORD_MANIFEST.json；不把未执行回合补为失败或成功，不以Git其他块替代。已有重放文件数{replays}，不发布完整96配对表或完整价值误差结论。所有残余JSON作为raw_272000_incomplete和replay_incomplete独立保存，不能用于与完整块混合排行榜。锁、临时文件、模型与完整原目录仅本地归档，不上传Git。

## 来源、归档与后续

本包直接附本轮原始结果、三模型训练来源、模型SHA、官方读数及逐文件SHA。模型不上传，不嵌套历史ZIP。此前阶段审查7b2d838、上层署名审查c34e17e为固定历史来源；其中根因推断不因本次收口就变成实测。

最新代码从科学分支4795910整理回D:/py/DRL2，实际F01仿真版本仍f2f8169，两个版本不可混同。历史代码、文档、已有证据通过该科学版本的docs/archive索引、evidence及archive/*标签定位；新阶段讨论以HANDOFF、docs/current和DEAD_ENDS为入口，具体新实验仍待用户与上层明确执行单。本次用户放弃补跑，旧六回合复现验收不自动执行，保留“未做”事实；完整pytest另见工程整理回执。
'''
write(D/'README.md',report)
copy(T/'脚本/close_f01_records.py',D/'close_f01_records.py')
write(D/'.gitattributes','* -text\n')
hashes={p.relative_to(D).as_posix():sha(p) for p in D.rglob('*') if p.is_file()}
write(D/'FILES_SHA256.json',json.dumps(hashes,ensure_ascii=False,indent=2))
write(T/'记录/F01_CLOSURE.json',json.dumps(data,ensure_ascii=False,indent=2))
print('F01_CLOSED_EXISTING_RECORDS_ONLY',count,replays,flush=True)
