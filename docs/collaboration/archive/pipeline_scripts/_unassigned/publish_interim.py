from pathlib import Path
import json,csv,hashlib,shutil
TOP=Path(__file__).resolve().parents[1];OUT=TOP/'reports/figures_interim_20261009';SPACE=Path('C:/Users/35884/Documents/Spacecraft');REPO=SPACE/'过程文件/协作/Git工作树';DEST=REPO/'docs/collaboration/lower/handoffs/figures_interim_20261009'
assert not DEST.exists();DEST.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
manifest=read(OUT/'FILES_SHA256.json');sources=read(OUT/'INPUTS_SHA256.json')
# Keep new plotted data and the few necessary canonical inputs; reference historical raw rounds.
for p in OUT.rglob('*'):
    if not p.is_file() or p.name=='FILES_SHA256.json':continue
    rel=p.relative_to(OUT)
    if rel.parts[0]=='inputs' and '_seed_' in p.name:continue
    assert sha(p)==manifest[rel.as_posix()];target=DEST/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
for source,r in sources.items():
    snapshot=Path(r['snapshot']);r['snapshot_in_public_package']=(DEST/snapshot).is_file()
    if not r['snapshot_in_public_package']:
        r['local_snapshot']=str(OUT/snapshot);r.pop('snapshot')
        r['public_export']='plot_data/fig2_formal_rows.csv' if '/formal/' in source.replace('\\','/') else 'plot_data/fig3_baselines.csv'
        r['history']='https://github.com/251614030027r-max/spacecraft/tree/6c5cb97a319e425bd616d4fe323ed1a9c43902a4/docs/reviews/stopping_20261006' if '/formal/' in source.replace('\\','/') else '../final_mainline_screen_20261007/README.md'
(DEST/'INPUTS_SHA256.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf-8')
(DEST/'.gitattributes').write_text('* -text\n',encoding='ascii')
readout=REPO/'docs/collaboration/lower/handoffs/final_mainline_screen_20261007/readout_final.json';assert sha(readout)=='a74603ce19c7cc2756e67c02a1165a6a924ed0d58d0b48968b10b72a4db5bde8';shutil.copy2(readout,DEST/'inputs/official_screen_readout.json')
(DEST/'HISTORY_LINKS.md').write_text('''# 固定历史证据（不复制整轮）

- 停止头正式轮原始336结果：[6c5cb97独立审查](https://github.com/251614030027r-max/spacecraft/tree/6c5cb97a319e425bd616d4fe323ed1a9c43902a4/docs/reviews/stopping_20261006)。本包的fig2_formal_rows.csv完整提供每回合用于绘图的指标、失败、commit、模型SHA，并另有共同成功配对CSV。原ZIP SHA94ab2368ea2d50d35aecc67230e0a6e37606dd5047d4ada202e316e33583f2ae已在本机重核。
- 工况筛选：[原协作读数](../final_mainline_screen_20261007/README.md)，其中readout_final SHA a74603ce19c7cc2756e67c02a1165a6a924ed0d58d0b48968b10b72a4db5bde8。本包保留96回合绘图CSV与官方读数，不复制整九格筛选目录。
- 上一批50k案例与不利训练诊断：[342f2e6固定入口](https://github.com/251614030027r-max/spacecraft/blob/342f2e65d7ad4689b5dfc151002d6021577700f5/docs/collaboration/lower/handoffs/ppt_temporary_20261009/README.md)。该入口已被本版展示替代，图面修订不改真实物理数据。
- 执行单：[85bc7a8](https://github.com/251614030027r-max/spacecraft/blob/85bc7a8/docs/collaboration/upper/run_orders/FIGURES_INTERIM_20261009.md)。本次唯一数值修正：用户明确“按真实1／0／1标注，继续图2”。
''',encoding='utf-8')
files={p.relative_to(DEST).as_posix():sha(p) for p in DEST.rglob('*') if p.is_file() and p.name!='FILES_SHA256.json'};(DEST/'FILES_SHA256.json').write_text(json.dumps(files,ensure_ascii=False,indent=2),encoding='utf-8');assert all(sha(DEST/n)==v for n,v in files.items())
handoff=REPO/'docs/collaboration/lower/handoffs/figures_interim_20261009.md';handoff.write_text('''# 阶段基准图交接（2026-10-09）

TEMPORARY_REVIEW_ONLY / NOT_PAPER_PERFORMANCE，按85bc7a8执行，用户授权本次临时资产与执行单推进。五项七张图已完成并逐图目视审查：V3e历史60k训练、267000正式负结果、269000两基线、B1/B2事后上界、此前50k单案例。107项核对全部通过；3日志SHA、B1/B2固定SHA、正式ZIP SHA以及336原始结果对独立审查原件SHA均一致。

[新版图表、CSV、脚本、SHA与口径](figures_interim_20261009/README.md) · [独立支撑审核](figures_interim_20261009/REVIEW.md) · [核对表](figures_interim_20261009/CHECKS.md)。每个图有400dpi PNG/PDF/SVG、真实画入CSV、源路径/SHA。

图2learned实际违规1/0/1，已获用户明确授权据实绘制；Pure/stopping均0，+25/+10/+19完成增益不改变METHOD_DOES_NOT_HOLD。案例第一栏使用位置误差0.25m的真实阈值，而非目标质心距离；其余0.05m/s、10°阈值均来自配置。50k案例仍慢于Pure。V3e上升曲线不能冒充当前方法训练曲线；事后48/48不能冒充在线性能。

无新仿真、评估或检查点评估；当前训练出图搁置，混合训练图不进正文，未绘任何271000/272000方法行。可复用图2/3脚本及未来编号最小案例选择规则已写入REUSE，未来步骤未执行。旧本地展示入口已核验归档；旧342f2e6保持固定引用，本版成为临时展示最新入口，后续正式好结果可替换。实验总账只补下层本地列，没有更改上层科学状态标签。
''',encoding='utf-8')
index=REPO/'docs/collaboration/lower/README.md';s=index.read_text(encoding='utf-8');s=s.replace('**最新临时汇报资产（2026-10-09，待上层审查，可替换）：**','**旧临时资产（已被阶段基准图替代，仅历史引用）：**');s+='\n\n**最新阶段基准图（2026-10-09，待审查，可替换）：** [交接](handoffs/figures_interim_20261009.md) · [五项七张图及独立证据](handoffs/figures_interim_20261009/README.md)。按85bc7a8执行；当前训练出图搁置，无新仿真，正文不用混合训练图。\n';index.write_text(s,encoding='utf-8')
current=REPO/'docs/collaboration/CURRENT.json';v=read(current);v['temporary_asset_handoff']='lower/handoffs/figures_interim_20261009.md';v['temporary_asset_status']='TEMPORARY_REVIEW_ONLY_REPLACEABLE_NOT_PAPER_PERFORMANCE';v['temporary_asset_run_order']='upper/run_orders/FIGURES_INTERIM_20261009.md';v['current_training_figures']='on hold; no new simulation or checkpoint evaluation';current.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
registry=REPO/'docs/collaboration/registry/EXPERIMENT_REGISTRY_20261007.csv'
with registry.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fieldnames=reader.fieldnames;rows=list(reader)
for r in rows:
    eid=r['experiment_id']
    if eid.startswith('E01_'):r.update(local_result_path='D:/py/DRL2_v3e/logs/v3e_262420..262422/train.monitor.csv; D:/py/DRL2_v3e/eval/v3e/final_training_audit_60k.json',local_sha_verified='n/a',lower_note='2026-10-09 three frozen monitor SHA match run order; every 10k start bin matches audit; historical training only')
    if eid.startswith('E02_'):r.update(local_result_path='D:/py/DRL2/eval/v3e/stage_b/readout_b1.json',local_sha_verified='yes',lower_note='2026-10-09 primary SHA matches 8bd00e8; raw scan remains local; hindsight not online')
    if eid.startswith('E03_'):r.update(local_result_path='D:/py/DRL2/eval/v3e/stage_b/readout_b1b2.json; D:/py/DRL2/eval/v3e/stage_b2',local_sha_verified='yes',lower_note='2026-10-09 primary SHA matches 2b84e14; all 113 success timing records available')
    if eid.startswith('E06_'):r.update(local_result_path='D:/py/DRL2/eval/stopping/formal; C:/Users/35884/Documents/Spacecraft/上层交付/历史/STOPPING_FORMAL_20261006/STOPPING_FORMAL_20261006.zip',local_sha_verified='yes',lower_note='2026-10-09 ZIP SHA matches 94ab2368; 336 raw JSON match prior audit; learned violations 1/0/1 (stopping 0), plotted per user correction')
    if eid.startswith('E10_'):r.update(local_result_path='D:/py/DRL2/eval/regime_screen/w2.36_r15; collab lower/handoffs/final_mainline_screen_20261007/readout_final.json',local_sha_verified='yes',lower_note='2026-10-09 official readout SHA a74603ce matches SHA256.txt; selected cell 96 raw results match counts/medians/failures')
with registry.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader();w.writerows(rows)
receipt={'package':str(DEST),'files':len(files),'bytes':sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file()),'checks_passed':107,'no_new_simulation':True,'old_raw_round_not_duplicated':True,'manifest_sha':sha(DEST/'FILES_SHA256.json')};(TOP/'记录/PUBLICATION_PREPARATION.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(receipt,ensure_ascii=False))
