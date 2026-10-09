from pathlib import Path
import json,csv,hashlib,shutil
from datetime import datetime
from PIL import Image
TOP=Path(__file__).resolve().parents[1];OUT=TOP/'reports/figures_interim_20261009';SPACE=Path('C:/Users/35884/Documents/Spacecraft');REPO=SPACE/'过程文件/协作/Git工作树';MAIN=Path('D:/py/DRL2')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
zipfile=SPACE/'上层交付/历史/STOPPING_FORMAL_20261006/STOPPING_FORMAL_20261006.zip';zipsha=sha(zipfile);assert zipsha=='94ab2368ea2d50d35aecc67230e0a6e37606dd5047d4ada202e316e33583f2ae'
for path in (MAIN/'eval/stopping/formal').rglob('seed_*.json'):
    relative=path.relative_to(MAIN/'eval/stopping/formal');old=SPACE/'过程文件/停止头/审查材料_20261006/raw/formal'/relative
    if 267000<=int(path.stem.split('_')[1])<=267047:assert old.is_file() and sha(path)==sha(old),str(path)
checks=read(OUT/'CHECKS.json');assert all(r['passed'] for r in checks)
names=['fig1_v3e_training','fig2_formal_comparison','fig3_baseline_time_energy','fig4_hindsight_timing','fig_trajectory_3d','fig_state_response','fig_handoff_value']
for name in names:
    with Image.open(OUT/(name+'.png')) as im:assert min(im.info['dpi'])>=300
    assert (OUT/(name+'.pdf')).is_file() and (OUT/(name+'.svg')).is_file()
order=REPO/'docs/collaboration/upper/run_orders/FIGURES_INTERIM_20261009.md';shutil.copy2(order,OUT/'RUN_ORDER.md')
report='''# 阶段基准图：独立审核与展示口径

状态：TEMPORARY_REVIEW_ONLY / NOT_PAPER_PERFORMANCE。执行单85bc7a8；历史正式结果保留其原结论，本包是阶段汇报资产，不把新工况中间模型放进论文性能表。只读取已有文件，无新仿真、评估、检查点评估，当前训练与科学代码未修改。

## 核验及产出价值

1. **V3e训练图可用于说明历史学习层确实学到了任务行为。** 三份60k日志SHA与执行单逐一相同，按回合起点复算全部10k箱与审计JSON完全一致；图按5k箱作三种子细线、等权均值及最小最大带，无原始细线。10k箱三种子等权平均完成率28.7%→77.1%；它不是当前价值停止方法的训练改善或确定性部署性能。模型262420–262422，科学提交a03632a。
2. **267000正式基准图可用于说明相同策略加入停止头后的历史净完成增益。** Pure41，learned12/31/17，stopping37/41/36；增益+25/+10/+19，正式METHOD_DOES_NOT_HOLD。Pure与stopping违规0，learned1/0/1；用户已明确授权按实际绘制。336原始结果逐份与独立审查原件SHA一致，原ZIP SHA也核验相同。模型262430–262432，科学提交2e5c236。每行箱线图只看自身干净成功样本，样本不同不能单凭中位数外推方法效率优势；同时提供与Pure共同成功以及同模型stopping/learned共同成功的配对CSV。
3. **269000散点能展示两条基线的速度、能耗及失败模式差别。** Pure34/48、nominal45/48，中位时间98.7/196.3s；Pure14个失败均为超时，nominal3个失败为距离失败，失败全部画×。它是w2.36_r15的冻结工况筛选证据，科学提交867a3a1。271000/272000完整基线不存在，未补跑，不放任何学习方法行。
4. **B1/B2能支撑事后交接机会与时机价值，不能支撑在线方法成立。** 262000块Pure37，学习42/33/38，全部学习失败均有干净救回机会，全部学习成功均有更快干净交接，构成每模型事后48/48上界。262420节省时间中位59.7s、对应Delta-v变化中位−0.38442m/s；两分量中位数不一定来自同一开局。散点完整展示113个学习成功开局，保留能耗增加的点，不以最省时交接当作最省能耗。扫描a714c61、读数修订版本与该SHA固定JSON区分。
5. **266019案例只展示价值规则、交接位置与闭环响应。** 只重绘此前已核验0.1s导出，完全没有再仿真。两个50k中间模型262460/262462，科学提交f2f8169；交接21/39步、42/78s，133.9/161.1s完成，均零违规，仍慢于Pure106.8s。第一栏从质心相对距离改为真实预捕获位置误差，使0.25m捕获阈值正确；另两栏阈值0.05m/s和10°，均来自配置。完整完成还包含角速度及保持时间，三条阈值线不替代全部条件；完成标志来自既有官方结果。三维图为目标本体系，坐标旋转曲线不直接代表惯性路径优劣，预捕获不是物理接触。

## 本轮剔除和保留

旧final2混合训练回报/完成率不进入本版，当前三种子的出图搁置；不画过程图、违规训练曲线、累计能耗或缺数据的正式方法图。备份页的条件单飞与交接比例为可选，本轮不新增，历史不利诊断仍在342f2e6固定入口，不抹除。旧本地演示入口将归档并标为被本版替换，数据哈希保留。

## 绘图复用

图2/3使用plot_interim.py的load_block、comparison、baseline_scatter，直接读取每行48个seed JSON。正式结果全部完成并通过完整性核验后，按CLI参数接入，不改变原始结果。当前脚本仅用于绘图，绝不启动仿真。使用说明见REUSE.md；未来典型轨迹仅登记编号规则，未运行：每模型271000中“救回、毁掉、共同成功”各取编号最小开局。

## 独立审查证据

plot_data包含所有画入的数值CSV及同块配对CSV，INPUTS_SHA256列出源绝对路径、字节SHA、当地只读快照。公开包不再复制已有336历史JSON和96基线JSON，完整逐回合指标在绘图CSV中，源SHA与固定历史审查/读数链接保留；训练原CSV、审计JSON和B1/B2读数及案例导出随包提供。FILES_SHA256用于本包字节核验，不等同于模型哈希。
'''
(OUT/'REVIEW.md').write_text(report,encoding='utf-8')
(OUT/'REUSE.md').write_text('''# 正式完成后复用（现在不执行）

项目Python解释器：D:/py/DRL2/.venv/Scripts/python.exe。脚本不会启停训练、执行仿真或修改科学工作区。

输入结构：root/pure/seed_<开局>.json，root/learned/<模型>/seed_<开局>.json，root/stopping/<模型>/seed_<开局>.json；基线模式再读root/nominal/seed_<开局>.json。每行要求48开局、seed/row吻合、code_dirty=false。应先做当轮正式的commit、模型SHA、工况与完整性核验，再绘图；绘图不替代官方判读。

```powershell
& 'D:/py/DRL2/.venv/Scripts/python.exe' -B './plot_interim.py' --generic-root 'D:/py/DRL2/eval/final2/formal' --block 271000 --models 262460 262461 262462 --caption '271000–271047，w2.36_r15，60k正式模型，结论按当轮官方判读'
& 'D:/py/DRL2/.venv/Scripts/python.exe' -B './plot_interim.py' --generic-root '<该块完整基线目录>' --block 272000 --baseline-only --caption '272000–272047，w2.36_r15，Pure与nominal，同块基线'
```

复用时输出TOP为脚本父目录的父目录，请将脚本先复制到下一主题的脚本/目录，防止覆盖本临时资产。画失败回合的真实结束时间，无成功筛选；正式方法箱线图同时输出共同成功配对，不用自己的成功样本组成掩盖效率比较。

论文轨迹编号固定规则：每个模型在271000–271047中分别找Pure失败/stopping成功、Pure成功/stopping失败、两者均成功的开局，各取编号最小的一个；这里成功指干净完成。类别为空则如实报告，不换块、不手挑。只有正式结果出来且重放授权到位后才能导出新物理轨迹。
''',encoding='utf-8')
labels=[('fig1_v3e_training','1 · V3e历史学习层训练'),('fig2_formal_comparison','2 · 原工况正式完成数/能耗/时间'),('fig3_baseline_time_energy','3 · 新工况Pure与nominal'),('fig4_hindsight_timing','4 · 事后交接时机价值'),('fig_trajectory_3d','5a · 已有50k案例轨迹'),('fig_state_response','5b · 案例闭环与捕获阈值'),('fig_handoff_value','5c · 案例价值触发')]
index='# 阶段汇报基准图（临时、待审查）\n\nTEMPORARY_REVIEW_ONLY / NOT_PAPER_PERFORMANCE。执行单85bc7a8，五项共七张图；仅已有资产，无新实验，训练不动。历史正式与新方法中间案例分开注明，不能混成当前方法已成立。\n\n'
for name,label in labels:index+=f'- [{label}]({name}.png) · [PDF]({name}.pdf) · [SVG]({name}.svg)\n'
index+='\n[核对数字](CHECKS.md) · [科学支撑与限制](REVIEW.md) · [实际绘图CSV](plot_data/) · [源文件及SHA](INPUTS_SHA256.json) · [脚本复用](REUSE.md) · [全文件SHA](FILES_SHA256.json)。\n\n按真实learned违规1/0/1绘制已获用户明确授权。当前final2混合训练曲线剔除，旧图仅历史留存。后续更好的正式资产可替换本入口，不能抹掉不利证据。\n'
(OUT/'README.md').write_text(index,encoding='utf-8')
with (OUT/'ASSET_INVENTORY.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['experiment','local_path','available','hash_check','role']);w.writerows([['E01','D:/py/DRL2_v3e/logs/v3e_262420..262422/train.monitor.csv','3 complete logs','matches frozen SHA','historical training'],['E02/E03','D:/py/DRL2/eval/v3e/stage_b/readout_b1b2.json','113 success trajectories + failure gates','matches 2b84e14','hindsight'],['E06','D:/py/DRL2/eval/stopping/formal','336 raw JSON','each matches prior independent audit; ZIP matches','official negative'],['E10','D:/py/DRL2/eval/regime_screen/w2.36_r15','96 raw JSON','counts/median/failures match','frozen screening baselines'],['F03/F04','D:/py/DRL2/eval/final2/formal','absent','not available','no new plots'],['case','342f2e6 frozen collab package','two 50k exports + Pure','all match package SHA','single development opening']])
(OUT/'EXTRA_INTEGRITY.json').write_text(json.dumps({'formal_zip_sha256':zipsha,'formal_raw_files_match_independent_review':336,'checks_passed':len(checks),'simulations_started':0,'training_modified':False,'user_correction':'按真实1／0／1标注，继续图2'},ensure_ascii=False,indent=2),encoding='utf-8')
shutil.copy2(Path(__file__),OUT/Path(__file__).name)
files={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='FILES_SHA256.json'};(OUT/'FILES_SHA256.json').write_text(json.dumps(files,ensure_ascii=False,indent=2),encoding='utf-8');assert all(sha(OUT/n)==v for n,v in files.items())
# Only closed local figure assets are moved; no logs/state or scientific directories.
old=SPACE/'过程文件/汇报图表_20261009/reports/ppt_20261009';archive=old.parent/'历史'/('ppt_superseded_by_interim_'+datetime.now().strftime('%H%M%S'))
assert old.resolve().is_relative_to(SPACE.resolve()) and archive.resolve().is_relative_to(SPACE.resolve()) and not archive.exists()
before={p.relative_to(old).as_posix():sha(p) for p in old.rglob('*') if p.is_file()};shutil.move(str(old),str(archive));assert all(sha(archive/n)==h for n,h in before.items())
project=SPACE/'项目说明.md';s=project.read_text(encoding='utf-8');start=s.index('**10月9日汇报图表：**');end=s.index('\n\n',start);s=s[:start]+'**10月9日汇报图表：** 按FIGURES_INTERIM_20261009执行，五项七张基准图已核验：V3e历史训练、267000正式负结果、269000基线、B1/B2事后时机、既有50k单案例。当前训练出图搁置，无新仿真；旧图入口已归档。入口：[新版图表与完整审核](C:/Users/35884/Documents/Spacecraft/过程文件/阶段基准图_20261009/reports/figures_interim_20261009/README.md)。必须区分历史学习层上升与当前价值停止方法；learned违规实际1/0/1按用户确认标注。'+s[end:];project.write_text(s,encoding='utf-8')
(TOP/'记录/LOCAL_DELIVERY.json').write_text(json.dumps({'current':str(OUT),'archived_previous':str(archive),'archived_files_verified':len(before),'files_verified':len(files),'checks_passed':len(checks)},ensure_ascii=False,indent=2),encoding='utf-8');print('DELIVERY_VERIFIED',len(files),'files;',len(checks),'checks')
