"""Render snapshot curves, seal and replace current figures; archive old audited evidence."""
import importlib.util,shutil,json,hashlib,sys,csv
from pathlib import Path
from datetime import datetime
from PIL import Image
spec=importlib.util.spec_from_file_location('b',Path(__file__).with_name('make_figures.py'));b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
root=b.TOP/'reports';build=root/'ppt_20261009_latest_build';b.OUT=build;b.DATA=build/'plot_data';b.style()
if '--curves-only' not in sys.argv:assert (build/'FIGURE_REPORT.md').exists(),'Replay/build has not completed'
curves=[]
for seed,color in zip((262460,262461,262462),b.COLORS):
    with (b.DATA/f'train_{seed}_plotted.csv').open(encoding='utf-8-sig',newline='') as f:r=list(csv.DictReader(f))
    a={k:b.np.array([float(row[k]) for row in r]) for k in r[0]};curves.append((seed,color,a))
def panel(ax,kind):
    for seed,color,a in curves:
        x=a['outer_decisions']
        if kind=='return':ax.plot(x,a['return'],color=color,alpha=.18,lw=.8);ax.plot(x,a['return_mean50'],color=color,lw=2.4,label=str(seed))
        else:ax.plot(x,a['completed_mean50_percent'],color=color,lw=2.4,label=str(seed))
    ax.set_xlim(0,60000);ax.set_xticks(b.np.arange(0,60001,10000));ax.xaxis.set_major_formatter(b.ticker.FuncFormatter(lambda v,p:f'{int(v/1000)}k' if v else '0'))
    ax.set_xlabel('累计外层决策步数');ax.set_ylabel('回合回报' if kind=='return' else '训练回合完成率 / %');b.axis(ax)
    if kind=='success':ax.set_ylim(0,100)
    ax.legend(title='训练种子',frameon=False,ncol=3,loc='best')
for kind,title,name in [('return','训练回报','fig_training_return'),('success','训练任务完成率','fig_training_success')]:
    fig,ax=b.plt.subplots(figsize=(12.8,7.2));panel(ax,kind);ax.set_title(title,pad=18);fig.subplots_adjust(left=.10,right=.97,bottom=.15,top=.88);b.save(fig,name)
fig,axs=b.plt.subplots(1,2,figsize=(16,9))
for ax,kind,title in zip(axs,('return','success'),('（a）训练回报','（b）训练任务完成率')):panel(ax,kind);ax.set_title(title,pad=15)
fig.subplots_adjust(left=.07,right=.98,bottom=.14,top=.89,wspace=.28);b.save(fig,'fig_training_combined')
if '--curves-only' in sys.argv:sys.exit(0)
evidence=b.read(b.DATA/'ASSET_VERIFICATION.json');metrics=evidence['results']
rows={'pure':b.np.genfromtxt(b.DATA/'pure_physical_266019.csv',delimiter=',',names=True,encoding='utf-8-sig',dtype=None)}
results={s:b.read(b.DATA/f'stopping_{s}_result_266019.json') for s in (262460,262462)}
for s in results:rows[s]=b.np.genfromtxt(b.DATA/f'stopping_{s}_physical_266019.csv',delimiter=',',names=True,encoding='utf-8-sig',dtype=None)
colors={'pure':'#777777',262460:b.COLORS[0],262462:b.COLORS[2]};labels={'pure':'Pure MPC',262460:'价值协调（262460）',262462:'价值协调（262462）'}
task=b.read(b.MAIN/'logs/final2_262460/manifest.json')['training_environment']['precapture_task']
for fields,ylabels,title,name in [
    (('distance_m','relative_speed_m_s','attitude_error_deg'),('相对距离 / m','相对速度 / (m/s)','姿态误差 / °'),'预捕获状态响应','fig_state_response'),
    (('position_error_m','keepout_margin_m','fov_margin_deg'),('位置误差 / m','安全距离裕度 / m','视场裕度 / °'),'预捕获误差与约束裕度','fig_constraint_response')]:
    fig,axs=b.plt.subplots(3,1,figsize=(12.8,7.2),sharex=True)
    for ax,field,label in zip(axs,fields,ylabels):
        for row,a in rows.items():ax.plot(a['time_s'],a[field],color=colors[row],lw=1.8,ls='--' if row=='pure' else '-',label=labels[row])
        for seed,r in results.items():
            if r['handoff_time_s'] is not None:ax.axvline(r['handoff_time_s'],color=colors[seed],ls=':',lw=1)
        ax.set_ylabel(label);b.axis(ax)
        if field in ('keepout_margin_m','fov_margin_deg'):ax.axhline(0,color='#555555',ls='--',lw=1)
        if field=='position_error_m':ax.axhline(task['completion_position_m'],color='#888888',ls=':',lw=1)
        if field=='attitude_error_deg':ax.axhline(b.np.degrees(task['completion_attitude_rad']),color='#888888',ls=':',lw=1)
    handles,legends=axs[0].get_legend_handles_labels();fig.legend(handles,legends,frameon=False,ncol=3,fontsize=12,loc='upper center',bbox_to_anchor=(.56,.94))
    fig.suptitle(title,y=.985,fontsize=19);axs[-1].set_xlabel('时间 / s');axs[-1].set_xlim(0,max(a['time_s'][-1] for a in rows.values()))
    fig.subplots_adjust(left=.14,right=.97,bottom=.12,top=.84,hspace=.18);b.save(fig,name)
audit='# 新版图表产出价值审核\n\n本版有阶段汇报价值：完整三种子训练曲线呈现实际学习波动，两模型固定开局轨迹呈现真实闭环，价值图显示原部署判据的实际执行。不能据此宣布收敛、价值已校准、协调增益或全块优于Pure。\n\n## 固定案例结果（266019）\n\n| 方法/模型 | 完成且零违规 | 交接步 | 完成时间(s) | Delta-v(m/s) | 末端位置误差(m) | 末端姿态误差(度) |\n|---|---|---:|---:|---:|---:|---:|\n'
for key,r in metrics.items():audit+=f'| {"Pure MPC" if key=="pure" else "50k模型 "+key} | {r["clean_completion"]} | {r["handoff_k"]} | {r["survival_s"]:.1f} | {r["equivalent_delta_v_m_s"]:.4f} | {r["final_position_error_m"]:.4f} | {r["final_attitude_error_deg"]:.3f} |\n'
audit+='\n## 应当怎样讲\n\n训练曲线是过程记录，50回合滑动完成率不是独立评估成功率。轨迹、状态、裕度与价值图合起来说明该固定案例中机制是否触发、如何闭环、约束是否满足；两个种子都展示，不挑更好者。全约束零违规按原评估函数结果，不能只靠两条裕度曲线外推。正式方法成立与协调贡献仍须60k同块learned/stopping/Pure评估。\n\n'
for seed,color,a in curves:audit+=f'- {seed}：快照最后完整回合结束于{int(a["outer_decisions"][-1])}步；最近50回合平均回报{a["return_mean50"][-1]:.3f}，完成率{a["completed_mean50_percent"][-1]:.1f}%。\n'
ref=b.read(b.REF);now=metrics['262460']
audit+=f'\n同一262460模型固定案例由30k到50k：时间{ref["survival_s"]:.1f}→{now["survival_s"]:.1f}s，Delta-v {ref["equivalent_delta_v_m_s"]:.4f}→{now["equivalent_delta_v_m_s"]:.4f}m/s，交接步{ref["handoff_k"]}→{now["handoff_k"]}。这只是同案例的检查点变化，不能外推总体性能改善。\n'
(build/'FIGURE_AUDIT.md').write_text(audit,encoding='utf-8')
for p in build.glob('fig_*.png'):
    with Image.open(p) as im:assert min(im.info['dpi'])>=300
    assert p.with_suffix('.pdf').exists() and p.with_suffix('.svg').exists()
shutil.copy2(Path(__file__),build/Path(__file__).name)
shutil.copy2(Path(__file__).with_name('make_latest_figures.py'),build/'make_latest_figures.py')
shutil.copy2(Path(__file__).with_name('make_figures.py'),build/'make_figures.py')
current=root/'ppt_20261009';archive=root/'历史'/('ppt_20261009_30k_'+datetime.now().strftime('%H%M%S'))
space=Path('C:/Users/35884/Documents/Spacecraft').resolve()
for p in (build,current,archive):assert p.resolve().is_relative_to(space)
assert not archive.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before={p.relative_to(current).as_posix():sha(p) for p in current.rglob('*') if p.is_file()}
archive.parent.mkdir(exist_ok=True);shutil.move(str(current),str(archive))
assert all(sha(archive/rel)==digest for rel,digest in before.items())
shutil.move(str(build),str(current))
receipt={'old_figures_removed_from_current':True,'old_evidence_preserved':str(archive),'old_files_verified':len(before),'current':str(current),'new_figures':[p.name for p in current.glob('fig_*.png')]}
(current/'REPLACEMENT_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
files={p.relative_to(current).as_posix():sha(p) for p in current.rglob('*') if p.is_file() and p.name!='FILES_SHA256.json'}
(current/'FILES_SHA256.json').write_text(json.dumps(files,ensure_ascii=False,indent=2),encoding='utf-8');assert all(sha(current/rel)==digest for rel,digest in files.items())
project=space/'项目说明.md';s=project.read_text(encoding='utf-8');start=s.index('**10月9日汇报图表：**');end=s.index('\n\n',start)
s=s[:start]+'**10月9日汇报图表：** 最新版已更新三种子完整训练日志快照（约50k），并使用262460/262462的50k模型在固定266019开局绘制真实0.1s目标系轨迹、状态与约束响应、价值判据；Pure复用同提交同开局已核验轨迹。原30k版从当前目录移除并核验归档。所有阶段限制与模型SHA见[图表来源报告](C:/Users/35884/Documents/Spacecraft/过程文件/汇报图表_20261009/reports/ppt_20261009/FIGURE_REPORT.md)。图面不混入30k结构统计，不宣称正式性能或训练收敛；训练及科学代码未改。'+s[end:];project.write_text(s,encoding='utf-8')
print('SEALED_AND_REPLACED',json.dumps(receipt,ensure_ascii=False));print('HASHES_VERIFIED',len(files))
