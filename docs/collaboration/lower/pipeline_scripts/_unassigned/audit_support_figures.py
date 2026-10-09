"""Evidence audit and supplementary figures from already completed assets only."""
import importlib.util,json,csv
from pathlib import Path
spec=importlib.util.spec_from_file_location('ppt',Path(__file__).with_name('make_figures.py'))
ppt=importlib.util.module_from_spec(spec);spec.loader.exec_module(ppt)
np=ppt.np;plt=ppt.plt;ppt.style();ppt.guard()
OUT=ppt.OUT;DATA=ppt.DATA
checks={}
for seed in (262460,262461,262462):
    source=ppt.MAIN/f'eval/final2/devcheck/devcheck_{seed}.json'
    content=source.read_bytes();(DATA/f'devcheck_{seed}_snapshot.json').write_bytes(content)
    r=json.loads(content);assert r['openings']==48 and r['all_pass'];checks[seed]=r
labels=list(map(str,checks));d1=[r['D1_not_collapsed'] for r in checks.values()]
zero=np.array([r['handoff_at_k0'] for r in d1]);mid=np.array([r['mid_episode_handoffs'] for r in d1]);never=np.array([r['never_handoff'] for r in d1])
assert np.all(zero+mid+never==48)
gaprange=np.array([r['D3_state_dependent_stopping_value']['median_within_episode_gap_range'] for r in checks.values()])
fig,axs=plt.subplots(1,2,figsize=(12.8,7.2));y=np.arange(3)
left=np.zeros(3)
for values,color,label in [(zero,'#b0b0b0','开局即交接'),(mid,'#004b87','中途交接'),(never,'#006d5b','始终未交接')]:
    axs[0].barh(y,values,left=left,height=.52,color=color,label=label)
    for i,v in enumerate(values):axs[0].text(left[i]+v/2,i,str(v),ha='center',va='center',color='white' if color!='#b0b0b0' else '#222222',fontsize=15)
    left+=values
axs[0].set_yticks(y,labels);axs[0].invert_yaxis();axs[0].set_xlim(0,48);axs[0].set_xticks([0,12,24,36,48]);axs[0].set_xlabel('开局数');axs[0].set_ylabel('训练种子');axs[0].set_title('交接行为分布',pad=17)
axs[0].legend(frameon=False,ncol=3,fontsize=12,columnspacing=1.1,loc='upper center',bbox_to_anchor=(.5,-.20));ppt.axis(axs[0]);axs[0].grid(axis='y',visible=False)
axs[1].barh(y,gaprange,color=ppt.COLORS,height=.52)
for i,v in enumerate(gaprange):axs[1].text(v+.15,i,f'{v:.2f}',va='center',fontsize=15)
axs[1].axvline(1,color='#777777',ls='--',lw=1.2,label='结构检查阈值')
axs[1].set_yticks(y,labels);axs[1].invert_yaxis();axs[1].set_xlim(0,11);axs[1].set_xlabel('回合内价值差幅度中位数');axs[1].set_title('价值差随状态发生变化',pad=17);ppt.axis(axs[1]);axs[1].grid(axis='y',visible=False)
axs[1].legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.20))
agree=[r['D4_stopping_matches_value_rule'] for r in checks.values()];assert all(r['agreeing']==r['decisions'] for r in agree)
fig.text(.5,.035,'判据逐点一致：'+ '   ·   '.join(f'{s}：{r["agreeing"]}/{r["decisions"]}' for s,r in zip(checks,agree)),ha='center',fontsize=13)
fig.subplots_adjust(left=.09,right=.97,top=.87,bottom=.29,wspace=.38);ppt.save(fig,'fig_structural_evidence')

rows={row:ppt.physical(row) for row in ('pure','stopping')};st=ppt.read(DATA/'stopping_result_266019.json');ht=st['handoff_time_s']
manifest=ppt.read(ppt.MAIN/'logs/final2_262460/manifest.json');task=manifest['training_environment']['precapture_task']
fig,axs=plt.subplots(3,1,figsize=(12.8,7.2),sharex=True)
for ax,field,label in zip(axs,('position_error_m','keepout_margin_m','fov_margin_deg'),('位置误差 / m','安全距离裕度 / m','视场裕度 / °')):
    for row,a in rows.items():ax.plot(a['time_s'],a[field],color=ppt.METHOD_COLORS[row],lw=1.9,ls='--' if row=='pure' else '-',label='Pure MPC' if row=='pure' else '价值协调 SAC–MPC')
    ax.axvline(ht,color='#d98c00',ls=':',lw=1.3);ppt.axis(ax);ax.set_ylabel(label)
axs[0].axhline(task['completion_position_m'],color='#888888',ls=':',lw=1)
for ax in axs[1:]:ax.axhline(0,color='#555555',ls='--',lw=1.1)
axs[0].set_title('预捕获误差与约束裕度',pad=15);axs[0].legend(frameon=False,ncol=2,loc='upper right');axs[0].text(ht,.97,'交接',transform=axs[0].get_xaxis_transform(),color='#ad6b00',ha='left',va='top')
axs[-1].set_xlabel('时间 / s');axs[-1].set_xlim(0,max(a['time_s'][-1] for a in rows.values()));fig.subplots_adjust(left=.13,right=.97,top=.89,bottom=.12,hspace=.2);ppt.save(fig,'fig_constraint_response')
metrics={}
for row,a in rows.items():
    r=ppt.read(DATA/f'{row}_result_266019.json')
    metrics[row]={k:r[k] for k in ('survival_s','equivalent_delta_v_m_s','completed','zero_violation')}
    metrics[row].update(position_error_final_m=float(a['position_error_m'][-1]),attitude_error_final_deg=float(a['attitude_error_deg'][-1]),min_keepout_margin_m=float(a['keepout_margin_m'].min()),min_fov_margin_deg=float(a['fov_margin_deg'].min()))
pure=metrics['pure'];hybrid=metrics['stopping'];extra_time=100*(hybrid['survival_s']/pure['survival_s']-1);extra_dv=100*(hybrid['equivalent_delta_v_m_s']/pure['equivalent_delta_v_m_s']-1)
ppt.write(DATA/'FIGURE_EVIDENCE_AUDIT.json',{'structure':{str(s):r for s,r in checks.items()},'single_case':metrics,'hybrid_time_increase_percent':extra_time,'hybrid_delta_v_increase_percent':extra_dv,'additional_experiments':0})
report=f'''# 图表科学支撑审核\n\n## 审核结论\n\n这些图有阶段汇报价值，但不能用于宣布训练收敛或优于Pure MPC。原训练曲线显示三种子分化，并非一致改善；三维目标系曲线也含坐标系旋转效应，不能把弧线直接解读为更优路径。\n\n现有证据支持三项有限结论：①原控制与模型代码成功执行了状态相关、单向的价值交接；②三个模型在48开局结构检查中均不是固定开局停或永不停止，且全部学习段决策的部署判据一致；③固定案例交接后误差继续下降，并在任务完成阈值内终止，距离与视场裕度在记录中均保持正值。零违规结论来自原官方结果核验，不由两条裕度曲线代替全部约束判读。\n\n补充图fig_structural_evidence用全部三种子48开局官方读数，表示结构，不是性能排行榜；中途交接27/27/30，开局停11/16/14，从不交接10/5/4；逐点判据一致2096/2096、1566/1566、1347/1347。回合内价值差幅度中位数9.68/6.20/6.44，只说明变化，没有证明价值预测已校准。\n\n补充图fig_constraint_response用已核验两次物理重放的原CSV，不开展额外仿真。位置误差阈值取真实配置{task['completion_position_m']}m，另两图零线为裕度边界，竖线是60s真实交接。\n\n必须如实披露：本案例Pure MPC在106.8s、Delta-v=1.34053m/s干净完成，协调方案143.6s、Delta-v=1.82373m/s干净完成；协调时间增加{extra_time:.1f}%、Delta-v增加{extra_dv:.1f}%。所以此例能解释交接发生与交接后的闭环，不能证明交接带来效率收益或救回Pure失败。案例已事先固定，不为了汇报换成漂亮样本。该结果也不能由单案例外推为方法总体失败。\n\n价值图增加阈值附近的原数据放大：k29差值-0.03113，k30差值+0.05073，首次触发与k30核验一致。无重新平滑、移位或改阈值。\n\n图面保持干净，版本和限制集中在本报告及FIGURE_REPORT.md。正式方法贡献仍以60k最终模型、同块Pure/learned/stopping的预注册评估判定。\n'''
(OUT/'FIGURE_AUDIT.md').write_text(report,encoding='utf-8')
import shutil
shutil.copy2(Path(__file__),OUT/Path(__file__).name)
print('EVIDENCE_AUDIT',json.dumps({'extra_time_percent':extra_time,'extra_delta_v_percent':extra_dv}),flush=True)
