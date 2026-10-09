"""Read-only 50k demonstration: fixed opening, no screening, no training."""
import importlib.util,inspect,shutil,json,csv,hashlib
from pathlib import Path
spec=importlib.util.spec_from_file_location('base',Path(__file__).with_name('make_figures.py'))
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
b.OUT=b.TOP/'reports/ppt_20261009_latest_build';b.DATA=b.OUT/'plot_data'
b.DATA.mkdir(parents=True,exist_ok=True)
np=b.np;plt=b.plt
b.guard();b.style()
# Reuse the reviewed monitor parser, cumulative outer decisions and 50-episode windows.
curve_source=inspect.getsource(b.curves).replace('30000','60000').replace('30001,5000','60001,10000')
exec(curve_source,b.__dict__)
b.curves()
old=b.TOP/'reports/ppt_20261009/plot_data'
for name in ('pure_physical_266019.csv','pure_result_266019.json'):
    shutil.copy2(old/name,b.DATA/name)
b.write(b.DATA/'pure_provenance.json',{'source':str(old),'reuse':'previously verified exact same opening/environment/pure controller','source_sha256':{n:b.sha(old/n) for n in ('pure_physical_266019.csv','pure_result_266019.json')}})

# Retain the original environment and evaluator plus read-only post-step instrumentation.
replay_source=inspect.getsource(b.replay).split('    ref=read(REF);actual=results')[0]
replay_source=replay_source.replace("for row in ('pure','stopping'):","for row in ('stopping',):")
replay_source=replay_source.replace("run=MAIN/'logs/final2_262460'","run=MAIN/f'logs/final2_{CURRENT_SEED}'")
replay_source=replay_source.replace('stopping_30000_outer_decisions.zip','stopping_50000_outer_decisions.zip')
exec(replay_source,b.__dict__)
models={};results={}
pure=b.read(b.DATA/'pure_result_266019.json')
pure_physical=np.genfromtxt(b.DATA/'pure_physical_266019.csv',delimiter=',',names=True,encoding='utf-8-sig',dtype=None)
for seed in (262460,262462):
    checkpoint=b.MAIN/f'logs/final2_{seed}/checkpoints/stopping_50000_outer_decisions.zip'
    b.CURRENT_SEED=seed;b.CP_SHA=b.sha(checkpoint)
    # Original guard checks the previous frozen model; retain that guard unchanged.
    original_guard=b.guard
    def guarded():
        active_sha=b.CP_SHA;b.CP_SHA='37bf65b864305adc403d859dd3b4a79590684e9efa32c4e895cf3d34e3380f9c'
        try:original_guard()
        finally:b.CP_SHA=active_sha
        assert b.sha(checkpoint)==active_sha
    b.guard=guarded
    b.replay()
    b.guard=original_guard
    result=b.read(b.DATA/'stopping_result_266019.json')
    physical=np.genfromtxt(b.DATA/'stopping_physical_266019.csv',delimiter=',',names=True,encoding='utf-8-sig',dtype=None)
    gap=np.array(result['trace']['q_handoff'])-np.array(result['trace']['q_continue'])
    hk=result['handoff_k']
    checks={'same_initial_condition':all(physical[k][0]==pure_physical[k][0] for k in ('x_m','y_m','z_m','vx_m_s','vy_m_s','vz_m_s','attitude_error_deg')),
            'physical_time_matches_result':bool(np.isclose(physical['time_s'][-1],result['survival_s'],atol=1e-8)),
            'checkpoint_unchanged':b.sha(checkpoint)==b.CP_SHA,
            'first_value_threshold_matches_handoff':bool((hk is None and np.all(gap<0)) or (hk is not None and gap[hk]>=0 and np.all(gap[:hk]<0)))}
    assert all(checks.values()),checks
    for suffix in ('physical_266019.csv','result_266019.json'):
        (b.DATA/f'stopping_{suffix}').rename(b.DATA/f'stopping_{seed}_{suffix}')
    results[seed]=result;models[seed]={'checkpoint':str(checkpoint),'sha256':b.CP_SHA,'outer_decisions':50000,'checks':checks}
    b.write(b.DATA/f'REPLAY_VERIFICATION_{seed}.json',{'model':models[seed],'result':result,'comparison':'50k has no prior development JSON; verified against original evaluator return and physical trace, not against 30k outcome'})
    print('VERIFIED_50K',seed,json.dumps({k:result[k] for k in ('handoff_k','survival_s','clean_completion','equivalent_delta_v_m_s')}) ,flush=True)

rows={'pure':pure_physical}
for seed in results:rows[seed]=np.genfromtxt(b.DATA/f'stopping_{seed}_physical_266019.csv',delimiter=',',names=True,encoding='utf-8-sig',dtype=None)
colors={'pure':'#777777',262460:b.COLORS[0],262462:b.COLORS[2]}
labels={'pure':'Pure MPC',262460:'价值协调（262460）',262462:'价值协调（262462）'}
desired=np.asarray(results[262460]['geometry']['desired_position']);port=np.asarray(results[262460]['geometry']['port_position'])
fig=plt.figure(figsize=(12.8,7.2));ax=fig.add_subplot(111,projection='3d');points=[]
for row,a in rows.items():
    p=np.column_stack([a[k] for k in ('x_m','y_m','z_m')]);points.append(p)
    ax.plot(*p.T,color=colors[row],lw=2,ls='--' if row=='pure' else '-',label=labels[row]);ax.scatter(*p[-1],color=colors[row],s=35,depthshade=False)
    if row!='pure' and results[row]['handoff_position'] is not None:
        hp=np.asarray(results[row]['handoff_position']);points.append(hp.reshape(1,3));ax.scatter(*hp,marker='D',s=60,color=colors[row],depthshade=False,label=f'交接点（{row}）')
ax.scatter(*points[0][0],marker='*',s=140,color='#333333',label='共同起点',depthshade=False)
ax.scatter(0,0,0,marker='s',s=50,color='#555555',label='目标质心',depthshade=False)
ax.scatter(*port,marker='^',s=55,color='#ad6b00',label='捕获端口',depthshade=False)
ax.scatter(*desired,marker='x',s=65,color='#ad6b00',label='预捕获目标位置',depthshade=False)
p=np.vstack([*points,desired,port,np.zeros(3)]);center=(p.min(0)+p.max(0))/2;rad=np.max(p.max(0)-p.min(0))*.55
for f,c in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),center):f(c-rad,c+rad)
ax.set_box_aspect((1,1,1));ax.view_init(elev=24,azim=-57)
for a in (ax.xaxis,ax.yaxis,ax.zaxis):a.set_major_locator(b.ticker.MaxNLocator(nbins=5));a.set_pane_color((1,1,1,0));a._axinfo['grid'].update(color='#e6e6e6',linewidth=.5)
ax.set_xlabel('目标系 X / m',labelpad=12);ax.set_ylabel('目标系 Y / m',labelpad=12);ax.set_zlabel('目标系 Z / m',labelpad=9);ax.set_title('预捕获三维轨迹',pad=18)
ax.legend(loc='upper left',bbox_to_anchor=(1.16,.96),frameon=False,fontsize=12);fig.subplots_adjust(left=.02,right=.76,bottom=.17,top=.90);b.save(fig,'fig_trajectory_3d')
for fields,ylabels,title,name in [
    (('distance_m','relative_speed_m_s','attitude_error_deg'),('相对距离 / m','相对速度 / (m/s)','姿态误差 / °'),'预捕获状态响应','fig_state_response'),
    (('position_error_m','keepout_margin_m','fov_margin_deg'),('位置误差 / m','安全距离裕度 / m','视场裕度 / °'),'预捕获误差与约束裕度','fig_constraint_response')]:
    fig,axs=plt.subplots(3,1,figsize=(12.8,7.2),sharex=True)
    for ax,field,label in zip(axs,fields,ylabels):
        for row,a in rows.items():ax.plot(a['time_s'],a[field],color=colors[row],lw=1.8,ls='--' if row=='pure' else '-',label=labels[row])
        for seed,r in results.items():
            if r['handoff_time_s'] is not None:ax.axvline(r['handoff_time_s'],color=colors[seed],ls=':',lw=1)
        ax.set_ylabel(label);b.axis(ax)
        if field in ('keepout_margin_m','fov_margin_deg'):ax.axhline(0,color='#555555',ls='--',lw=1)
    axs[0].legend(frameon=False,ncol=3,fontsize=12,loc='upper right');axs[0].set_title(title,pad=15);axs[-1].set_xlabel('时间 / s');axs[-1].set_xlim(0,max(a['time_s'][-1] for a in rows.values()))
    fig.subplots_adjust(left=.14,right=.97,bottom=.12,top=.89,hspace=.18);b.save(fig,name)

fig,axs=plt.subplots(1,2,figsize=(16,9))
for ax,(seed,r) in zip(axs,results.items()):
    gap=np.array(r['trace']['q_handoff'])-np.array(r['trace']['q_continue']);k=np.arange(len(gap));hk=r['handoff_k']
    ax.plot(k,gap,color=colors[seed],lw=2.2,marker='o',ms=3);ax.axhline(0,color='#555555',ls='--',lw=1,label='交接阈值')
    if hk is not None:
        ax.scatter(hk,gap[hk],color='#d98c00',s=75,zorder=5);ax.axvline(hk,color='#d98c00',ls=':',lw=1.2)
        ax.annotate(f'交接 k={hk}',xy=(hk,gap[hk]),xytext=(-100,25),textcoords='offset points',color='#ad6b00',arrowprops={'arrowstyle':'->','color':'#ad6b00'})
    else:ax.text(.04,.90,'本回合未交接',transform=ax.transAxes)
    ax.set_xlabel('外层决策编号 k');ax.set_ylabel('交接价值差（Q_H − Q_C）');ax.set_title(f'模型 {seed}',pad=20);b.axis(ax)
    with (b.DATA/f'handoff_value_{seed}_266019.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['decision_k','Q_H_minus_Q_C']);w.writerows(zip(k,gap))
fig.subplots_adjust(left=.08,right=.97,bottom=.14,top=.88,wspace=.27);b.save(fig,'fig_handoff_value')

sources=b.read(b.DATA/'training_sources.json');metrics={}
for row,a in rows.items():
    r=pure if row=='pure' else results[row]
    metrics[str(row)]={k:r[k] for k in ('completed','zero_violation','clean_completion','survival_s','equivalent_delta_v_m_s','handoff_k')}
    metrics[str(row)].update(final_position_error_m=float(a['position_error_m'][-1]),final_attitude_error_deg=float(a['attitude_error_deg'][-1]),minimum_keepout_margin_m=float(a['keepout_margin_m'].min()),minimum_fov_margin_deg=float(a['fov_margin_deg'].min()))
b.write(b.DATA/'ASSET_VERIFICATION.json',{'code_commit':b.COMMIT,'regime':'w2.36_r15','opening':266019,'models':models,'training_sources':sources,'results':metrics,'new_simulations':2,'pure_reused':True})
report='# 最新汇报图表与科学审核\n\n本版替换原30k图片。科学提交 '+b.COMMIT+'；工况w2.36_r15。三种子训练曲线来自追加日志的独立完整行快照，累计episode长度表示外层决策数，全部展示至各自当前进度，横轴预留到60k，绝不延长数据。原始回报浅线，50回合尾随均值；完成率也是50回合窗口，任务完成不等同零违规，不宣称已收敛。\n\n'
report+='轨迹使用262460、262462的冻结50k模型，在已固定266019开局各做一次确定性重放；没有换案例或额外48开局评估。Pure复用此前同提交同开局已核验物理数据；两新模型采用原科学环境、加载器和官方决策函数，以低优先级单线程逐一重放。仅给当前重放实例加只读采样，真实0.1s数据，不平滑，不改控制。50k没有对应原开发JSON，不能套用30k的k30/143.6s结论；本版依据原评估函数返回、物理终点、相同初态、首次价值阈值与模型前后SHA核验。详细检查见plot_data。\n\n'
report+='三维轨迹是目标本体系，弧线含目标旋转效应，不能直接解释为更优路径。预捕获完成不是物理接触。价值越零只支持部署规则执行，不证明价值已校准或交接贡献。没有learned-only同模型同块对照，不能宣称协调增益；单案例也不能代替预注册正式性能。30k结构统计保留在历史版，不放进本版冒充50k结构验证。\n\n## 资产与结果\n\n```json\n'+json.dumps({'models':models,'training_sources':sources,'results':metrics},ensure_ascii=False,indent=2)+'\n```\n\n所有PNG400dpi，另附PDF/SVG。脚本和快照随版保存。训练进程、主工程及正式结果未改。\n'
(b.OUT/'FIGURE_REPORT.md').write_text(report,encoding='utf-8')
(b.OUT/'FIGURE_AUDIT.md').write_text(report,encoding='utf-8')
shutil.copy2(Path(__file__),b.OUT/Path(__file__).name);shutil.copy2(Path(__file__).with_name('make_figures.py'),b.OUT/'make_figures.py')
b.write(b.OUT/'FILES_SHA256.json',{p.relative_to(b.OUT).as_posix():b.sha(p) for p in b.OUT.rglob('*') if p.is_file() and p.name!='FILES_SHA256.json'})
print('LATEST_BUILD_COMPLETE',str(b.OUT),flush=True)
