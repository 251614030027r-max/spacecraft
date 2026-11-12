"""Frozen PPT figures: monitor snapshots and exactly two deterministic rollouts."""
import os,sys,json,csv,hashlib,subprocess,time,ctypes
from pathlib import Path
from datetime import datetime
TOP=Path('C:/Users/35884/Documents/Spacecraft/过程文件/汇报图表_20261009')
OUT=TOP/'reports/ppt_20261009'; DATA=OUT/'plot_data'; REC=TOP/'记录'
for d in (OUT,DATA,REC):d.mkdir(parents=True,exist_ok=True)
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR=str(REC/'matplotlib'))
if os.name=='nt':
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=ctypes.c_void_p
    kernel.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
    if not kernel.SetPriorityClass(kernel.GetCurrentProcess(),0x4000):raise OSError('Cannot set below-normal priority')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,ticker
MAIN=Path('D:/py/DRL2');sys.path.insert(0,str(MAIN))
COMMIT='f2f8169acd580ea96a578d45022dd8952447eca5'
CP=MAIN/'logs/final2_262460/checkpoints/stopping_30000_outer_decisions.zip'
CP_SHA='37bf65b864305adc403d859dd3b4a79590684e9efa32c4e895cf3d34e3380f9c'
REF=MAIN/'eval/final2/devcheck/262460/seed_266019.json'
REF_SHA='7e092499169281a6f2e36d734606ade347b8cc20517122d124f20f91c9a02091'
COLORS=['#004b87','#b22222','#006d5b'];METHOD_COLORS={'pure':'#b22222','stopping':'#004b87'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def guard():
    def git(*a):return subprocess.check_output(['git','-c',f'safe.directory={MAIN}','-C',str(MAIN),*a],encoding='utf-8').strip()
    assert git('rev-parse','HEAD')==COMMIT and not git('status','--porcelain','--untracked-files=no')
    assert sha(CP)==CP_SHA and sha(REF)==REF_SHA
    for seed in (262460,262461,262462):
        m=read(MAIN/f'logs/final2_{seed}/manifest.json')
        assert m['code_commit']==COMMIT and not m['code_dirty'] and m['regime']['name']=='w2.36_r15'
        assert m['stopping']['stop_rule']=='value' and m['stopping']['bellman_stop_value']=='soft'
def style():
    font_manager.findfont('SimSun',fallback_to_default=False)
    plt.rcParams.update({'font.family':['Times New Roman','SimSun'],'mathtext.fontset':'stix','axes.unicode_minus':False,
        'font.size':15,'axes.labelsize':17,'axes.titlesize':19,'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':14,
        'axes.linewidth':1,'xtick.direction':'in','ytick.direction':'in','figure.facecolor':'white','axes.facecolor':'white',
        'savefig.dpi':400,'pdf.fonttype':42,'svg.fonttype':'none'})
def axis(ax):
    ax.tick_params(top=True,right=True);ax.grid(color='#e6e6e6',ls='--',lw=.6,alpha=.6);ax.set_axisbelow(True)
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=400,facecolor='white')
    fig.savefig(OUT/(name+'.pdf'));fig.savefig(OUT/(name+'.svg'));plt.close(fig)
    print('FIGURE_SAVED',name,flush=True)
def rolling(y,n=50):
    y=np.asarray(y,float);z=np.full(len(y),np.nan)
    if len(y)>=n:z[n-1:]=np.convolve(y,np.ones(n)/n,mode='valid')
    return z
def curves():
    guard();style();curves=[];sources={}
    for seed,color in zip((262460,262461,262462),COLORS):
        source=MAIN/f'logs/final2_{seed}/train.monitor.csv';snapshot=DATA/f'train_{seed}_snapshot.csv'
        raw=source.read_bytes();raw=raw[:raw.rfind(b'\n')+1];snapshot.write_bytes(raw)
        with snapshot.open(newline='',encoding='utf-8-sig') as f:f.readline();rows=list(csv.DictReader(f))
        x=np.cumsum([int(r['l']) for r in rows]);mask=x<=30000
        y=np.array([float(r['r']) for r in rows])[mask];success=np.array([r['completed']=='True' for r in rows],float)[mask];x=x[mask]
        assert len(x)>50 and np.isfinite(y).all()
        curves.append((seed,color,x,y,rolling(y),success*100,rolling(success)*100))
        sources[str(seed)]={'source':str(source),'snapshot_sha256':sha(snapshot),'episodes_shown':len(x),'last_outer_decision':int(x[-1])}
        with (DATA/f'train_{seed}_plotted.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.writer(f);writer.writerow(['outer_decisions','return','return_mean50','completed_percent','completed_mean50_percent'])
            writer.writerows(zip(x,y,rolling(y),success*100,rolling(success)*100))
    write(DATA/'training_sources.json',sources)
    def panel(ax,kind):
        for seed,color,x,y,ys,c,cs in curves:
            if kind=='return':
                ax.plot(x,y,color=color,alpha=.18,lw=.8);ax.plot(x,ys,color=color,lw=2.4,label=str(seed))
            else:ax.plot(x,cs,color=color,lw=2.4,label=str(seed))
        ax.set_xlim(0,30000);ax.set_xticks(np.arange(0,30001,5000));ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v,p:f'{int(v/1000)}k' if v else '0'))
        ax.set_xlabel('累计外层决策步数');ax.set_ylabel('回合回报' if kind=='return' else '训练回合完成率 / %');axis(ax)
        if kind=='success':ax.set_ylim(0,100)
        ax.legend(title='训练种子',loc='best',frameon=False,ncol=3)
    for kind,title,filename in [('return','训练回报','fig_training_return'),('success','训练任务完成率','fig_training_success')]:
        fig,ax=plt.subplots(figsize=(12.8,7.2));panel(ax,kind);ax.set_title(title,pad=18);fig.subplots_adjust(left=.10,right=.97,bottom=.15,top=.88);save(fig,filename)
    fig,axs=plt.subplots(1,2,figsize=(16,9))
    for ax,kind,title in zip(axs,('return','success'),('（a）训练回报','（b）训练任务完成率')):panel(ax,kind);ax.set_title(title,pad=15)
    fig.subplots_adjust(left=.07,right=.98,bottom=.14,top=.89,wspace=.28);save(fig,'fig_training_combined')
def replay():
    guard()
    import torch
    torch.set_num_threads(1)
    from experiments.v3_stopping import env_for_stopping_run,load_stopping_model,run_stopping_episode
    from experiments.v3_handoff_scan import ImpulseMeter
    from env.task import compute_precapture_metrics
    results={};run=MAIN/'logs/final2_262460'
    for row in ('pure','stopping'):
        print('REPLAY_STARTED',row,flush=True)
        _,env=env_for_stopping_run(run,True);model=None if row=='pure' else load_stopping_model(run,'checkpoints/stopping_30000_outer_decisions.zip')
        meter=ImpulseMeter(env.controller,float(env.environment_config.dt_s));physical=[];ctx={'k':0,'branch':None,'handoff_time_s':None,'handoff_position':None}
        def record():
            inner=env.env;relative=inner.relative
            metrics=compute_precapture_metrics(inner.target_state,inner.chaser_state,relative,inner.config.precapture_task,
                terminal_region_active=inner._terminal_region_entered,staging_direction_inertial=inner._staging_direction_inertial)
            v=relative.rotation@relative.velocity
            physical.append({'time_s':float(inner.time_seconds),'x_m':float(relative.position[0]),'y_m':float(relative.position[1]),'z_m':float(relative.position[2]),
                'vx_m_s':float(v[0]),'vy_m_s':float(v[1]),'vz_m_s':float(v[2]),'relative_speed_m_s':float(metrics.target_frame_speed_m_s),
                'distance_m':float(np.linalg.norm(relative.position)),'position_error_m':metrics.position_error_m,'attitude_error_deg':float(np.degrees(metrics.attitude_error_rad)),
                'fov_margin_deg':float(np.degrees(metrics.fov_margin_rad)),'keepout_margin_m':metrics.keepout_margin_m,'branch':ctx['branch'],'decision_k':ctx['k']})
        original_reset=env.reset
        def reset(*args,**kwargs):
            out=original_reset(*args,**kwargs);ctx.update(k=0,branch='baseline' if row=='pure' else 'learned');record();return out
        env.reset=reset
        original_physical=env.env.step
        def physical_step(*args,**kwargs):
            out=original_physical(*args,**kwargs);record();return out
        env.env.step=physical_step
        original_outer=env.step_with_branch
        def outer_step(*args,**kwargs):
            branch=kwargs['branch']
            if row=='stopping' and branch=='baseline' and ctx['branch']=='learned':
                ctx['handoff_time_s']=float(env.env.time_seconds);ctx['handoff_position']=env.env.relative.position.tolist()
            ctx['branch']=branch;out=original_outer(*args,**kwargs);ctx['k']+=1;return out
        env.step_with_branch=outer_step
        try:
            result=run_stopping_episode(env,model,266019,row,meter)
            geometry={'desired_position':np.asarray(env.environment_config.precapture_task.desired_position).tolist(),
                      'port_position':np.asarray(env.environment_config.precapture_task.port_position).tolist(),
                      'keepout_radius_m':float(env.environment_config.precapture_task.keepout_radius_m)}
            result.update(model_sha256=None if row=='pure' else CP_SHA,code_commit=COMMIT,dt_s=env.environment_config.dt_s,
                handoff_time_s=ctx['handoff_time_s'],handoff_position=ctx['handoff_position'],geometry=geometry)
            assert np.allclose(np.diff([r['time_s'] for r in physical]),.1,rtol=0,atol=1e-9),'Physical sample interval not 0.1s'
            with (DATA/f'{row}_physical_266019.csv').open('w',newline='',encoding='utf-8-sig') as f:
                w=csv.DictWriter(f,fieldnames=list(physical[0]));w.writeheader();w.writerows(physical)
            write(DATA/f'{row}_result_266019.json',result);results[row]=result
            print('REPLAY_COMPLETED',row,result['survival_s'],result['clean_completion'],flush=True)
        finally:env.close()
    ref=read(REF);actual=results['stopping'];checks={}
    for key in ('seed','row','handoff_k','completed','zero_violation','clean_completion','decisions','learned_decisions','qp_zero_fallbacks','failure'):
        checks[key]=actual[key]==ref[key]
    for key in ('survival_s','equivalent_delta_v_m_s'):checks[key]=bool(np.isclose(actual[key],ref[key],rtol=0,atol=1e-9))
    checks['value_trace']=all(np.array_equal(np.asarray(actual['trace'][k]),np.asarray(ref['trace'][k])) for k in ('beta','q_handoff','q_continue'))
    a=np.genfromtxt(DATA/'pure_physical_266019.csv',delimiter=',',names=True,encoding='utf-8-sig',dtype=None)
    b=np.genfromtxt(DATA/'stopping_physical_266019.csv',delimiter=',',names=True,encoding='utf-8-sig',dtype=None)
    checks['same_initial_condition']=all(a[k][0]==b[k][0] for k in ('x_m','y_m','z_m','vx_m_s','vy_m_s','vz_m_s','attitude_error_deg'))
    write(DATA/'REPLAY_VERIFICATION.json',{'reference':str(REF),'reference_sha256':REF_SHA,'checkpoint_sha256':CP_SHA,'checks':checks,'passed':all(checks.values()),'actual':actual})
    assert all(checks.values()),'Replay differs from original: '+str(checks)
    guard();print('REPLAY_VERIFICATION_PASSED',flush=True)
def physical(row):return np.genfromtxt(DATA/f'{row}_physical_266019.csv',delimiter=',',names=True,encoding='utf-8-sig',dtype=None)
def final_plots():
    assert read(DATA/'REPLAY_VERIFICATION.json')['passed'];style();rows={r:physical(r) for r in ('pure','stopping')}
    result=read(DATA/'stopping_result_266019.json');handoff=np.asarray(result['handoff_position']);ht=result['handoff_time_s'];desired=np.asarray(result['geometry']['desired_position']);port=np.asarray(result['geometry']['port_position'])
    labels={'pure':'Pure MPC','stopping':'价值协调 SAC–MPC'}
    fig=plt.figure(figsize=(12.8,7.2));ax=fig.add_subplot(111,projection='3d')
    pts=[]
    for row,arr in rows.items():
        p=np.column_stack([arr[k] for k in ('x_m','y_m','z_m')]);pts.append(p)
        ax.plot(*p.T,color=METHOD_COLORS[row],lw=2.1,label=labels[row],ls='--' if row=='pure' else '-')
        ax.scatter(*p[-1],color=METHOD_COLORS[row],s=40,marker='o',depthshade=False)
    ax.scatter(*pts[0][0],marker='*',s=145,color='#444444',label='共同起点',depthshade=False)
    ax.scatter(*handoff,marker='D',s=65,color='#d98c00',label='实际交接点',depthshade=False)
    ax.scatter(0,0,0,marker='s',s=65,color='#555555',label='目标质心',depthshade=False)
    ax.scatter(*port,marker='^',s=50,color='#006d5b',label='捕获端口',depthshade=False)
    ax.scatter(*desired,marker='x',s=70,color='#006d5b',label='预捕获目标位置',depthshade=False)
    ax.text(*handoff,'  交接',color='#ad6b00',fontsize=13)
    allp=np.vstack([*pts,handoff,desired,port,np.zeros(3)]);lo=allp.min(0);hi=allp.max(0);center=(lo+hi)/2;rad=max(hi-lo)*.55
    for setlim,c in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),center):setlim(c-rad,c+rad)
    ax.set_box_aspect((1,1,1));ax.view_init(elev=24,azim=-57)
    for a in (ax.xaxis,ax.yaxis,ax.zaxis):a.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.set_xlabel('目标系 X / m',labelpad=12);ax.set_ylabel('目标系 Y / m',labelpad=12);ax.set_zlabel('目标系 Z / m',labelpad=9)
    ax.set_title('预捕获三维轨迹',pad=18)
    for a in (ax.xaxis,ax.yaxis,ax.zaxis):a.set_pane_color((1,1,1,0));a._axinfo['grid'].update(color='#e6e6e6',linewidth=.5)
    ax.legend(loc='upper left',bbox_to_anchor=(1.16,.93),frameon=False,fontsize=13);fig.subplots_adjust(left=.02,right=.76,bottom=.17,top=.90);save(fig,'fig_trajectory_3d')
    fig,axs=plt.subplots(3,1,figsize=(12.8,7.2),sharex=True)
    for ax,field,label in zip(axs,('distance_m','relative_speed_m_s','attitude_error_deg'),('相对距离 / m','相对速度 / (m/s)','姿态误差 / °')):
        for row,arr in rows.items():ax.plot(arr['time_s'],arr[field],color=METHOD_COLORS[row],lw=1.8,ls='--' if row=='pure' else '-',label=labels[row])
        ax.axvline(ht,color='#d98c00',ls=':',lw=1.4);ax.set_ylabel(label);axis(ax)
    axs[0].legend(frameon=False,ncol=2,loc='upper right');axs[-1].set_xlabel('时间 / s');axs[-1].set_xlim(0,max(a['time_s'][-1] for a in rows.values()))
    axs[0].set_title('预捕获状态响应',pad=15);axs[0].text(ht,.98,'交接',transform=axs[0].get_xaxis_transform(),color='#ad6b00',va='top',ha='left')
    fig.subplots_adjust(left=.14,right=.97,bottom=.12,top=.89,hspace=.18);save(fig,'fig_state_response')
    trace=result['trace'];gap=np.array(trace['q_handoff'])-np.array(trace['q_continue']);k=np.arange(len(gap))
    with (DATA/'handoff_value_266019.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['decision_k','Q_H','Q_C','Q_H_minus_Q_C','beta']);w.writerows(zip(k,trace['q_handoff'],trace['q_continue'],gap,trace['beta']))
    fig,ax=plt.subplots(figsize=(12.8,7.2));ax.plot(k,gap,color=COLORS[0],lw=2.4,marker='o',ms=4)
    ax.axhline(0,color='#555555',lw=1.2,ls='--',label='交接阈值');ax.axvline(30,color='#d98c00',lw=1.6,ls=':')
    ax.scatter(30,gap[30],s=90,color='#d98c00',zorder=5);ax.annotate('实际交接（k = 30）',xy=(30,gap[30]),xytext=(-160,32),textcoords='offset points',arrowprops={'arrowstyle':'->','color':'#ad6b00'},color='#ad6b00')
    ax.set_xlim(0,31);ax.set_xlabel('外层决策编号 k');ax.set_ylabel('交接价值差（Q_H − Q_C）');ax.set_title('价值差与交接决策',pad=18);axis(ax);ax.legend(frameon=False,loc='lower right')
    zoom=ax.inset_axes([.46,.23,.36,.32])
    zoom.plot(k[23:],gap[23:],color=COLORS[0],lw=1.7,marker='o',ms=4)
    zoom.axhline(0,color='#555555',ls='--',lw=1);zoom.axvline(30,color='#d98c00',ls=':',lw=1.3)
    zoom.scatter(30,gap[30],color='#d98c00',s=45,zorder=5)
    zoom.set_xlim(22.7,30.4);zoom.set_ylim(min(gap[23:])-.03,max(gap[23:])+.03)
    zoom.set_xticks([23,25,27,29,30]);zoom.set_title('交接阈值附近',fontsize=13,pad=5);zoom.tick_params(labelsize=11)
    zoom.text(.04,.88,f'k=29: {gap[29]:.3f}\nk=30: +{gap[30]:.3f}',transform=zoom.transAxes,fontsize=11,va='top',bbox=dict(facecolor='white',edgecolor='none',alpha=.95,pad=2))
    axis(zoom)
    fig.subplots_adjust(left=.11,right=.97,bottom=.15,top=.88);save(fig,'fig_handoff_value')
    report=f'''# 汇报图表来源与核验\n\n科学提交：{COMMIT}；工况w2.36_r15；模型262460固定30k检查点SHA256 {CP_SHA}。图面依用户要求不写阶段标识；所有图均为阶段训练与单案例演示，不是60k最终性能。\n\n训练三种子262460–262462先读取Monitor字节快照、去掉不完整末行，再累计episode的l得到外层决策步数，取结束步≤30000的回合。回报原始浅线+50完整回合尾随均值；完成率也是50完整回合窗口，前49回合不填充趋势，不补造端点。completed是训练任务完成，不代表零违规。三个种子全展示，不作已收敛结论。快照及绘制数据在plot_data。\n\n轨迹固定开局266019，Pure MPC与价值协调均使用原env_for_stopping_run、load_stopping_model、run_stopping_episode；低优先级、单线程，两回合，不筛选或开展完整评估。仅在重放进程的环境实例中给原step增加返回后只读采样，原控制逻辑及返回值保持原样。记录初态+每0.1s物理步后的状态。真实位置使用relative.position；目标系相对速度为relative.rotation @ relative.velocity，模长用compute_precapture_metrics.target_frame_speed_m_s，与任务定义一致；它不是简单将惯性速度旋转后忽略旋转项。姿态误差用该任务desired_transform的SO(3)测地误差，单位度。FOV及keepout裕度随CSV提供。\n\n原参考开发JSON SHA256 {REF_SHA}；全部逐项核验见REPLAY_VERIFICATION.json：k=30、完成、零违规、143.6s与Delta-v=1.8237301960865566，连续量绝对容差1e-9，价值trace逐位一致，初始位置/速度/姿态误差相同。图中交接时间取真实记录{ht:.1f}s。图上目标质心/捕获端口/预捕获目标位置来自配置，不虚构本体外形或安全走廊，不把预捕获完成说成物理接触。三维轴等比例，状态响应不平滑；两个方法各自自然终止，不补延长轨迹。\n\n价值图只有Q_H−Q_C判据及零线，k=30标实测交接；未重复展示beta。参考第一篇论文DRL/Test/plot_01与plot_02的字体、配色和线条样式，不使用其实验数据、场景筛选、best模型、旧安全阈值或硬编码汇总。\n\n优先400dpi PNG，另提供PDF/SVG。没有性能柱图或违规训练曲线。训练与科学文件未修改。纯MPC实际结果：\n```json\n{json.dumps(read(DATA/'pure_result_266019.json'),ensure_ascii=False,indent=2)}\n```\n'''
    (OUT/'FIGURE_REPORT.md').write_text(report,encoding='utf-8')
    import shutil
    shutil.copy2(Path(__file__),OUT/'make_figures.py')
    write(OUT/'FILES_SHA256.json',{p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='FILES_SHA256.json'})
if __name__=='__main__':
    mode=sys.argv[1] if len(sys.argv)>1 else 'all'
    if mode in ('curves','all'):curves()
    if mode in ('replay','all'):replay()
    if mode in ('finalize','all'):final_plots()
