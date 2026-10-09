"""Existing evidence only. Generic formal/baseline CSV and raw-directory plotting APIs."""
import os,sys,csv,json,hashlib,shutil,importlib.util,argparse,ctypes
from pathlib import Path
TOP=Path(__file__).resolve().parents[1];OUT=TOP/'reports/figures_interim_20261009';DATA=OUT/'plot_data';RAW=OUT/'inputs';REC=TOP/'记录'
for p in (OUT,DATA,RAW,REC):p.mkdir(parents=True,exist_ok=True)
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR=str(REC/'matplotlib'))
if os.name=='nt':
    k=ctypes.WinDLL('kernel32');k.GetCurrentProcess.restype=ctypes.c_void_p;k.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_uint32];assert k.SetPriorityClass(k.GetCurrentProcess(),0x4000)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker,font_manager
MAIN=Path('D:/py/DRL2');COLLAB=Path('C:/Users/35884/Documents/Spacecraft/过程文件/协作/Git工作树');CASE=COLLAB/'docs/collaboration/lower/handoffs/ppt_temporary_20261009'
COLORS=['#004b87','#b22222','#006d5b'];PURE='#777777';NOMINAL='#d18424'
plt.rcParams.update({'font.family':['Times New Roman','SimSun'],'mathtext.fontset':'stix','axes.unicode_minus':False,'font.size':14,'axes.labelsize':16,'axes.titlesize':18,'xtick.labelsize':12,'ytick.labelsize':12,'axes.linewidth':1,'xtick.direction':'in','ytick.direction':'in','pdf.fonttype':42,'svg.fonttype':'none'})
font_manager.findfont('SimSun',fallback_to_default=False)
sources={};checks=[];captions={}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def inp(p,expected=None):
    p=Path(p);digest=sha(p)
    if expected:assert digest==expected,f'SHA mismatch {p}'
    dest=RAW/(digest[:16]+'_'+p.name)
    if not dest.exists():shutil.copy2(p,dest)
    assert sha(dest)==digest;sources[str(p)]={'sha256':digest,'snapshot':dest.relative_to(OUT).as_posix()};return dest
def check(name,actual,expected,tol=None):
    ok=bool(np.isclose(actual,expected,atol=tol,rtol=0)) if tol is not None else actual==expected
    checks.append({'check':name,'actual':actual,'expected':expected,'passed':ok});assert ok,(name,actual,expected)
def csvout(name,rows):
    with (DATA/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def axis(ax):ax.tick_params(top=True,right=True);ax.grid(color='#e6e6e6',ls='--',lw=.6,alpha=.6);ax.set_axisbelow(True)
def save(fig,name,caption):
    fig.text(.5,.025,caption,ha='center',va='bottom',fontsize=10)
    for ext in ('png','pdf','svg'):fig.savefig(OUT/(name+'.'+ext),dpi=400,facecolor='white')
    plt.close(fig);captions[name]=caption;print('SAVED',name,flush=True)
def load_block(root,block,models,rows=('pure','learned','stopping')):
    records=[]
    for row in rows:
        for model in ([None] if row in ('pure','nominal') else models):
            path=Path(root)/row
            if model is not None:path=path/str(model)
            for seed in range(block,block+48):
                r=read(inp(path/f'seed_{seed}.json'));assert r['seed']==seed and r['row']==row and not r['code_dirty']
                records.append({'row':row,'model':model or '', 'seed':seed,'clean':r['clean_completion'],'violation':not r['zero_violation'],'time_s':r['survival_s'],'delta_v_m_s':r['equivalent_delta_v_m_s'],'failure':'|'.join(r['failure']),'code_commit':r['code_commit'],'model_sha256':r.get('model_sha256') or ''})
    return records
def comparison(records,models,name,caption):
    groups=[('pure','')]+[(row,s) for row in ('learned','stopping') for s in models]
    sets=[[r for r in records if r['row']==row and str(r['model'])==str(seed)] for row,seed in groups]
    assert all(len(s)==48 for s in sets)
    colors=[PURE]+[COLORS[i] for i in range(len(models))]*2
    labels=['Pure']+[f'单飞\n{s}' for s in models]+[f'交接\n{s}' for s in models]
    fig,axs=plt.subplots(1,3,figsize=(19.2,9));x=np.arange(7)
    for i,(rs,c) in enumerate(zip(sets,colors)):
        clean=sum(r['clean'] for r in rs);violation=sum(r['violation'] for r in rs)
        axs[0].bar(i,clean,color=c,alpha=.45 if 1<=i<=3 else 1,width=.7)
        axs[0].text(i,clean+1,str(clean),ha='center',fontsize=13)
        axs[0].text(i,46,f'违规{violation}',ha='center',fontsize=9,color='#555555')
    axs[0].set_ylim(0,50);axs[0].set_ylabel('干净完成数 / 48');axs[0].set_title('（a）任务完成数',pad=18)
    rng=np.random.default_rng(20261009)
    for ax,field,label,title in zip(axs[1:],('delta_v_m_s','time_s'),('等效 Δv / (m/s)','完成时间 / s'),('（b）各行干净完成样本的 Δv','（c）各行干净完成样本的时间')):
        values=[[r[field] for r in rs if r['clean']] for rs in sets]
        boxes=ax.boxplot(values,positions=x,widths=.5,patch_artist=True,showfliers=False,medianprops={'color':'#222222','linewidth':1.2})
        for i,(patch,c,v) in enumerate(zip(boxes['boxes'],colors,values)):
            patch.set_facecolor(c);patch.set_alpha(.25);ax.scatter(i+rng.uniform(-.12,.12,len(v)),v,s=13,color=c,alpha=.55,edgecolors='none')
        ax.set_ylabel(label);ax.set_title(title,pad=18)
    for ax in axs:ax.set_xticks(x,labels,fontsize=10);axis(ax)
    fig.subplots_adjust(left=.055,right=.99,top=.87,bottom=.20,wspace=.32);save(fig,name,caption)
    paired=[]
    pure={r['seed']:r for r in sets[0]}
    for row,s in groups[1:]:
        for r in records:
            if r['row']==row and str(r['model'])==str(s) and r['clean'] and pure[r['seed']]['clean']:
                p=pure[r['seed']];paired.append({'model':s,'row':row,'seed':r['seed'],'candidate_time_s':r['time_s'],'pure_time_s':p['time_s'],'time_change_s':r['time_s']-p['time_s'],'candidate_delta_v_m_s':r['delta_v_m_s'],'pure_delta_v_m_s':p['delta_v_m_s'],'delta_v_change_m_s':r['delta_v_m_s']-p['delta_v_m_s']})
    if paired:csvout(name+'_paired_common_success.csv',paired)
    same_policy=[]
    for model in models:
        learned={r['seed']:r for r in records if r['row']=='learned' and str(r['model'])==str(model)}
        for r in records:
            if r['row']=='stopping' and str(r['model'])==str(model) and r['clean'] and learned[r['seed']]['clean']:
                l=learned[r['seed']];same_policy.append({'model':model,'seed':r['seed'],'stopping_time_s':r['time_s'],'learned_time_s':l['time_s'],'time_change_s':r['time_s']-l['time_s'],'stopping_delta_v_m_s':r['delta_v_m_s'],'learned_delta_v_m_s':l['delta_v_m_s'],'delta_v_change_m_s':r['delta_v_m_s']-l['delta_v_m_s']})
    if same_policy:csvout(name+'_stopping_vs_learned_paired.csv',same_policy)
def baseline_scatter(records,name,caption):
    blocks=sorted(set(r['seed']//1000*1000 for r in records));fig,axs=plt.subplots(1,len(blocks),figsize=(12.8*len(blocks),7.2),squeeze=False)
    for ax,block in zip(axs[0],blocks):
        for row,color in [('pure',PURE),('nominal',NOMINAL)]:
            selected=[r for r in records if r['row']==row and block<=r['seed']<block+48]
            assert len(selected)==48
            for success in (True,False):
                rs=[r for r in selected if bool(r['clean'])==success]
                ax.scatter([r['time_s'] for r in rs],[r['delta_v_m_s'] for r in rs],color=color,s=45,marker='o' if success else 'x',alpha=.8,label=f'{"Pure MPC" if row=="pure" else "nominal"}（{"完成" if success else "失败"}）')
        ax.set_xlabel('结束时间 / s');ax.set_ylabel('等效 Δv / (m/s)');ax.set_title(f'工况 w2.36_r15 · 块 {block}',pad=18);axis(ax);ax.legend(frameon=False,fontsize=12)
    fig.subplots_adjust(left=.09,right=.97,top=.86,bottom=.18,wspace=.25);save(fig,name,caption)
def training():
    audit=read(inp(Path('D:/py/DRL2_v3e/eval/v3e/final_training_audit_60k.json')))
    expected=['41c3aa05c005687045a174142a7e38a2c89d8d8b15a59f58f2d84384d77bfe29','920e721d63aa742ba8fd7103bd830b0e14fbe9283ec071a02a559bf078068325','9f1df32eaf46c42723ca67e01a32f6049a3fe450c0aafe48fed6cc119baaf2f6']
    plotted=[];curves=[]
    for seed,digest,run in zip((262420,262421,262422),expected,audit['runs']):
        p=inp(Path(f'D:/py/DRL2_v3e/logs/v3e_{seed}/train.monitor.csv'),digest)
        with p.open(encoding='utf-8-sig',newline='') as f:f.readline();rows=list(csv.DictReader(f))
        length=np.array([int(r['l']) for r in rows]);start=np.r_[0,np.cumsum(length)[:-1]]
        for j,ref in enumerate(run['bins_by_episode_start']):
            rs=[r for r,k in zip(rows,start) if j*10000<=k<(j+1)*10000]
            check(f'fig1 {seed} 10k bin{j} episodes',len(rs),ref['episodes']);check(f'fig1 {seed} 10k bin{j} completed',sum(r['completed']=='True' for r in rs),ref['completed'])
            check(f'fig1 {seed} 10k bin{j} completion rate',float(np.mean([r['completed']=='True' for r in rs])),ref['completion_rate'],1e-12)
        vals=[]
        for lo in range(0,60000,5000):
            rs=[r for r,k in zip(rows,start) if lo<=k<lo+5000];assert rs
            v={'model':seed,'bin_start':lo,'bin_end':lo+5000,'bin_center':lo+2500,'episodes':len(rs),'completed':sum(r['completed']=='True' for r in rs),'completion_percent':100*np.mean([r['completed']=='True' for r in rs]),'mean_return':np.mean([float(r['r']) for r in rs])};vals.append(v);plotted.append(v)
        curves.append(vals)
    csvout('fig1_v3e_training.csv',plotted);fig,axs=plt.subplots(1,2,figsize=(16,9));x=np.arange(2500,60000,5000)
    for ax,key,label,title in zip(axs,('completion_percent','mean_return'),('训练回合完成率 / %','平均回合回报'),('（a）学习层训练完成率','（b）学习层训练回报')):
        ys=np.array([[v[key] for v in curve] for curve in curves])
        for seed,color,y in zip((262420,262421,262422),COLORS,ys):ax.plot(x,y,color=color,lw=1.2,label=str(seed))
        ax.fill_between(x,ys.min(0),ys.max(0),color=COLORS[0],alpha=.12,label='三种子范围');ax.plot(x,ys.mean(0),color='#222222',lw=2.7,label='三种子均值')
        ax.set_xlim(0,60000);ax.set_xticks(np.arange(0,60001,10000));ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v,p:f'{int(v/1000)}k' if v else '0'));ax.set_xlabel('累计外层决策步数');ax.set_ylabel(label);ax.set_title(title,pad=18);axis(ax);ax.legend(frameon=False,fontsize=11,ncol=2)
        if key=='completion_percent':ax.set_ylim(0,100)
    fig.subplots_adjust(left=.07,right=.98,top=.88,bottom=.18,wspace=.28)
    save(fig,'fig1_v3e_training','V3e 262420–262422，历史60k训练、原工况、无交接；5k箱按回合起点统计。\n训练行为策略统计，非确定性部署性能，亦非本轮价值停止方法的训练曲线。')
def screen():
    rs=load_block(MAIN/'eval/regime_screen/w2.36_r15',269000,[],('pure','nominal'))
    for row,num,time in [('pure',34,98.7),('nominal',45,196.3)]:
        subset=[r for r in rs if r['row']==row];check('fig3 '+row+' clean',sum(r['clean'] for r in subset),num);check('fig3 '+row+' median time',float(np.median([r['time_s'] for r in subset if r['clean']])),time,1e-9)
    failures=[r for r in rs if r['row']=='pure' and not r['clean']];check('fig3 Pure failures',len(failures),14);check('fig3 Pure all failures timeout',all(r['failure']=='time_failure' for r in failures),True)
    csvout('fig3_baselines.csv',rs);baseline_scatter(rs,'fig3_baseline_time_energy','269000–269047，w2.36_r15，Pure / nominal，冻结工况筛选基线；×为失败时刻。\nPure完成34/48，中位98.7s；nominal完成45/48，中位196.3s；不含学习方法，非本轮正式性能。')
def hindsight():
    r=read(inp(MAIN/'eval/v3e/stage_b/readout_b1b2.json','2b84e14a19aec4d319862a781fbfa1f015fe69fcd698eb4471da5154cf5cbcd0'));check('fig4 fidelity problems',r['fidelity_problems'],[])
    pure=read(inp(MAIN/'eval/v3e/pure_mpc.json'));check('fig4 Pure records',len(pure['records']),48);check('fig4 Pure seed block',sorted(r['seed'] for r in pure['records']),list(range(262000,262048)));check('fig4 Pure complete',sum(r['truth_geometry_zero_violation_completed'] for r in pure['records']),37)
    data=[];counts=[]
    for seed,n in zip((262420,262421,262422),(42,33,38)):
        s=str(seed);b2=r['b2'][s];check('fig4 learned '+s,48-r['gates'][s]['failures'],n);check('fig4 B2 successes '+s,b2['summary']['successes'],n);check('fig4 all failed rescusable '+s,r['gates'][s]['rescuable'],48-n);check('fig4 oracle '+s,n+r['gates'][s]['rescuable'],48)
        counts.append(n)
        for e in b2['trajectories']:data.append({'model':seed,'seed':e['seed'],'time_saving_s':e['best_time_saving_s'],'delta_v_change_m_s':e['delta_v_change_at_best_m_s']})
        check('fig4 '+s+' median saving',float(np.median([e['best_time_saving_s'] for e in b2['trajectories']])),b2['summary']['median_best_time_saving_s'],1e-9)
    subset=[e for e in data if e['model']==262420];check('fig4 262420 median saving stated',float(np.median([e['time_saving_s'] for e in subset])),59.7,1e-9);check('fig4 262420 delta-v rounded',round(float(np.median([e['delta_v_change_m_s'] for e in subset])),2),-.38)
    csvout('fig4_hindsight_scatter.csv',data);csvout('fig4_hindsight_counts.csv',[{'method':'Pure','model':'','clean':37}]+[{'method':row,'model':s,'clean':n if row=='learned' else 48} for row in ('learned','hindsight') for s,n in zip((262420,262421,262422),counts)])
    fig,axs=plt.subplots(1,2,figsize=(16,9));bar=[37,*counts,48,48,48];colors=[PURE,*COLORS,*COLORS]
    for i,(n,c) in enumerate(zip(bar,colors)):axs[0].bar(i,n,color=c,alpha=.45 if 1<=i<=3 else 1,width=.7);axs[0].text(i,n+1,str(n),ha='center')
    axs[0].set_xticks(range(7),['Pure']+[f'单飞\n{s}' for s in (262420,262421,262422)]+[f'事后最优\n{s}' for s in (262420,262421,262422)],fontsize=10);axs[0].set_ylim(0,52);axs[0].set_ylabel('干净完成数 / 48');axs[0].set_title('（a）事后交接机会',pad=18);axis(axs[0])
    for seed,c in zip((262420,262421,262422),COLORS):
        rs=[e for e in data if e['model']==seed];x=[e['time_saving_s'] for e in rs];y=[e['delta_v_change_m_s'] for e in rs];axs[1].scatter(x,y,color=c,s=24,alpha=.55,label=str(seed));axs[1].scatter(np.median(x),np.median(y),color=c,edgecolors='#222222',marker='D',s=85,label=f'{seed}中位数')
    axs[1].axhline(0,color='#555555',ls='--',lw=1);axs[1].axvline(0,color='#555555',ls='--',lw=1);axs[1].set_xlabel('较原学习策略节省时间 / s');axs[1].set_ylabel('Δv变化 / (m/s)');axs[1].set_title('（b）学习成功开局的事后最优交接',pad=18);axs[1].legend(frameon=False,fontsize=10,ncol=2);axis(axs[1]);fig.subplots_adjust(left=.07,right=.98,top=.87,bottom=.20,wspace=.28)
    save(fig,'fig4_hindsight_timing','262000–262047，原工况，冻结模型262420–262422；B1/B2事后扫描，非可部署方法性能。\n48/48是事后机会的上界；右图仅学习成功开局，Δv变化对应最省时交接；分量中位数不代表同一开局。')
def cases():
    manifest=read(inp(MAIN/'logs/final2_262460/manifest.json'));task=manifest['training_environment']['precapture_task'];check('fig5 position threshold',task['completion_position_m'],.25,1e-12);check('fig5 speed threshold',task['completion_speed_m_s'],.05,1e-12);check('fig5 attitude threshold deg',float(np.degrees(task['completion_attitude_rad'])),10.,1e-12)
    hash_manifest=read(CASE/'FILES_SHA256.json');rows={};results={}
    for key in ('pure',262460,262462):
        prefix='pure' if key=='pure' else f'stopping_{key}';csvname=f'plot_data/{prefix}_physical_266019.csv';rname=f'plot_data/{prefix}_result_266019.json'
        p=inp(CASE/csvname,hash_manifest[csvname]);r=read(inp(CASE/rname,hash_manifest[rname]));rows[key]=np.genfromtxt(p,delimiter=',',names=True,encoding='utf-8-sig',dtype=None);results[key]=r
        check(f'fig5 {key} time',r['survival_s'],{'pure':106.8,262460:133.9,262462:161.1}[key],1e-9);check(f'fig5 {key} clean',r['clean_completion'],True)
        shutil.copy2(p,DATA/f'fig5_{prefix}_physical.csv')
    colors={'pure':PURE,262460:COLORS[0],262462:COLORS[2]};labels={'pure':'Pure MPC',262460:'价值协调（262460）',262462:'价值协调（262462）'}
    cap='50k中间模型262460 / 262462，w2.36_r15，开发块单个开局266019，非正式。\n本例方法完成但比Pure慢：133.9 / 161.1s 对106.8s；不代表总体性能或协调增益。'
    # Redraw the same frozen coordinates/value traces; only labels/captions change.
    fig=plt.figure(figsize=(12.8,8));ax=fig.add_subplot(111,projection='3d');points=[]
    for key,a in rows.items():
        p=np.column_stack([a[k] for k in ('x_m','y_m','z_m')]);points.append(p);ax.plot(*p.T,color=colors[key],lw=2,ls='--' if key=='pure' else '-',label=labels[key]);ax.scatter(*p[-1],color=colors[key],s=35,depthshade=False)
        if key!='pure':ax.scatter(*results[key]['handoff_position'],marker='D',s=60,color=colors[key],depthshade=False,label=f'交接点（{key}）')
    desired=np.asarray(results[262460]['geometry']['desired_position']);port=np.asarray(results[262460]['geometry']['port_position'])
    ax.scatter(*points[0][0],marker='*',s=140,color='#333333',label='共同起点',depthshade=False);ax.scatter(0,0,0,marker='s',s=50,color='#555555',label='目标质心',depthshade=False);ax.scatter(*port,marker='^',s=55,color='#ad6b00',label='捕获端口',depthshade=False);ax.scatter(*desired,marker='x',s=65,color='#ad6b00',label='预捕获目标位置',depthshade=False)
    p=np.vstack([*points,desired,port,np.zeros(3)]);center=(p.min(0)+p.max(0))/2;rad=np.max(p.max(0)-p.min(0))*.55
    for f,c in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),center):f(c-rad,c+rad)
    ax.set_box_aspect((1,1,1));ax.view_init(elev=24,azim=-57)
    for a in (ax.xaxis,ax.yaxis,ax.zaxis):a.set_major_locator(ticker.MaxNLocator(nbins=5));a.set_pane_color((1,1,1,0));a._axinfo['grid'].update(color='#e6e6e6',linewidth=.5)
    ax.set_xlabel('目标系 X / m',labelpad=12);ax.set_ylabel('目标系 Y / m',labelpad=12);ax.set_zlabel('目标系 Z / m',labelpad=9);ax.set_title('预捕获三维轨迹',pad=18);ax.legend(loc='upper left',bbox_to_anchor=(1.16,.96),frameon=False,fontsize=12);fig.subplots_adjust(left=.02,right=.76,bottom=.22,top=.90);save(fig,'fig_trajectory_3d',cap)
    fig,axs=plt.subplots(1,2,figsize=(16,9))
    for ax,seed in zip(axs,(262460,262462)):
        r=results[seed];gap=np.array(r['trace']['q_handoff'])-np.array(r['trace']['q_continue']);hk=r['handoff_k'];k=np.arange(len(gap))
        ax.plot(k,gap,color=colors[seed],lw=2.2,marker='o',ms=3);ax.axhline(0,color='#555555',ls='--',lw=1);ax.scatter(hk,gap[hk],color='#d98c00',s=75,zorder=5);ax.axvline(hk,color='#d98c00',ls=':',lw=1.2);ax.annotate(f'交接 k={hk}',xy=(hk,gap[hk]),xytext=(-100,25),textcoords='offset points',color='#ad6b00',arrowprops={'arrowstyle':'->','color':'#ad6b00'})
        ax.set_xlabel('外层决策编号 k');ax.set_ylabel('交接价值差（Q_H − Q_C）');ax.set_title(f'模型 {seed}',pad=20);axis(ax)
    fig.subplots_adjust(left=.08,right=.97,bottom=.18,top=.88,wspace=.27);save(fig,'fig_handoff_value',cap)
    fig,axs=plt.subplots(3,1,figsize=(12.8,8.1),sharex=True)
    for ax,field,label,threshold in zip(axs,('position_error_m','relative_speed_m_s','attitude_error_deg'),('位置误差 / m','相对速度 / (m/s)','姿态误差 / °'),(.25,.05,10)):
        for key,a in rows.items():ax.plot(a['time_s'],a[field],color=colors[key],lw=1.8,ls='--' if key=='pure' else '-',label=labels[key])
        for seed in (262460,262462):ax.axvline(results[seed]['handoff_time_s'],color=colors[seed],ls=':',lw=1.2)
        ax.axhline(threshold,color='#ad6b00',ls='--',lw=1.1);ax.text(.99,threshold,f' 捕获阈值 {threshold:g}',transform=ax.get_yaxis_transform(),ha='right',va='bottom',fontsize=10,color='#ad6b00');ax.set_ylabel(label);axis(ax)
    for seed in (262460,262462):axs[0].text(results[seed]['handoff_time_s']+1,.08,f'交接（{seed}）',transform=axs[0].get_xaxis_transform(),color=colors[seed],fontsize=10,rotation=90,va='bottom')
    handles,labels_=axs[0].get_legend_handles_labels();fig.legend(handles,labels_,ncol=3,frameon=False,loc='upper center',bbox_to_anchor=(.56,.945),fontsize=12);fig.suptitle('预捕获状态与捕获条件',y=.985,fontsize=18);axs[-1].set_xlabel('时间 / s');axs[-1].set_xlim(0,161.1);fig.subplots_adjust(left=.14,right=.97,top=.85,bottom=.17,hspace=.20);save(fig,'fig_state_response',cap)
    value=[]
    for seed in (262460,262462):
        r=results[seed];check(f'fig5 {seed} handoff k',r['handoff_k'],21 if seed==262460 else 39)
        for k,(qh,qc) in enumerate(zip(r['trace']['q_handoff'],r['trace']['q_continue'])):value.append({'model':seed,'decision_k':k,'Q_H':qh,'Q_C':qc,'gap':qh-qc,'actual_handoff':k==r['handoff_k']})
    csvout('fig5_value.csv',value)
    csvout('fig5_geometry.csv',[{'model':s,'handoff_time_s':results[s]['handoff_time_s'],'handoff_k':results[s]['handoff_k'],'desired_position':json.dumps(results[s]['geometry']['desired_position']),'port_position':json.dumps(results[s]['geometry']['port_position']),'position_threshold_m':.25,'speed_threshold_m_s':.05,'attitude_threshold_deg':10} for s in (262460,262462)])
def official(allow_corrected=False):
    rs=load_block(MAIN/'eval/stopping/formal',267000,[262430,262431,262432]);csvout('fig2_formal_rows.csv',rs)
    check('fig2 science commit',sorted(set(r['code_commit'] for r in rs)),['2e5c236f7412caea3b203519619ef7a48bb16df9'])
    for seed in (262430,262431,262432):check(f'fig2 {seed} learned/stopping same frozen SHA',len(set(r['model_sha256'] for r in rs if r['model']==seed)),1)
    for row,ns in [('learned',[12,31,17]),('stopping',[37,41,36])]:
        for model,n in zip((262430,262431,262432),ns):check(f'fig2 {row} {model} clean',sum(r['clean'] for r in rs if r['row']==row and r['model']==model),n)
    check('fig2 Pure clean',sum(r['clean'] for r in rs if r['row']=='pure'),41);check('fig2 stopping violations',sum(r['violation'] for r in rs if r['row']=='stopping'),0)
    for model,v in zip((262430,262431,262432),(1,0,1)):check(f'fig2 learned {model} actual violations',sum(r['violation'] for r in rs if r['row']=='learned' and r['model']==model),v)
    if not allow_corrected:
        (OUT/'FIG2_BLOCKED.md').write_text('图2完成数全部符合执行单，但“违规0”若指所有行则不符合：learned=1/0/1，Pure与stopping=0。遵照数字不符停止报告规则，已准备只读数据，未生成图2。等待用户明确按实际逐行标注。\n',encoding='utf-8');return
    comparison(rs,[262430,262431,262432],'fig2_formal_comparison','267000–267047，原工况历史正式轮，模型262430–262432；METHOD_DOES_NOT_HOLD。\n交接较同策略单飞多完成+25/+10/+19；stopping违规0，learned违规1/0/1；箱线图各取自身干净成功样本。')
def seal():
    (OUT/'INPUTS_SHA256.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf-8');(OUT/'CHECKS.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'CHECKS.md').write_text('# 输入数字核对\n\n'+''.join(f'- {r["check"]}：实际 `{r["actual"]}`；预期 `{r["expected"]}`；通过 `{r["passed"]}`。\n' for r in checks)+'\n图2learned实际违规1/0/1，stopping0；用户已明确授权“按真实1／0／1标注，继续图2”。案例距离阈值使用真实位置误差，不以质心距离代替；第一栏相应改标。271000/272000完整基线尚不存在，不补仿真。\n',encoding='utf-8')
    (OUT/'FIGURE_CAPTIONS.json').write_text(json.dumps(captions,ensure_ascii=False,indent=2),encoding='utf-8');shutil.copy2(Path(__file__),OUT/Path(__file__).name)
    files={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='FILES_SHA256.json'};(OUT/'FILES_SHA256.json').write_text(json.dumps(files,ensure_ascii=False,indent=2),encoding='utf-8');assert all(sha(OUT/n)==v for n,v in files.items());print('SEALED',len(files),'files',len(checks),'checks',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--allow-corrected-violations',action='store_true');p.add_argument('--generic-root',type=Path);p.add_argument('--block',type=int);p.add_argument('--models',type=int,nargs='*');p.add_argument('--baseline-only',action='store_true');p.add_argument('--caption',default='正式结果；样本块与工况以输入记录为准。');args=p.parse_args()
    if args.generic_root:
        records=load_block(args.generic_root,args.block,args.models or [],('pure','nominal') if args.baseline_only else ('pure','learned','stopping'));csvout('generic_rows.csv',records)
        if args.baseline_only:baseline_scatter(records,'generic_baselines',args.caption)
        else:comparison(records,args.models,'generic_comparison',args.caption)
    else:training();screen();hindsight();cases();official(args.allow_corrected_violations)
    seal()
