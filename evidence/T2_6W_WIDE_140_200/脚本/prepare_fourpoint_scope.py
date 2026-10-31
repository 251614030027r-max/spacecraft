from pathlib import Path
import shutil,ast,json
R=Path('过程文件/T2_6W_WIDE_140_200');b=R/'记录/FOURPOINT_BACKUP';b.mkdir(exist_ok=False)
for n in ('run_t26w_reduced.py','parallel_eval_reduced.py','publish_t26w.py'):shutil.copy2(R/'脚本'/n,b/n)
p=(R/'脚本/parallel_eval_reduced.py').read_text(encoding='utf-8').replace('MARKS=(0,500,1000,1500,2000,2500,3000,4000)','MARKS=(1000,2000,3000,4000)').replace('1536','768')
(R/'脚本/parallel_eval_fourpoint.py').write_text(p,encoding='utf-8')
p=(R/'脚本/run_t26w_reduced.py').read_text(encoding='utf-8').replace('REDUCED_CONTROLLER','FOURPOINT_CONTROLLER').replace("assert old['phase']=='C_evaluation'","assert old['phase'] in ('C_evaluation','C_evaluation_scope_drain')").replace("'run_t26w.py' in","'run_t26w_reduced.py' in").replace('REDUCED_HANDOFF','FOURPOINT_HANDOFF').replace('User reduced1536:8marks excluding3500; native t26w gate unchanged; final4000 retained','User fourpoint768:1000/2000/3000/4000; full native gate not applicable').replace('expected_eval_cases=1536','expected_eval_cases=768')
p=p.replace("any('_step_3500_' in s.get('case','') for s in states)","any(s.get('case') and int(s['case'].split('_step_')[1].split('_')[0]) not in (1000,2000,3000,4000) for s in states)").replace('REDUCED_DRAIN_STATE','FOURPOINT_DRAIN_STATE').replace('retired_only_cancelled3500','retired_only_cancelled_points')
start=p.index(' archived=[]');end=p.index(' import parallel_eval_reduced',start)
p=p[:start]+''' archived=[]
 for arm in ('e_sac','e_mve'):
  for seed in (0,1):
   for k in (0,500,1500,2500,3500):
    src=R/f'C_eval/{arm}_s{seed}/step_{k}'
    if src.exists():
     dest=R/f'记录/fourpoint_cancelled/{arm}_s{seed}/step_{k}';assert not dest.exists() and dest.resolve().is_relative_to(R.resolve());dest.parent.mkdir(parents=True,exist_ok=True)
     hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in src.glob('*') if f.is_file()};shutil.move(str(src),str(dest));assert hashes=={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in dest.glob('*') if f.is_file()};archived.append({'path':str(dest),'hashes':hashes})
 a.dump(R/'记录/FOURPOINT_CANCELLED_PRESERVED.json',archived)
''' +p[end:]
p=p.replace('parallel_eval_reduced','parallel_eval_fourpoint').replace('==1536','==768').replace(";assert a.read(R/'DECISION.json')['verdict']!='incomplete'",'')
pos=p.index(" a.progress(phase='final_publication'")
p=p[:pos]+''' native=a.read(R/'DECISION.json');a.dump(R/'DECISION_NATIVE.json',native)
 native.update(verdict='scope_revised_descriptive_only',native_verdict=native.get('verdict'),scope_revision={'marks':[1000,2000,3000,4000],'full_t26w_window':'MISSING; cancelled by user','no_full_pass_claim':True})
 a.dump(R/'DECISION.json',native)
'''+p[pos:]
(R/'脚本/run_t26w_fourpoint.py').write_text(p,encoding='utf-8')
p=(R/'脚本/publish_t26w.py').read_text(encoding='utf-8').replace('1536','768').replace('48 openings294000-294047 x8marks0,500,1000,1500,2000,2500,3000,4000 x4=768','48 openings294000-294047 x4marks1000,2000,3000,4000 x4=768')
p=p.replace("R/'记录/REDUCED_SCOPE.json',R/'记录/REDUCED_HANDOFF.json'","R/'记录/FOURPOINT_SCOPE.json',R/'记录/FOURPOINT_HANDOFF.json',R/'DECISION_NATIVE.json'")
p=p.replace('用户授权缩减：只取消3500步192例；500–3000原生判定样本及4000最终点完整保留。3500和三点末期极差MISSING；已有取消点产物另行保全。','用户最新授权仅评估1000/2000/3000/4000步共768例。取消点已有结果SHA保全，不补算；原完整t26w窗口不再适用，不宣称完整pass。')
p=p.replace('用户授权取消3500步192例，保留500–3000完整判定窗口及4000最终点；原生3500值和三点极差为MISSING，不填补、不宣称完整三点末期稳定性。','用户取消0/500/1500/2500/3500评估；完整窗口及3500读数MISSING，不填补。')
p=p.replace('判定窗口为500–3000；4000完成不等于收敛，不用末期最佳点替换预登记判据。原生pass仅本轮开发门通过，不能替代论文级证据。','原完整500–3000判据因用户缩减不再适用；保留原生命令输出DECISION_NATIVE和机制读数，最终标scope_revised_descriptive_only，不重新定义门槛。')
p=p.replace('native t26w;500-3000 gate only; late3000/4000;3500 cancelled by user;native late spread MISSING and pressure bins reported separately','scope_revised_descriptive_only; original native gate incomplete; four marks only').replace('## 原生t26w判定（500–3000）','## 缩减范围结果（原生完整判定不适用）')
(R/'脚本/publish_t26w.py').write_text(p,encoding='utf-8')
scope={'registered_marks':[1000,2000,3000,4000],'cancelled_marks':[0,500,1500,2500,3500],'expected_cases':768,'user_authority':'保留1000、2000、3000、4000步即可，重新调整','native_gate':'scope_revised_descriptive_only','training_finished_unchanged':True}
(R/'记录/FOURPOINT_SCOPE.json').write_text(json.dumps(scope,ensure_ascii=False,indent=2),encoding='utf-8')
for n in ('run_t26w_fourpoint.py','parallel_eval_fourpoint.py','publish_t26w.py'):ast.parse((R/'脚本'/n).read_text(encoding='utf-8'))
print('Fourpoint768 prepared, syntax valid.')
