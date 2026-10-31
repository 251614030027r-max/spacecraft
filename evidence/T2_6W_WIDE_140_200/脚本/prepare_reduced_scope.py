from pathlib import Path
import json,hashlib,shutil,ast
R=Path(__file__).resolve().parents[1]
b=R/'记录/REDUCED_SCOPE_BACKUP';b.mkdir(exist_ok=False)
for n in ('parallel_eval.py','publish_t26w.py','run_t26w.py'):
 shutil.copy2(R/'脚本'/n,b/n)
p=(R/'脚本/parallel_eval.py').read_text(encoding='utf-8')
p=p.replace('MARKS=range(0,4001,500)','MARKS=(0,500,1000,1500,2000,2500,3000,4000)').replace('1728','1536')
(R/'脚本/parallel_eval_reduced.py').write_text(p,encoding='utf-8')
p=(R/'脚本/publish_t26w.py').read_text(encoding='utf-8')
p=p.replace('1728','1536').replace('x9marks0,500,...4000 x4=1536','x8marks0,500,1000,1500,2000,2500,3000,4000 x4=1536')
p=p.replace('late3000/3500/4000','late3000/4000;3500 cancelled by user;native late spread MISSING')
p=p.replace('末期3000/3500/4000极差用于给定预算的稳定性描述，不宣称数学收敛。','用户授权取消3500步192例，保留500–3000完整判定窗口及4000最终点；原生3500值和三点极差为MISSING，不填补、不宣称完整三点末期稳定性。')
p=p.replace("src+=cases+reference+", "src+=[R/'记录/REDUCED_SCOPE.json',R/'记录/REDUCED_HANDOFF.json']+cases+reference+")
p=p.replace("report=['# T2.6W", "report=['用户授权缩减：只取消3500步192例；500–3000原生判定样本及4000最终点完整保留。3500和三点末期极差MISSING；已有取消点产物另行保全。','# T2.6W")
ast.parse(p);(R/'脚本/publish_t26w.py').write_text(p,encoding='utf-8')
scope={'registered_marks':[0,500,1000,1500,2000,2500,3000,4000],'cancelled_marks':[3500],'expected_cases':1536,'openings':list(range(294000,294048)),'native_criteria':'t26w','gate_marks_unchanged':[500,1000,1500,2000,2500,3000],'final4000_retained':True,'user_authority':'按照你定的范围缩减，尽早完成然后推送汇报','no_training_restart':True,'late3500_and_threepoint_spread':'MISSING','method':'Existing workers keep required cases uninterrupted; retire only upon entering cancelled3500, then reduced pool fills4000.'}
(R/'记录/REDUCED_SCOPE.json').write_text(json.dumps(scope,ensure_ascii=False,indent=2),encoding='utf-8')
for f in (R/'脚本/parallel_eval_reduced.py',R/'脚本/publish_t26w.py',R/'脚本/run_t26w_reduced.py'):ast.parse(f.read_text(encoding='utf-8'))
print('Reduced scope prepared; native science and observers unchanged; 1536 unique cases.')
