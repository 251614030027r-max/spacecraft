from pathlib import Path
import subprocess,json,time,psutil
if __name__=='__main__':
 r=Path(__file__).resolve().parents[1];p=psutil.Process();assert not any('codex' in q.name().lower() or 'chatgpt' in q.name().lower() for q in p.parents());p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 with (r/'记录/LOGGING_GATE_NATIVE.lock').open('x') as f:f.write(str(p.pid))
 with (r/'记录/logging_gate.out.log').open('ab') as fo,(r/'记录/logging_gate.err.log').open('ab') as fe:
  cmd=['D:/py/DRL2/.venv/Scripts/python.exe','-B',str(r/'脚本/verify_logging.py')];child=subprocess.Popen(cmd,cwd='D:/py/DRL2',stdout=fo,stderr=fe,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
  (r/'记录/LOGGING_GATE_LAUNCH.json').write_text(json.dumps({'pid':p.pid,'child_pid':child.pid,'created':p.create_time(),'command':cmd,'start':time.time()},ensure_ascii=False,indent=2),encoding='utf-8');rc=child.wait();(r/'记录/LOGGING_GATE_EXIT.json').write_text(json.dumps({'returncode':rc,'end':time.time()}),encoding='utf-8')
