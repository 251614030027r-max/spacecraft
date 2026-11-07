"""T2.5.1 tests and byte-identical historical controls/audits; no training here."""
from pathlib import Path
import os,json,time,subprocess,hashlib,shutil,traceback,re
import psutil
R=Path(__file__).resolve().parents[1];REPO=Path('D:/py/DRL2');PY=str(REPO/'.venv/Scripts/python.exe');OLD=R.parent/'T2_5_VALIDATE_20261026';SHA='34f877e933dc78740674fc82ee0e36f918c541f4'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ.update(GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='safe.directory',GIT_CONFIG_VALUE_0=str(REPO))
os.chdir(REPO)
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 me=psutil.Process();anc=[{'pid':p.pid,'name':p.name(),'created':p.create_time()} for p in me.parents()];assert not any('codex' in x['name'].lower() or 'chatgpt' in x['name'].lower() for x in anc);me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 with (R/'记录/PREPARE.lock').open('x') as f:f.write(str(me.pid))
 assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==SHA
 assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
 dump(R/'STATE.json',{'pid':me.pid,'phase':'pytest','ancestor_proof':anc})
 with (R/'pytest.log').open('wb') as fo,(R/'记录/pytest.err.log').open('wb') as fe:
  cmd=[PY,'-B','-m','pytest','-q','tests/test_t2_stage.py','-o','cache_dir='+str(R/'记录/pytest_cache'),'--basetemp',str(R/'记录/pytest_tmp')]
  t=time.time();p=subprocess.Popen(cmd,cwd=REPO,stdout=fo,stderr=fe,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS);dump(R/'记录/PYTEST_LAUNCH.json',{'pid':p.pid,'command':cmd});rc=p.wait()
 text=(R/'pytest.log').read_text(encoding='utf-8',errors='replace');m=re.search(r'(\d+) passed',text);assert rc==0 and m and int(m[1])>=37,text
 dump(R/'记录/PYTEST_VERIFIED.json',{'status':'passed','science_SHA':SHA,'expected_passed':37,'actual_passed':int(m[1]),'wall_s':time.time()-t,'returncode':rc})
 reused=[]
 for arm,ks in (('e_sac_s0',(0,1000,2000,3000,4000)),('e_sac_s1',(1000,2000,3000))):
  for k in ks:
   src=OLD/f'B_eval/{arm}/step_{k}';dest=R/f'B_eval/{arm}/step_{k}';dest.mkdir(parents=True,exist_ok=False);fs=sorted(src.glob('*.json'));assert len(fs)==24 and not any(f.name.startswith('ERROR') for f in fs)
   for f in fs:
    shutil.copy2(f,dest/f.name);h=sha(f);assert sha(dest/f.name)==h;reused.append({'source':str(f),'destination':str(dest/f.name),'SHA256':h})
 dump(R/'记录/SAC_REUSE.json',reused)
 audits=[]
 for src in [OLD/'A1_LABEL_AUDIT.json',OLD/'A1_CAL.json',*sorted((OLD/'A1_audit').glob('audit_*.json'))]:
  dest=R/src.relative_to(OLD);dest.parent.mkdir(exist_ok=True);shutil.copy2(src,dest);assert sha(src)==sha(dest);audits.append({'source':str(src),'destination':str(dest),'SHA256':sha(src)})
 assert len(list((R/'A1_audit').glob('audit_*.json')))==6
 dump(R/'记录/AUDIT_REUSE.json',{'source_science_SHA':'84ee9c1045f32bbbacfef6cd220fdad55c5765c8','not_rerun':True,'reason':'T2.5.1 order explicitly reuses unchanged environment/label audit','files':audits})
 src=OLD/'记录/STOP_MECHANISM_PLACEHOLDER.json'
 stop=OLD/'发布快照/stopped/assets/STOP_MECHANISM_READOUT.json';assert stop.exists();shutil.copy2(stop,R/'T25_STOP_MECHANISM_READOUT.json');shutil.copy2(OLD/'记录/STOP_USER_T251_PRIORITY.json',R/'T25_STOP_RECORD.json')
 dump(R/'记录/BASELINE_PROVENANCE.json',{'T25_stop_mechanism_source':str(stop),'SHA256':sha(stop),'SAC_count':len(reused),'audit_reused_count':6})
 dump(R/'STATE.json',{'pid':me.pid,'phase':'preparation_complete','SAC_count':len(reused),'audit_reused_count':6,'actual_tests_passed':int(m[1]),'ancestor_proof':anc});dump(R/'记录/PREPARE_EXIT.json',{'status':'passed','SAC_count':192,'audit_reused':True})
if __name__=='__main__':
 try:main()
 except Exception:dump(R/'记录/PREPARE_ERROR.json',{'traceback':traceback.format_exc()});raise
