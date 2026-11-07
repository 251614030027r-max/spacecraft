"""Prepare V5.1 administrative drivers and verify the unchanged scientific freeze."""
import os,sys,json,hashlib,subprocess,ctypes,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SPACE=ROOT.parents[1];OLD=SPACE/'过程文件/T1B_V4_ROUND_20261019_restart1';REPO=Path('D:/py/DRL2');PY=str(REPO/'.venv/Scripts/python.exe')
FROZEN='b6f5c8a22728b6ca3970d6201525cd4047cba8d8'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ.update(GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='safe.directory',GIT_CONFIG_VALUE_0=str(REPO))
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    class CORE(ctypes.Structure):_fields_=[('mask',ctypes.c_size_t),('relation',ctypes.c_int),('padding',ctypes.c_int),('data',ctypes.c_byte*16)]
    n=ctypes.c_ulong();api=ctypes.windll.kernel32.GetLogicalProcessorInformation;api(None,ctypes.byref(n));buf=ctypes.create_string_buffer(n.value);assert api(buf,ctypes.byref(n))
    cores=[CORE.from_buffer_copy(buf.raw[i:i+ctypes.sizeof(CORE)]).mask for i in range(0,n.value,ctypes.sizeof(CORE)) if CORE.from_buffer_copy(buf.raw[i:i+ctypes.sizeof(CORE)]).relation==0]
    reserved=cores[0];allowed=((1<<os.cpu_count())-1)^reserved
    policy={'physical_core_masks':cores,'reserved_physical_core_mask':hex(reserved),'experiment_affinity_mask':allowed,'logical_cpus_reserved':reserved.bit_count(),'priority':'BelowNormal','numerical_threads':1,'max_requested_pairs':3,'commit_reserve_gb':4.0,'physical_free_floor_gb':2.0,'admission':'Add whole SAC/TCPF pairs only after resident manifest/step0 and measured memory admission; otherwise queue; never change n_envs/hyperparameters','prior_resource_failure':'V4 actual WinError1455 and system2004; all old artifacts retained'}
    dump(ROOT/'记录/CPU_MEMORY_POLICY.json',policy)
    k=ctypes.windll.kernel32;k.SetPriorityClass(k.GetCurrentProcess(),0x4000);k.SetProcessAffinityMask(k.GetCurrentProcess(),ctypes.c_size_t(allowed))
    assert subprocess.check_output(['git','-c','safe.directory=D:/py/DRL2','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==FROZEN
    expert=SPACE/'过程文件/T1B_V3_ROUND_20261019/expert/expert_replay.pkl';assert sha(expert)=='9c91cb76f10999ca36aecf4172bb092e4e52de9fcfbf62402a76306e701e1d0f'
    dump(ROOT/'记录/ASSET_PROTECTION.json',{'frozen':FROZEN,'expert_sha256':sha(expert),'old5k_models':{str(p):sha(p) for p in OLD.glob('TRAIN_*/checkpoints/step_5000.zip')},'old_round_unmodified':True})
    for name in ('readonly_eval.py','policy_audit.py'):
        s=(OLD/'脚本'/name).read_text(encoding='utf-8')
        s=s.replace('FeasibleArrivalEnvV4','FeasibleArrivalEnvV5').replace('env.t1_b_feasible_v4','env.t1_b_feasible_v5').replace('FeasibleSupportSACV4','FeasibleSupportSACV5').replace('train.t1_b_v4_sac','train.t1_b_v5_sac').replace("interface=='v4'","interface=='v5'").replace("'interface':'v4'","'interface':'v5'")
        (ROOT/'脚本'/name).write_text(s,encoding='utf-8')
    # All logs, pytest cache and tmp stay under this round from their creation.
    with (ROOT/'记录/pytest.log').open('w',encoding='utf-8') as f:
        rc=subprocess.run([PY,'-B','-m','pytest','-q','tests/test_t1_b_v2.py','tests/test_t1_b.py','tests/test_t1_b_learning.py','tests/test_t1_b_checkpoint_loading.py','-o','cache_dir='+str(ROOT/'记录/pytest_cache'),'--basetemp',str(ROOT/'记录/pytest_tmp')],cwd=REPO,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW).returncode
    dump(ROOT/'记录/PYTEST_EXIT.json',{'exit_code':rc,'frozen':FROZEN});assert rc==0
    with (ROOT/'记录/contract.log').open('w',encoding='utf-8') as f:
        rc=subprocess.run([PY,'-B','-m','experiments.t1b_v2','contract','--interface','v5','--output',str(ROOT/'contract')],cwd=REPO,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW).returncode
    contract=json.loads((ROOT/'contract/CONTRACT.json').read_text());assert rc==0 and contract['passed'] and contract['behaviour_epsilon']==.5
    dump(ROOT/'记录/PREFLIGHT_EXIT.json',{'passed':True,'pytest_exit':0,'contract_exit':0,'frozen':FROZEN,'cpu_memory_policy':policy,'jst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()})
if __name__=='__main__':main()
