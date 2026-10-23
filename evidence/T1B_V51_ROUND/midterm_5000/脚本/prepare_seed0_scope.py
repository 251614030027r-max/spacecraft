"""Apply the user's pilot scope to scheduling/reporting only; preserve scientific PIDs."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'脚本/run_round_accel2.py').read_text(encoding='utf-8')
s=s.replace('PIPELINE_ACCEL2.lock','PIPELINE_SEED0.lock').replace("scheduler='run_round_accel2.py'","scheduler='run_seed0_pilot.py'")
s=s.replace("def launch_pair(seed):\n", "def launch_pair(seed):\n    assert seed in (0,1) and seed in read(ROOT/'记录/RUN_SCOPE.json')['automatic_seeds']\n")
s=s.replace("            mem=memory();running=[k for k,p in children.items() if p.poll() is None]", "            scope=read(ROOT/'记录/RUN_SCOPE.json');allowed=tuple(scope['automatic_seeds']);assert 0 in allowed and set(allowed).issubset({0,1})\n            S.update(run_scope=scope,requested_runs=2*len(allowed),planned_evaluations=128*len(allowed),conditional_reserved_seeds=[1] if 1 not in allowed else [],cancelled_seeds=[2],phase='effectiveness_pilot_training_and_readout')\n            mem=memory();running=[k for k,p in children.items() if p.poll() is None]")
s=s.replace("if len(S['started_seeds'])<3 and primed and not inflight", "if any(seed not in S['started_seeds'] for seed in allowed) and primed and not inflight")
s=s.replace("launch_pair(len(S['started_seeds']));last_admission", "launch_pair(next(seed for seed in allowed if seed not in S['started_seeds']));last_admission")
s=s.replace("queued_seeds=[s for s in (0,1,2) if s not in S['started_seeds']]", "queued_seeds=[s for s in allowed if s not in S['started_seeds']]")
s=s.replace("if len(children)==6 and not running and not pending and not inflight and len(scheduled)==24:", "if len(children)==2*len(allowed) and not running and not pending and not inflight and len(scheduled)==8*len(allowed):")
s=s.replace("S['phase']='complete_three_seeds_readout_required'", "S['phase']='complete_effectiveness_pilot_readout_required'")
s=s.replace("    for _ in range(20):", "    # Drain only the already submitted read-only evaluations before rebuilding their pool.\n    drain=read(ROOT/'记录/SCOPE_CHANGE_DRAIN.json')\n    for _ in range(300):\n        if all(Path(p).exists() for p in drain.get('result_files',[])):break\n        time.sleep(2)\n    for pid in drain.get('readonly_worker_pids',[]):\n        assert pid not in (26136,14684,1044,20096)\n        subprocess.run(['taskkill','/PID',str(pid),'/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)\n    S['completed_evaluations']=sum(1 for p in ROOT.glob('eval_e_*/*.json') if 'opening' in read(p))\n    S['completed_shared_0k']=sum(1 for p in (ROOT/'eval_shared_0k').glob('*.json') if 'opening' in read(p));save()\n    for _ in range(20):")
compile(s,str(ROOT/'脚本/run_seed0_pilot.py'),'exec');(ROOT/'脚本/run_seed0_pilot.py').write_text(s,encoding='utf-8')
p=ROOT/'脚本/publish_readout.py';s=p.read_text(encoding='utf-8')
s=s.replace('Six runs authorized from start; only seed0 M gate; no V3 L gate','Latest user scope: seed0 paired effectiveness pilot; seed1 conditional reserve only, seed2 cancelled. No final three-seed figure campaign.')
s=s.replace('用户授权seed0–2两臂各20k；预留一个物理核（逻辑CPU0/1）、BelowNormal。当前物理/提交内存仅按完整双臂配对准入，容量不足时种子排队；不是六运行均已启动。', '用户最新范围：资源优先seed0两臂各20k效果验证；seed1仅作为有证据需要时的条件复核，不自动排队；seed2取消，不做三种子最终出图。预留一个物理核（逻辑CPU0/1）、BelowNormal。')
s=s.replace('只有三个seed全齐才生成三seed均值。技术/M/非零梯度不等于科研成功。','本轮至多两个种子，三种子均值字段不适用且保持null，不以此要求加训。单seed仅说明本次效果，不能替代最终统计结论；技术/M/非零梯度不等于科研成功。')
compile(s,str(p),'exec');p.write_text(s,encoding='utf-8')
