from pathlib import Path
import subprocess,json,time
B=Path(r'C:\Users\35884\Documents\Spacecraft\过程文件\整理_20261010')
temp=B/'验证/pytest_retry';temp.parent.mkdir(parents=True,exist_ok=True)
R=B/'记录/RETURN_DRL2'
names=[
'tests/test_evaluator_observation_parity.py::test_evaluator_and_training_path_see_the_same_observations',
'tests/test_terminal_value.py::test_save_and_load_round_trip',
'tests/test_v3_cli.py::test_evaluator_cli_accepts_task_state_v3',
'tests/test_v3_handoff_scan.py::test_cli_writes_one_resumable_file_per_seed',
'tests/test_v3_mainline.py::test_evaluator_pure_and_nominal_rows_follow_the_mainline_path_bitwise',
'tests/test_v3_nominal_control.py::test_v3_nominal_runs_on_the_learned_branch_at_full_rate',
'tests/test_v3_stage_c.py::test_classifier_is_deterministic_and_round_trips',
'tests/test_v3_stage_c.py::test_record_cli_checks_against_b1',
'tests/test_v3_stopping.py::test_handoff_step_is_not_a_continue_transition_and_earns_its_updates',
'tests/test_v3_stopping.py::test_train_and_evaluate_cli',
'tests/test_v3_stopping_training_health.py::test_report_bins_and_flags',
'tests/test_v3_stopping_training_health.py::test_report_flags_a_dirty_or_smoke_run']
cmd=[r'D:\py\DRL2\.venv\Scripts\python.exe','-B','-m','pytest','-q','-p','no:cacheprovider','--basetemp='+str(temp),*names]
start=time.time()
with (R/'pytest_setup_retry.log').open('w',encoding='utf-8') as f:
    p=subprocess.run(cmd,cwd=r'D:\py\DRL2',stdout=f,stderr=subprocess.STDOUT)
result={'science_commit':'4795910102cf9fbf1dfec74b77e9449623a7b51e','initial_full_suite':{'passed':341,'xfailed':3,'setup_errors':13,'seconds':403.08,'reason':'Requested basetemp parent directory was not pre-created; no science-code error'},'retry_test_cases':13,'retry_exit_code':p.returncode,'retry_seconds':time.time()-start,'retry_log':str(R/'pytest_setup_retry.log'),'science_code_modified':False,'behavioral_6_episode_rerun':'Not performed: user cancelled further experimental simulations'}
(R/'TEST_VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print((R/'pytest_setup_retry.log').read_text(encoding='utf-8'));print(json.dumps(result,ensure_ascii=False))
raise SystemExit(p.returncode)
