# V2实际训练状态

STOPPED / 按门槛或技术失败停止，未调参重跑

|e_sac|5000|4496|1498|PASS so far|
|e_tcpf|2600|2096|698|PASS so far|

{
  "topic": "t1b_v2_pilot",
  "phase": "stopped_A1_failed",
  "pid": 14420,
  "scientific_commit": "a1698acefd0f3f4cf7298438bd842ef63ae0aee0",
  "run_order": "docs/collaboration/upper/run_orders/T1_B_V2_PILOT_20261019.md",
  "training": {
    "e_sac": {
      "pid": 25328,
      "command": [
        "D:\\py\\DRL2\\.venv\\Scripts\\python.exe",
        "-B",
        "-m",
        "experiments.t1b_v2",
        "train",
        "--arm",
        "e_sac",
        "--seed",
        "0",
        "--steps",
        "10000",
        "--expert",
        "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T1B学习修复V1_20261019\\结果\\expert\\expert_replay.pkl",
        "--output",
        "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T1B_V2_PILOT_20261019\\TRAIN_e_sac_s0"
      ],
      "output": "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T1B_V2_PILOT_20261019\\TRAIN_e_sac_s0"
    },
    "e_tcpf": {
      "pid": 5468,
      "command": [
        "D:\\py\\DRL2\\.venv\\Scripts\\python.exe",
        "-B",
        "-m",
        "experiments.t1b_v2",
        "train",
        "--arm",
        "e_tcpf",
        "--seed",
        "0",
        "--steps",
        "10000",
        "--expert",
        "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T1B学习修复V1_20261019\\结果\\expert\\expert_replay.pkl",
        "--output",
        "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T1B_V2_PILOT_20261019\\TRAIN_e_tcpf_s0"
      ],
      "output": "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T1B_V2_PILOT_20261019\\TRAIN_e_tcpf_s0"
    }
  },
  "active": [],
  "completed_evaluations": 32,
  "completed_scan": 24,
  "training_active": false,
  "scope": "seed0 two arms10k only; A1/A2 failures stop; no multi-seed",
  "gate_protocol": {
    "A1_marks": [
      2000,
      5000
    ],
    "A2_windows": "501–1500,1501–2500,...; complete1000-transition windows only",
    "A2_min_non_earliest": 0.05
  },
  "cpu_policy": {
    "logical": 12,
    "training_envs_per_arm": 4,
    "remaining_slots_for_true_jobs": "4 /8 /12 for2/1/0 active training arms",
    "numerical_library_threads_per_process": 1
  },
  "next_action": "Monitor preregistered gates; frozen eval/scan; deliver actual evidence, no automatic parameter changes or multi-seed.",
  "last_native_check_utc": "2026-10-19T09:55:33.127511+00:00",
  "eval_worker_budget": 4,
  "last_steps": {
    "e_sac": 5000,
    "e_tcpf": 2600
  },
  "stop_reason": {
    "type": "A1_failed",
    "arm": "e_sac",
    "mark": 5000,
    "evidence": {
      "pass": false,
      "original8_entropy_gt_005": 2,
      "actor_gradient_fraction_of_0k": 0.768436168660113,
      "baseline_observations": "Original8 gate; expanded8 reported separately; same actual V1 reset observations, unchanged frozen task/support."
    },
    "retune_or_restart_permitted": false
  }
}