# V4 actual state

STOPPED

{
  "topic": "t1b_v4_resource_restart1",
  "phase": "stopped_host_reboot_5k_review_complete",
  "pid": 14460,
  "scientific_commit": "8684cca3d3c3a2997f3ed3f2c106c247b94e20e4",
  "run_order": "T1_B_V4_ROUND_20261019 plus2026-10-20 user autonomous deadline authorization",
  "training": {
    "e_sac_s0": {
      "pid": 25976,
      "command": [
        "D:\\py\\DRL2\\.venv\\Scripts\\python.exe",
        "-B",
        "-m",
        "experiments.t1b_v2",
        "train",
        "--interface",
        "v4",
        "--arm",
        "e_sac",
        "--seed",
        "0",
        "--steps",
        "20000",
        "--expert",
        "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T1B_V3_ROUND_20261019\\expert\\expert_replay.pkl",
        "--output",
        "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T1B_V4_ROUND_20261019_restart1\\TRAIN_e_sac_s0"
      ],
      "output": "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T1B_V4_ROUND_20261019_restart1\\TRAIN_e_sac_s0"
    },
    "e_tcpf_s0": {
      "pid": 13696,
      "command": [
        "D:\\py\\DRL2\\.venv\\Scripts\\python.exe",
        "-B",
        "-m",
        "experiments.t1b_v2",
        "train",
        "--interface",
        "v4",
        "--arm",
        "e_tcpf",
        "--seed",
        "0",
        "--steps",
        "20000",
        "--expert",
        "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T1B_V3_ROUND_20261019\\expert\\expert_replay.pkl",
        "--output",
        "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T1B_V4_ROUND_20261019_restart1\\TRAIN_e_tcpf_s0"
      ],
      "output": "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T1B_V4_ROUND_20261019_restart1\\TRAIN_e_tcpf_s0"
    }
  },
  "training_active": false,
  "completed_evaluations": 64,
  "active_evaluations": [],
  "cpu_policy": {
    "each_arm_envs": 4,
    "active_training_runs": 2,
    "eval_slots": "4/8/12 as2/1/0 arms remain; pool destroyed before next seed",
    "blas_threads": 1
  },
  "authorization": "Fresh technical_restart1, same frozen learner/epsilon/task; seed0,1,2 pairs sequential after real host commit-limit failure. No V3 L gate.",
  "restart_provenance": "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T1B_V4_ROUND_20261019\\记录\\RESOURCE_STOP_RECEIPT.json",
  "updated_utc": "2026-10-20T01:58:39.6445+09:00",
  "seed": 0,
  "eval_worker_budget": 4,
  "last_steps": {
    "e_sac": 6200,
    "e_tcpf": 5100
  },
  "stop_reason": "Host crash/reboot 01:59 JST; bugcheck0x9F and AMD amdkmdag LiveKernelEvent141. No science processes alive at10:35 verification; exact crash driver unresolved.",
  "verified_jst": "2026-10-20T10:36:07.9356646+09:00",
  "checkpoint_evaluation_active": false,
  "evaluation_receipt": "C:/Users/35884/Documents/Spacecraft/过程文件/T1B_V4_ROUND_20261019_restart1/记录/FINISH_5K_STATUS.json",
  "review_completed_jst": "2026-10-20T10:51:05.117471+09:00",
  "user_latest_scope": "Complete matched frozen5k evaluation and push key data only; no20k training started"
}