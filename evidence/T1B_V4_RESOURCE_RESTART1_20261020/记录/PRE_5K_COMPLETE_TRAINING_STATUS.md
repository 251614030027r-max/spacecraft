# V4 actual state

RUNNING

{
  "topic": "t1b_v4_resource_restart1",
  "phase": "seed0_training_and_readout",
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
  "training_active": true,
  "completed_evaluations": 48,
  "active_evaluations": [
    [
      292140,
      "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T1B_V4_ROUND_20261019_restart1\\TRAIN_e_tcpf_s0\\checkpoints\\step_5000.zip",
      "e_tcpf_s0_5000",
      "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T1B_V4_ROUND_20261019_restart1\\eval_e_tcpf_s0_5000",
      "v4"
    ],
    [
      292141,
      "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T1B_V4_ROUND_20261019_restart1\\TRAIN_e_tcpf_s0\\checkpoints\\step_5000.zip",
      "e_tcpf_s0_5000",
      "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T1B_V4_ROUND_20261019_restart1\\eval_e_tcpf_s0_5000",
      "v4"
    ],
    [
      292142,
      "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T1B_V4_ROUND_20261019_restart1\\TRAIN_e_tcpf_s0\\checkpoints\\step_5000.zip",
      "e_tcpf_s0_5000",
      "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T1B_V4_ROUND_20261019_restart1\\eval_e_tcpf_s0_5000",
      "v4"
    ],
    [
      292143,
      "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T1B_V4_ROUND_20261019_restart1\\TRAIN_e_tcpf_s0\\checkpoints\\step_5000.zip",
      "e_tcpf_s0_5000",
      "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T1B_V4_ROUND_20261019_restart1\\eval_e_tcpf_s0_5000",
      "v4"
    ]
  ],
  "cpu_policy": {
    "each_arm_envs": 4,
    "active_training_runs": 2,
    "eval_slots": "4/8/12 as2/1/0 arms remain; pool destroyed before next seed",
    "blas_threads": 1
  },
  "authorization": "Fresh technical_restart1, same frozen learner/epsilon/task; seed0,1,2 pairs sequential after real host commit-limit failure. No V3 L gate.",
  "restart_provenance": "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T1B_V4_ROUND_20261019\\记录\\RESOURCE_STOP_RECEIPT.json",
  "updated_utc": "2026-10-19T16:57:39.510361+00:00",
  "seed": 0,
  "eval_worker_budget": 4,
  "last_steps": {
    "e_sac": 6100,
    "e_tcpf": 5000
  }
}