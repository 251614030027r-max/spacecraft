# V5.1 actual state

RUNNING

{
  "topic": "t1b_v51_round",
  "phase": "effectiveness_pilot_training_and_readout",
  "pid": 20964,
  "scientific_commit": "b6f5c8a22728b6ca3970d6201525cd4047cba8d8",
  "run_order": "docs/collaboration/upper/run_orders/T1_B_V51_ROUND_20261020.md",
  "training": {
    "e_sac_s0": {
      "pid": 1044,
      "command": [
        "D:\\py\\DRL2\\.venv\\Scripts\\python.exe",
        "-B",
        "-m",
        "experiments.t1b_v2",
        "train",
        "--interface",
        "v5",
        "--arm",
        "e_sac",
        "--seed",
        "0",
        "--steps",
        "20000",
        "--expert",
        "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T1B_V3_ROUND_20261019\\expert\\expert_replay.pkl",
        "--output",
        "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T1B_V51_ROUND_20261020\\TRAIN_e_sac_s0"
      ],
      "output": "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T1B_V51_ROUND_20261020\\TRAIN_e_sac_s0"
    },
    "e_tcpf_s0": {
      "pid": 20096,
      "command": [
        "D:\\py\\DRL2\\.venv\\Scripts\\python.exe",
        "-B",
        "-m",
        "experiments.t1b_v2",
        "train",
        "--interface",
        "v5",
        "--arm",
        "e_tcpf",
        "--seed",
        "0",
        "--steps",
        "20000",
        "--expert",
        "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T1B_V3_ROUND_20261019\\expert\\expert_replay.pkl",
        "--output",
        "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T1B_V51_ROUND_20261020\\TRAIN_e_tcpf_s0"
      ],
      "output": "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T1B_V51_ROUND_20261020\\TRAIN_e_tcpf_s0"
    }
  },
  "training_active": true,
  "started_seeds": [
    0
  ],
  "requested_runs": 2,
  "planned_steps_each": 20000,
  "completed_evaluations": 80,
  "completed_shared_0k": 16,
  "resource_policy": {
    "physical_core_masks": [
      3,
      12,
      48,
      192,
      768,
      3072
    ],
    "reserved_physical_core_mask": "0x3",
    "experiment_affinity_mask": 4092,
    "logical_cpus_reserved": 2,
    "priority": "BelowNormal",
    "numerical_threads": 1,
    "max_requested_pairs": 3,
    "commit_reserve_gb": 4.0,
    "physical_free_floor_gb": 2.0,
    "admission": "Add whole SAC/TCPF pairs only after resident manifest/step0 and measured memory admission; otherwise queue; never change n_envs/hyperparameters",
    "prior_resource_failure": "V4 actual WinError1455 and system2004; all old artifacts retained"
  },
  "resource_admission_note": "All six authorized; add matched pairs only when current physical/commit capacity admits them. This may queue seeds to avoid repeating confirmed V4 host exhaustion. Science unchanged.",
  "updated_jst": "2026-10-20T15:47:18.483502+09:00",
  "active_runs": [
    "e_sac_s0",
    "e_tcpf_s0"
  ],
  "queued_seeds": [],
  "last_steps": {
    "e_sac_s0": 18900,
    "e_tcpf_s0": 5500
  },
  "memory": {
    "free_physical_gb": 3.854724884033203,
    "free_commit_gb": 4.817020416259766,
    "commit_limit_gb": 31.855525970458984,
    "used_commit_gb": 27.03850555419922
  },
  "eval_worker_budget": 3,
  "active_evaluations": [],
  "pending_evaluations": 0,
  "scheduler": "run_seed0_pilot.py",
  "scheduler_only_adoption": true,
  "resource_adjustment": "User requested modest acceleration; existing scientific PIDs preserved; eval target3 with2.8GB commit/1.2GB physical reserve. Original resource_policy above is admission baseline, RUNTIME_RESOURCE_POLICY is current.",
  "run_scope": {
    "automatic_seeds": [
      0
    ],
    "conditional_reserved_seeds": [
      1
    ],
    "cancelled_seeds": [
      2
    ],
    "maximum_seeds_this_round": 2,
    "purpose": "Effectiveness pilot, not final figure/statistical campaign",
    "seed1_rule": "Only if complete seed0 evidence warrants a seed-effect replication check; not automatically queued and never used to evade M failure",
    "steps_per_method": 20000,
    "methods": [
      "e_sac",
      "e_tcpf"
    ],
    "jst": "2026-10-20T13:08:30.1394029+09:00"
  },
  "conditional_reserved_seeds": [
    1
  ],
  "cancelled_seeds": [
    2
  ],
  "planned_evaluations": 128
}