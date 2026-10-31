# T2.2 deadline_0900 事实交付（2026-10-23）

冻结科学 `fa54ed1391602386063290b7d34222313a4a427e`；collab执行书 `1745c28`；生成时间 2026-10-23T08:30:25.770763+09:00。
仅本单有界测试，非正式出图。不改代码、开关、开局、步数和判定阈值；新旧回报比较只用于总体读数，同版本同种子MVE−SAC才是本轮耦合配对。

状态：截止时刻部分结果，仍在继续；不是完整判定。

## 判定文件（原生 t2_decide，无人工改写）

## 曲线：每个固定检查点，含任务分层

| 读出 | 臂 | 步数 | 任务 | n | 平均回报 | 捕获 | unsafe |
|---|---|---:|---|---:|---:|---:|---:|
尚无完整原生曲线文件；真实已完成数量见RUNTIME_SNAPSHOT和必要原始JSON，禁止把少量完成开局当总体结果。

部分进度的覆盖见PROGRESS_COVERAGE.json；DECISION_PROGRESS_ONLY.json仅调用冻结逻辑报告当前可判定性，不是第1段最终DECISION_1。完整部署读出不存在时C0=incomplete，C1–C5尚不能计算，不能选择第二段分支。已有检查点的本机路径及稳定SHA见LOCAL_MODEL_SHA256.json，模型ZIP不入Git。

## 配对和固定参照

完整 paired 字段见相应原生READOUT以及PAIRED.csv；不挑检查点、不删除负差。

V-A表：c_*、标准分层b_*、pure和gstar。G★空间外参照不得混入同动作空间最优策略；VA门仅如实列出，本单分支由DECISION_1决定。
```json
{
  "policies": null,
  "gates": null,
  "errors": null,
  "gain_vs_best_constant": null,
  "gain_vs_rally_rule_report_only": null
}
```

上一轮固定历史参照（3107420，完整24例）：MVE3000均23.4217272911、捕获20/24、unsafe0；SAC均12.8969787684、捕获19/24、unsafe1；配对均差10.5247485226、去最大两例−2.3966303800。对应最终主证据science26fd421、补充science28c96e0。本次C1沿用原生常量，不把跨版本改善当耦合因果证据。

## 门原因和入口事件一致率

训练replay.gate、入口候选数、manifest、模型项探针原值见GATE_AND_MODEL_AUDIT.json；未完成训练只列已有heartbeat，不当作finished。冻结训练日志只存门原因计数，没有所有“预测入口/实际入口”布尔值，故总体入口事件一致率标MISSING。明确记录的“实际入口一致”与entry_event_mismatch子集另列分子分母，不能冒充所有样本一致率。评估逐决策的可辨识计数和其限制见ENTRY_EVENT_LOG_LIMITS.json。保持科学代码不变，不为补日志改训练。

## 墙钟、资源和故障
```json
{
  "pid": 3312,
  "started_jst": "2026-10-22T23:29:25.347094+09:00",
  "science_SHA": "fa54ed1391602386063290b7d34222313a4a427e",
  "order_collab": "1745c28",
  "active_stage": "phase1",
  "seed_scope": [
    0,
    1
  ],
  "formal_training": false,
  "deadline_jst": "2026-10-23T09:00:00+09:00",
  "desktop_reserved": false,
  "logical_cpus": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11
  ],
  "updated_jst": "2026-10-23T08:30:24.779214+09:00",
  "memory": {
    "commit_free_GiB": 18.998424530029297,
    "physical_free_GiB": 3.3513031005859375,
    "commit_limit_GiB": 45.855525970458984
  },
  "phase": "phase1_training",
  "training_jobs": [
    {
      "tag": "TRAIN_e_sac_s0",
      "arm": "e_sac",
      "seed": 0,
      "steps": 3000,
      "marks": "1000,2000",
      "extra": []
    },
    {
      "tag": "TRAIN_e_sac_s1",
      "arm": "e_sac",
      "seed": 1,
      "steps": 3000,
      "marks": "1000,2000",
      "extra": []
    },
    {
      "tag": "TRAIN_e_mve_s0",
      "arm": "e_mve",
      "seed": 0,
      "steps": 3000,
      "marks": "1000,2000",
      "extra": []
    },
    {
      "tag": "TRAIN_e_mve_s1",
      "arm": "e_mve",
      "seed": 1,
      "steps": 3000,
      "marks": "1000,2000",
      "extra": []
    }
  ],
  "training": {
    "TRAIN_e_sac_s0": {
      "pid": 23244,
      "start_jst": "2026-10-22T23:29:25.355598+09:00",
      "attempt": 0,
      "command": [
        "D:\\py\\DRL2\\.venv\\Scripts\\python.exe",
        "-B",
        "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T2_2_TEST_20261022\\脚本\\native_entry.py",
        "t2",
        "train",
        "--arm",
        "e_sac",
        "--seed",
        "0",
        "--steps",
        "3000",
        "--envs",
        "3",
        "--checkpoints",
        "1000,2000",
        "--output",
        "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T2_2_TEST_20261022\\TRAIN_e_sac_s0"
      ]
    },
    "TRAIN_e_sac_s1": {
      "pid": 21336,
      "start_jst": "2026-10-22T23:29:30.379019+09:00",
      "attempt": 0,
      "command": [
        "D:\\py\\DRL2\\.venv\\Scripts\\python.exe",
        "-B",
        "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T2_2_TEST_20261022\\脚本\\native_entry.py",
        "t2",
        "train",
        "--arm",
        "e_sac",
        "--seed",
        "1",
        "--steps",
        "3000",
        "--envs",
        "3",
        "--checkpoints",
        "1000,2000",
        "--output",
        "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T2_2_TEST_20261022\\TRAIN_e_sac_s1"
      ]
    },
    "TRAIN_e_mve_s0": {
      "pid": 1776,
      "start_jst": "2026-10-22T23:29:35.393681+09:00",
      "attempt": 0,
      "command": [
        "D:\\py\\DRL2\\.venv\\Scripts\\python.exe",
        "-B",
        "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T2_2_TEST_20261022\\脚本\\native_entry.py",
        "t2",
        "train",
        "--arm",
        "e_mve",
        "--seed",
        "0",
        "--steps",
        "3000",
        "--envs",
        "3",
        "--checkpoints",
        "1000,2000",
        "--output",
        "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T2_2_TEST_20261022\\TRAIN_e_mve_s0"
      ]
    },
    "TRAIN_e_mve_s1": {
      "pid": 2368,
      "start_jst": "2026-10-22T23:29:40.416729+09:00",
      "attempt": 0,
      "command": [
        "D:\\py\\DRL2\\.venv\\Scripts\\python.exe",
        "-B",
        "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T2_2_TEST_20261022\\脚本\\native_entry.py",
        "t2",
        "train",
        "--arm",
        "e_mve",
        "--seed",
        "1",
        "--steps",
        "3000",
        "--envs",
        "3",
        "--checkpoints",
        "1000,2000",
        "--output",
        "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T2_2_TEST_20261022\\TRAIN_e_mve_s1"
      ]
    }
  },
  "training_progress": {
    "TRAIN_e_sac_s0": {
      "steps": 2001,
      "wall_s": 28433.042977809906,
      "episodes": 371,
      "replay": {
        "transitions": 1998,
        "with_candidates": 1121,
        "candidates": 8882,
        "entry_candidates": 1997,
        "gate": {
          "no_prediction": 57,
          "ok": 896,
          "stage_not_complete": 112,
          "entry_event_mismatch": 313,
          "entry_ok": 229,
          "factual_mismatch": 270,
          "entry_state_mismatch": 121
        }
      }
    },
    "TRAIN_e_sac_s1": {
      "steps": 2001,
      "wall_s": 27182.056378364563,
      "episodes": 383,
      "replay": {
        "transitions": 1998,
        "with_candidates": 1256,
        "candidates": 9858,
        "entry_candidates": 2073,
        "gate": {
          "no_prediction": 64,
          "factual_mismatch": 201,
          "ok": 1020,
          "stage_not_complete": 110,
          "entry_state_mismatch": 74,
          "entry_ok": 251,
          "entry_event_mismatch": 278
        }
      }
    },
    "TRAIN_e_mve_s0": {
      "steps": 2001,
      "wall_s": 30937.40734910965,
      "episodes": 407,
      "replay": {
        "transitions": 1998,
        "with_candidates": 1089,
        "candidates": 8631,
        "entry_candidates": 2013,
        "gate": {
          "no_prediction": 60,
          "ok": 839,
          "stage_not_complete": 116,
          "entry_event_mismatch": 346,
          "entry_ok": 252,
          "factual_mismatch": 266,
          "entry_state_mismatch": 119
        }
      }
    },
    "TRAIN_e_mve_s1": {
      "steps": 2001,
      "wall_s": 26640.284228801727,
      "episodes": 353,
      "replay": {
        "transitions": 1998,
        "with_candidates": 1158,
        "candidates": 9154,
        "entry_candidates": 1698,
        "gate": {
          "no_prediction": 55,
          "factual_mismatch": 277,
          "ok": 965,
          "stage_not_complete": 117,
          "entry_state_mismatch": 100,
          "entry_ok": 203,
          "entry_event_mismatch": 281
        }
      }
    }
  },
  "training_completed": [],
  "training_pending": []
}
```
全12逻辑核、BLAS/OMP各1，训练每进程3环境（不可改），四运行并行；筛选/评估按内存最高12工作进程，逐例释放。Windows隐藏后台，无系统驱动或页面文件修改。提交容量已由用户设置并重启生效，不能据此保证计算吞吐同比提升。
驱动、回合ERROR和重启记录仅本轮必要JSON随附；同配置同种子崩溃至多从零一次，不拼Replay/优化器。回合ERROR补算前原件归档记录，不删除失败记录。原生判定为incomplete时不得强行进入分支。

阶段1先推送并交付，然后只按原生判定走一个P/K/L分支；完成该分支即停止，不追加训练。09:00若未完整，明确缺项和实际运行状态，按已授权分支继续而不伪造按时完成。
必要原始结果包含于本段路径；模型ZIP仅本机保留，禁止入Git；SHA256SUMS与远端回执用于独立核验。
