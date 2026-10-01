# Stage C 独立交付

生成时间：2026-10-02T05:18:29.903785+09:00。官方判定：C_STOP；执行提交：e6694268634be63d5d66a4da637128ac2c5123a1。

本轮固定31维core/target_attitude/remaining_time，三模型池化、每轨迹等权，按开局6折，5个31→64→64→1网络、400轮；τ由预注册最小网格值且折外加权精度≥95%确定。只此一次拟合，不调参、不加数据。C3固定266000–266047，7行共336回合。效率要求消除相对Pure额外时间与Δv各至少一半，违规回合不多于Pure，毁掉Pure成功≤2；至少2模型通过。

必要历史对照：B块Pure 37/48、learned-only 42/33/38；B1非退化6/14/10，PROCEED。B2有效，毁掉交接轨迹18/42、10/33、18/38；节时中位59.7/67/57.5秒，对应Δv变化中位-0.384/-0.510/-0.307 m/s。原始状态相关，事后交接最优点不是在线收益。V3e价值第二轮does not hold不变。

B2已在主工程，以已核验ZIP的FILES_SHA256逐项比对144个B2与144个B1原始JSON，无需重复复制；PRECHECK中列出哈希、模型SHA、指定测试结果和预注册SHA。

## C2官方报告
```json
{
  "preregistration": "docs/STAGE_C_PREREGISTRATION_20261002.md",
  "feature_blocks": [
    "core",
    "target_attitude",
    "remaining_time"
  ],
  "constants": {
    "n_folds": 6,
    "ensemble_seeds": [
      0,
      1,
      2,
      3,
      4
    ],
    "hidden": 64,
    "epochs": 400,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "tau_grid": [
      0.5,
      0.55,
      0.6,
      0.65,
      0.7,
      0.75,
      0.8,
      0.85,
      0.9,
      0.95,
      0.97,
      0.99
    ],
    "target_precision": 0.95
  },
  "labelled_states": 4142,
  "labelled_trajectories": 144,
  "positive_fraction_weighted": 0.7637213468551636,
  "oof_weighted_auc": 0.773846997421673,
  "precision_curve": [
    {
      "tau": 0.5,
      "weighted_precision": 0.8289603590965271,
      "weighted_coverage": 0.8494868808322482
    },
    {
      "tau": 0.55,
      "weighted_precision": 0.838436484336853,
      "weighted_coverage": 0.8303689956665039
    },
    {
      "tau": 0.6,
      "weighted_precision": 0.8440619111061096,
      "weighted_coverage": 0.8118173281351725
    },
    {
      "tau": 0.65,
      "weighted_precision": 0.8489707708358765,
      "weighted_coverage": 0.7952814102172852
    },
    {
      "tau": 0.7,
      "weighted_precision": 0.8520879745483398,
      "weighted_coverage": 0.7822019788953993
    },
    {
      "tau": 0.75,
      "weighted_precision": 0.8555989265441895,
      "weighted_coverage": 0.769898944430881
    },
    {
      "tau": 0.8,
      "weighted_precision": 0.8601464629173279,
      "weighted_coverage": 0.7452497482299805
    },
    {
      "tau": 0.85,
      "weighted_precision": 0.8621683120727539,
      "weighted_coverage": 0.7172500822279189
    },
    {
      "tau": 0.9,
      "weighted_precision": 0.8650612831115723,
      "weighted_coverage": 0.6912285486857096
    },
    {
      "tau": 0.95,
      "weighted_precision": 0.8689482808113098,
      "weighted_coverage": 0.6489317682054307
    },
    {
      "tau": 0.97,
      "weighted_precision": 0.8739699721336365,
      "weighted_coverage": 0.6172552108764648
    },
    {
      "tau": 0.99,
      "weighted_precision": 0.8869231343269348,
      "weighted_coverage": 0.5459250344170464
    }
  ],
  "tau": null,
  "c2_verdict": "C_STOP",
  "first_trigger_oof": {
    "262420": {
      "B1": {
        "trajectories": 0,
        "never_triggered": 0,
        "triggered_clean": 0,
        "triggered_fails": 0,
        "triggered_off_grid": 0,
        "trigger_k": []
      },
      "B2": {
        "trajectories": 0,
        "never_triggered": 0,
        "triggered_clean": 0,
        "triggered_fails": 0,
        "triggered_off_grid": 0,
        "trigger_k": []
      }
    },
    "262421": {
      "B1": {
        "trajectories": 0,
        "never_triggered": 0,
        "triggered_clean": 0,
        "triggered_fails": 0,
        "triggered_off_grid": 0,
        "trigger_k": []
      },
      "B2": {
        "trajectories": 0,
        "never_triggered": 0,
        "triggered_clean": 0,
        "triggered_fails": 0,
        "triggered_off_grid": 0,
        "trigger_k": []
      }
    },
    "262422": {
      "B1": {
        "trajectories": 0,
        "never_triggered": 0,
        "triggered_clean": 0,
        "triggered_fails": 0,
        "triggered_off_grid": 0,
        "trigger_k": []
      },
      "B2": {
        "trajectories": 0,
        "never_triggered": 0,
        "triggered_clean": 0,
        "triggered_fails": 0,
        "triggered_off_grid": 0,
        "trigger_k": []
      }
    }
  },
  "first_trigger_rows": [],
  "code_commit": "e6694268634be63d5d66a4da637128ac2c5123a1",
  "code_dirty": false
}
```

## 边界与执行记录
events及全部进程起止、日志见记录；C3完成前仅数文件与核对进程退出，未读中间结果。阶段并行墙钟不作正式实时性数据。没有自动启动E训练：用户最新指令及预注册要求用户启动；A仅准备训练指令后待用户，B等待上层改接口，C_STOP/FIDELITY_FAIL停止。
