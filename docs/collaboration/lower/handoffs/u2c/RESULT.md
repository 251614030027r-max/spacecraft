# RUN_U2C 决策时认证配对评估

冻结科学提交 28a4079907c4d7a18c5c44e17383315e0d4a9705，collab 执行书 bfee66ae。复用预实验 EACH、SAC 的种子 10、11 各一个 2000 步模型。认证开关均评估 294220–294267，共 384 例。

自检全部通过。pytest 退出码 0，四个模型 SHA256 与执行书一致，认证关闭时 EACH s10 的开局 294100、294101 逐项一致。原生 CHECK 完整，独立核对了各组开局、标签、认证开关和介入数。

| 对比（各 96 对） | 原生判定 | 平均回报差 | 捕获数 |
|---|---|---:|---:|
| EACH + 认证 − SAC | pass | 23.3434460697025 | 83 / 72 |
| EACH + 认证 − EACH | no_effect | 6.30836172039615 | 83 / 80 |
| SAC + 认证 − SAC | no_effect | 5.4978172147255 | 75 / 72 |

原生判据、种子配对与介入次数如下。

```json
{
  "main": {
    "comparison": "EACH_CERT - SAC",
    "pairs": 96,
    "mean": 23.343446069702463,
    "median": 0.4374473538148358,
    "without_top2_per_seed": 15.48756159412435,
    "per_seed": {
      "10": {
        "n": 48,
        "mean": 37.94949882814581,
        "median": 0.332484493230659,
        "without_top2": 30.752507261294465
      },
      "11": {
        "n": 48,
        "mean": 8.7373933112591,
        "median": 0.5256627624590955,
        "without_top2": 0.22261592695422913
      }
    },
    "capture": [
      83,
      72
    ],
    "unsafe_first": 0,
    "flips": {
      "only_first_captures": 12,
      "only_second_captures": 1
    },
    "both_captured": {
      "n": 71,
      "time_s": [
        132.7535211267606,
        137.74788732394367
      ],
      "delta_v": [
        1.323830766848071,
        1.3686495241496044
      ]
    },
    "gates": {
      "G1": true,
      "G2": true,
      "G3": true,
      "G4": true,
      "G5": true
    },
    "verdict": "pass"
  },
  "u2_each": {
    "comparison": "EACH_CERT - EACH",
    "pairs": 96,
    "mean": 6.308361720396154,
    "median": 0.0,
    "without_top2_per_seed": 2.1876581969710025,
    "per_seed": {
      "10": {
        "n": 48,
        "mean": 12.704861085625732,
        "median": 0.0,
        "without_top2": 4.4786114800844725
      },
      "11": {
        "n": 48,
        "mean": -0.08813764483342545,
        "median": 0.0,
        "without_top2": -0.10329508614246787
      }
    },
    "capture": [
      83,
      80
    ],
    "unsafe_first": 0,
    "flips": {
      "only_first_captures": 3,
      "only_second_captures": 0
    },
    "both_captured": {
      "n": 80,
      "time_s": [
        133.48375000000001,
        133.52375
      ],
      "delta_v": [
        1.3322314233094226,
        1.3311141915427132
      ]
    },
    "gates": {
      "G1": false,
      "G2": true,
      "G3": false,
      "G4": true,
      "G5": true
    },
    "urgent": {
      "n": 54,
      "mean": 11.214865280704274,
      "capture": [
        41,
        38
      ]
    },
    "economy": {
      "n": 42,
      "mean": 0.0,
      "capture": [
        42,
        42
      ]
    },
    "verdict": "no_effect"
  },
  "u2_sac": {
    "comparison": "SAC_CERT - SAC",
    "pairs": 96,
    "mean": 5.497817214725501,
    "median": 0.0,
    "without_top2_per_seed": 1.3875376223290457,
    "per_seed": {
      "10": {
        "n": 48,
        "mean": 11.001137650831284,
        "median": 0.0,
        "without_top2": 2.901127132267689
      },
      "11": {
        "n": 48,
        "mean": -0.005503221380282734,
        "median": 0.0,
        "without_top2": -0.1260518876095977
      }
    },
    "capture": [
      75,
      72
    ],
    "unsafe_first": 0,
    "flips": {
      "only_first_captures": 4,
      "only_second_captures": 1
    },
    "both_captured": {
      "n": 71,
      "time_s": [
        135.8507042253521,
        135.95774647887325
      ],
      "delta_v": [
        1.3414952384625973,
        1.3419527640660576
      ]
    },
    "gates": {
      "G1": false,
      "G2": true,
      "G3": false,
      "G4": true,
      "G5": true
    },
    "urgent": {
      "n": 54,
      "mean": 13.56333878975113,
      "capture": [
        34,
        30
      ]
    },
    "economy": {
      "n": 42,
      "mean": -4.872139096021738,
      "capture": [
        41,
        42
      ]
    },
    "verdict": "no_effect"
  },
  "interventions": {
    "EACH_CERT_s10": 16,
    "EACH_CERT_s11": 5,
    "SAC_CERT_s10": 50,
    "SAC_CERT_s11": 17
  }
}
```

墙钟（评估阶段）：all done 1.56959648955556 h。

科学证据位于 evidence/U2C，提交 bcab9207030cd08d42c6a308a01ac435029dc9be，397 个文件已核对 Git 原字节。任务结束，无新训练。
