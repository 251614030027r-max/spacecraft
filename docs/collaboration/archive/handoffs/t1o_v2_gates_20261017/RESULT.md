# T1-O V2本机门控：事实交付（2026-10-17）

科学冻结`d3902a9dd62dbef6cb318e36ff4c2527d7340ec9`；执行单第2版collab `08a8b66`。本轮仅Gate A存档复用及B/C/零步/E串联门控，不含Gate D或正式训练。旧T1-L-S正式长训取消，旧T1-L/v4资产保持只读。

停止位置：**C**。控制器终态：`CHAIN_STOPPED`。遇首个FAIL后未继续后续科学阶段；未生成的结果明确记为未执行，不补值。

## 阶段原始判定与关键数

### A：PASS

```json
{
  "stage": "A",
  "pass": true,
  "utc": "2026-10-17T07:58:10.195455+00:00",
  "verdict": "PASS",
  "fuel_weight": 20.0
}
```

### B：PASS

```json
{
  "stage": "B",
  "pass": true,
  "utc": "2026-10-17T08:46:20.552338+00:00",
  "uniform": "A15",
  "in_flight": {
    "states": 16,
    "real_effects": 11,
    "preferences": {
      "keep": 10,
      "shift": 1
    },
    "repeatability_max": 0.0006437373054595241
  },
  "near_field_abs_dJ_ge_1": 10
}
```

### C：FAIL

```json
{
  "stage": "C",
  "pass": false,
  "utc": "2026-10-17T09:49:22.118567+00:00",
  "outcomes": {
    "uniform": {
      "success": 31,
      "unsafe": 0,
      "timeout": 1
    },
    "state_rule": {
      "success": 31,
      "unsafe": 0,
      "timeout": 1
    },
    "gstar_rule": {
      "success": 30,
      "unsafe": 1,
      "timeout": 1
    },
    "A0": {
      "success": 31,
      "unsafe": 0,
      "timeout": 1
    }
  },
  "mean_return": {
    "uniform": 51.96389045501348,
    "state_rule": 53.17733723052007,
    "gstar_rule": 42.67968224157862,
    "A0": 53.20661355495858
  },
  "state_rule_minus_uniform": {
    "n": 32,
    "mean": 1.2134467755065852,
    "sd": 3.3366384270747496,
    "ci95": [
      0.01046033858493911,
      2.416433212428231
    ],
    "better": 8,
    "worse": 4
  },
  "gstar_rule_minus_uniform": {
    "n": 32,
    "mean": -9.284208213434855,
    "sd": 38.955466443904996,
    "ci95": [
      -23.32915158989144,
      4.76073516302173
    ],
    "better": 10,
    "worse": 14
  },
  "state_rule_minus_gstar_rule": {
    "n": 32,
    "mean": 10.49765498894144,
    "sd": 38.986600228548554,
    "ci95": [
      -3.55851331388185,
      24.55382329176473
    ],
    "better": 16,
    "worse": 7
  }
}
```

- C_zero_step：未执行或未完成判定，以原始文件和技术日志为准。
- E：未执行或未完成判定，以原始文件和技术日志为准。

## 核验、资源与边界

- 本机6物理核、12逻辑线程；按执行单W=5，Normal优先级，OMP/MKL/OPENBLAS各1。冻结Gate B入口不接收workers，内部默认4路，其余受chain传入W控制。没有为提高利用率修改冻结代码或追加参数。
- Gate A只允许上层Linux存档复用，不是本机重算；实际归档SHA和160回合计数以execution/记录/GATE_A_ARCHIVE_RECEIPT.json为准。若该回执尚未产生，表示在准入阶段已停止。保留上层判据与敏感开局事实，不把源数据称为同机复现。
- 指定测试、开局SHA、依赖逐文件比较、磁盘与阶段墙钟见execution/记录；完整控制器状态如下。墙钟与资源采样不冒充OS累计CPU时间，不作实时性声明。
- 仅如链条到达C_zero_step才按冻结代码生成1步模型，learning_starts=1000，无梯度更新；模型ZIP仅列SHA和路径，不上传。
- 未传--reproduce-gate-a或--through-d，未新增任何科学参数、规则或开局；未访问正式块300000–999999。
- 无人工重判或失败后扩展。Gate C在32独立确认开局机械判定，不用发现集成绩替代。Gate A的单开局敏感性、是否固定κ/v_in及后续GO均交上层/研究层决定。
- 本轮原始Gate A回合、实际产生的Gate B/C/零步/E JSON、完整chain log、模型manifest、命令/测试/资源回执随独立ZIP；模型不含在内。

```json
{
  "phase": "CHAIN_STOPPED",
  "pid": 23788,
  "utc": "2026-10-17T09:49:22.148724+00:00",
  "scientific_commit": "d3902a9dd62dbef6cb318e36ff4c2527d7340ec9",
  "workers": 5,
  "wall_s": 6898.1720000000205,
  "exit_code": 0,
  "chain": {
    "stages": [
      {
        "stage": "A",
        "pass": true,
        "utc": "2026-10-17T07:58:10.195455+00:00",
        "verdict": "PASS",
        "fuel_weight": 20.0
      },
      {
        "stage": "B",
        "pass": true,
        "utc": "2026-10-17T08:46:20.552338+00:00",
        "uniform": "A15",
        "in_flight": {
          "states": 16,
          "real_effects": 11,
          "preferences": {
            "keep": 10,
            "shift": 1
          },
          "repeatability_max": 0.0006437373054595241
        },
        "near_field_abs_dJ_ge_1": 10
      },
      {
        "stage": "C",
        "pass": false,
        "utc": "2026-10-17T09:49:22.118567+00:00",
        "outcomes": {
          "uniform": {
            "success": 31,
            "unsafe": 0,
            "timeout": 1
          },
          "state_rule": {
            "success": 31,
            "unsafe": 0,
            "timeout": 1
          },
          "gstar_rule": {
            "success": 30,
            "unsafe": 1,
            "timeout": 1
          },
          "A0": {
            "success": 31,
            "unsafe": 0,
            "timeout": 1
          }
        },
        "mean_return": {
          "uniform": 51.96389045501348,
          "state_rule": 53.17733723052007,
          "gstar_rule": 42.67968224157862,
          "A0": 53.20661355495858
        },
        "state_rule_minus_uniform": {
          "n": 32,
          "mean": 1.2134467755065852,
          "sd": 3.3366384270747496,
          "ci95": [
            0.01046033858493911,
            2.416433212428231
          ],
          "better": 8,
          "worse": 4
        },
        "gstar_rule_minus_uniform": {
          "n": 32,
          "mean": -9.284208213434855,
          "sd": 38.955466443904996,
          "ci95": [
            -23.32915158989144,
            4.76073516302173
          ],
          "better": 10,
          "worse": 14
        },
        "state_rule_minus_gstar_rule": {
          "n": 32,
          "mean": 10.49765498894144,
          "sd": 38.986600228548554,
          "ci95": [
            -3.55851331388185,
            24.55382329176473
          ],
          "better": 16,
          "worse": 7
        }
      }
    ]
  }
}
```

## state_rule.json

```json
{
  "form": "two_level",
  "feature": "sin_phase",
  "threshold": -0.6604594886302948,
  "feature2": "cos_phase",
  "children": {
    "low": {
      "leaf": "A45"
    },
    "high": {
      "threshold": 0.8876912295818329,
      "low": "A15",
      "high": "A0"
    }
  },
  "objective": 937.8285932212732,
  "leaves": 3
}
```


Gate B逐开局分支原始数据见science/t1o_v2/gate_b.json。

## gate_c.json

```json
{
  "fuel_weight": 20.0,
  "outcomes": {
    "uniform": {
      "success": 31,
      "unsafe": 0,
      "timeout": 1
    },
    "state_rule": {
      "success": 31,
      "unsafe": 0,
      "timeout": 1
    },
    "gstar_rule": {
      "success": 30,
      "unsafe": 1,
      "timeout": 1
    },
    "A0": {
      "success": 31,
      "unsafe": 0,
      "timeout": 1
    }
  },
  "mean_return": {
    "uniform": 51.96389045501348,
    "state_rule": 53.17733723052007,
    "gstar_rule": 42.67968224157862,
    "A0": 53.20661355495858
  },
  "state_rule_minus_uniform": {
    "n": 32,
    "mean": 1.2134467755065852,
    "sd": 3.3366384270747496,
    "ci95": [
      0.01046033858493911,
      2.416433212428231
    ],
    "better": 8,
    "worse": 4
  },
  "gstar_rule_minus_uniform": {
    "n": 32,
    "mean": -9.284208213434855,
    "sd": 38.955466443904996,
    "ci95": [
      -23.32915158989144,
      4.76073516302173
    ],
    "better": 10,
    "worse": 14
  },
  "state_rule_minus_gstar_rule": {
    "n": 32,
    "mean": 10.49765498894144,
    "sd": 38.986600228548554,
    "ci95": [
      -3.55851331388185,
      24.55382329176473
    ],
    "better": 16,
    "worse": 7
  },
  "gstar_planning_cpu_s_mean": 1.43359375,
  "pass": false
}
```

