# T2.7 主比较已齐：即时上层问题判断材料

生成2026-11-01T15:52:10.677932+09:00。用户要求已有资产立即推送，评估不停。本包固定快照含463例本轮已完整写出的评估、384例复用SAC及四条完成训练的完整元数据/回合日志；辅助SAC+引导评估继续。科学冻结2208a51不变。

## 已经可以明确的问题

**主比较原生判定预览：harmful。** PRIMARY_NATIVE_PREVIEW_NOT_FINAL由未修改的t27原生工具对完整主比较输入生成；不是全轮交付完成声明，根目录正式DECISION和四配对最终读出仍待原队列完成。主比较已齐，不必等待辅助评估才看到主比较问题。

```json
{
  "criteria": {
    "C0_complete": true,
    "M_mechanism": true,
    "E1_efficiency": false,
    "U_urgent": true,
    "E2_final": true,
    "S_safety": false
  },
  "values": {
    "paired_auc": {
      "0": -5.742509874095309,
      "1": 10.307117444964351
    },
    "paired_auc_urgent": {
      "0": -2.107338642351479,
      "1": 12.808658202250045
    },
    "urgent_capture_gain": {
      "0": 0,
      "1": 6
    },
    "paired_end": {
      "0": 0.005700593623631771,
      "1": 8.782993846688303
    },
    "unsafe": {
      "0": {
        "mve": 4,
        "sac": 0
      },
      "1": {
        "mve": 1,
        "sac": 0
      }
    },
    "mechanism": {
      "TRAIN_e_mve_s0": {
        "two_stage_share": 0.5571441291984794,
        "decisive_share": 0.1351503190503328,
        "c_mean": 0.13472728576526602,
        "grad_cos": 0.43350630070342394,
        "loss_share": 0.33436491440529065,
        "increment_abs_mean": 0.9614027807821377,
        "w_mean": 0.941841104218574,
        "w2_mean": 0.962529859602204,
        "pi_gap_sigma_median": 0.054854887688370164,
        "stage2_rows": 1031.390041493776,
        "stage2_entry_share": 0.24428371479155117,
        "stage1_cutoff_share": 0.0,
        "stage2_cutoff_share": 0.002553193985254608,
        "guided_share": 0.089
      },
      "TRAIN_e_mve_s1": {
        "two_stage_share": 0.5818707212523142,
        "decisive_share": 0.13171414122047265,
        "c_mean": 0.15100382821812175,
        "grad_cos": 0.3898650415424929,
        "loss_share": 0.4476164425620055,
        "increment_abs_mean": 1.4108384756131787,
        "w_mean": 0.9403145023895991,
        "w2_mean": 0.9585739794608469,
        "pi_gap_sigma_median": 0.05197367313244531,
        "stage2_rows": 1108.8174273858922,
        "stage2_entry_share": 0.22331401368765416,
        "stage1_cutoff_share": 0.0,
        "stage2_cutoff_share": 0.008658076108759849,
        "guided_share": 0.09375
      }
    }
  },
  "verdict": "harmful",
  "reported": {
    "late_s0": {
      "paired": {
        "3000": 3.2636210034547832,
        "3500": null,
        "4000": 0.005700593623631771
      },
      "capture": {
        "3000": [
          44,
          43
        ],
        "3500": null,
        "4000": [
          43,
          43
        ]
      },
      "unsafe": {
        "3000": [
          1,
          0
        ],
        "3500": null,
        "4000": [
          0,
          0
        ]
      },
      "return": {
        "mve": [
          41.56709431732819,
          null,
          38.9094808843579
        ],
        "sac": [
          38.303473313873404,
          null,
          38.90378029073427
        ]
      },
      "spread": {
        "mve": null,
        "sac": null
      }
    },
    "late_s1": {
      "paired": {
        "3000": 5.011754943742893,
        "3500": null,
        "4000": 8.782993846688303
      },
      "capture": {
        "3000": [
          42,
          41
        ],
        "3500": null,
        "4000": [
          42,
          41
        ]
      },
      "unsafe": {
        "3000": [
          1,
          0
        ],
        "3500": null,
        "4000": [
          0,
          0
        ]
      },
      "return": {
        "mve": [
          33.23019595625863,
          null,
          34.94189987473951
        ],
        "sac": [
          28.218441012515743,
          null,
          26.158906028051216
        ]
      },
      "spread": {
        "mve": null,
        "sac": null
      }
    },
    "urgent_bins_all_checkpoints": {
      "high": {
        "openings": 11,
        "capture_mve": 54,
        "capture_sac": 52,
        "paired_mean": 2.0048047949955397
      },
      "mid": {
        "openings": 7,
        "capture_mve": 43,
        "capture_sac": 43,
        "paired_mean": -0.804428822404375
      },
      "low": {
        "openings": 7,
        "capture_mve": 53,
        "capture_sac": 49,
        "paired_mean": 16.763520501515966
      }
    },
    "both_success": {
      "economy": {
        "checkpoints": 8,
        "delta_v": -0.046330633863404316,
        "time_s": -4.054118435911914
      },
      "urgent": {
        "checkpoints": 8,
        "delta_v": -0.022453352545002914,
        "time_s": -3.3161188510491924
      }
    }
  }
}
```

|种子|四点平均EACC+引导−SAC|上轮EACC−SAC|4000步差|四点unsafe EACC+引导/SAC|紧急捕获累计差|
|---|---:|---:|---:|---|---:|
|0|-5.742510|-8.212150|0.005701|4/0|0|
|1|10.307117|19.685933|8.782994|1/0|6|

效率未过：seed0四点差仍为负，两种子平均约+2.28也低于+5。安全未过：seed0四检查点累计4次unsafe对SAC的0，超过允许的+1；这些是评估计数，可能包含同一开局在不同检查点的重复违规，不冒称4个独立场景。4000步无unsafe不能抹掉此前记录。种子0略改善、种子1相对上轮回落，尚无稳定整体提升。

## 问题线索与上层审查建议

先核查seed0的unsafe原始开局、轨迹终止字段和检查点分布，并对照引导撤除前后的策略表现；原始评估JSON随附足够定位开局，本包没有新增回放或真值分支。不得据此直接把安全退化归因于某个机制。

SAC+引导辅助对照未齐，暂不能区分引导本身与标签增量。下一步只需完成既定辅助评估并看C_guide_only/C_labels/C_vs_eacc；不等于批准扩深、加门、改奖励或新训练。所有四条训练均4000完成、无新增训练。本轮即使4000点局部转好，仍严格保留四点与安全门槛，不挑点。

## 附件与缺项

C_primary提供完整主比较768个逐例JSON；SAC_guided_partial仅提供已完成的辅助结果，未完成标MISSING。B包含四条训练manifest、finished、episodes、model_stats、heartbeat与完整外部回合/滚动日志；模型ZIP不上传。PRIMARY_READOUT含逐开局配对、紧急/经济、分档、共同成功燃料时间。正式四配对最终读出与本轮最终交付：MISSING、仍在评估。SHA256SUMS固定本包，不能用于推断后续活动目录完整性。

本即时交付不停止评估、不改训练、不改变预登记门槛。正式最终报告将在全部辅助评估及四种配对完整后另外交付。
