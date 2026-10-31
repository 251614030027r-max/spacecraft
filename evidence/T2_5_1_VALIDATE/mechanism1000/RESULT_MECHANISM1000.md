# T2.5.1验证：mechanism1000

生成2026-10-27T00:12:25.881929+09:00；科学`34f877e933dc78740674fc82ee0e36f918c541f4`；执行书collab`02c697d`；37tests入场门本机通过（实际数见PYTEST_VERIFIED）。双EACC(e_mve)种子0/1各4000、各3异步worker；SAC复用T2.3及T2.4补评估共192，不重训。只验证并交付，不运行任何后续分支。
滚动30秒准确步数/更新/回合，真实与训练回报、成功率、critic/min-Q、反馈探针；外部观察器不新增科学forward/backward/RNG，入场验收见LOGGING_VERIFIED。独立隐藏Windows后台、全CPU/BLAS1。Q1/Q2分别统计冻结源未提供，标MISSING，不捏造。

## 1000步机制早报（非最终判定）
统计窗口严格在1000之后，原生探针经事件文件读取，各字段列明实际样本数和首末步数，不以目录存在或合成数据代替。w2_mean低于0.5、pi_gap_sigma_median高于0.25、grad_cos不大于0或字段缺失即列警报；按书只报告、不停止训练。
```json
{
  "jst": "2026-10-27T00:12:25.422568+09:00",
  "source": "Actual native probes parsed from external TensorBoard; rows strictly after1000; provisional,not final criteria",
  "readings": {
    "e_mve_s0": {
      "two_stage_share": {
        "mean": 0.6509791016578674,
        "n": 4,
        "first_step": 1012,
        "last_step": 1050,
        "missing": false
      },
      "decisive_share": {
        "mean": 0.17987721040844917,
        "n": 4,
        "first_step": 1012,
        "last_step": 1050,
        "missing": false
      },
      "stage2_rows": {
        "mean": 501.75,
        "n": 4,
        "first_step": 1012,
        "last_step": 1050,
        "missing": false
      },
      "w2_mean": {
        "mean": 0.9784456938505173,
        "n": 4,
        "first_step": 1012,
        "last_step": 1050,
        "missing": false
      },
      "pi_gap_sigma_median": {
        "mean": 0.014485621126368642,
        "n": 4,
        "first_step": 1012,
        "last_step": 1050,
        "missing": false
      },
      "stage2_entry_share": {
        "mean": 0.27774128690361977,
        "n": 4,
        "first_step": 1012,
        "last_step": 1050,
        "missing": false
      },
      "c_mean": {
        "mean": 0.2217896431684494,
        "n": 4,
        "first_step": 1012,
        "last_step": 1050,
        "missing": false
      },
      "increment_abs_mean": {
        "mean": 2.2952449917793274,
        "n": 4,
        "first_step": 1012,
        "last_step": 1050,
        "missing": false
      },
      "grad_cos": {
        "mean": 0.29501381516456604,
        "n": 1,
        "first_step": 1050,
        "last_step": 1050,
        "missing": false
      },
      "loss_share": {
        "mean": 0.2500614933669567,
        "n": 4,
        "first_step": 1012,
        "last_step": 1050,
        "missing": false
      },
      "w_mean": {
        "mean": 0.9434952288866043,
        "n": 4,
        "first_step": 1012,
        "last_step": 1050,
        "missing": false
      }
    },
    "e_mve_s1": {
      "two_stage_share": {
        "mean": 0.6825675794056484,
        "n": 7,
        "first_step": 1012,
        "last_step": 1087,
        "missing": false
      },
      "decisive_share": {
        "mean": 0.16031402881656373,
        "n": 7,
        "first_step": 1012,
        "last_step": 1087,
        "missing": false
      },
      "stage2_rows": {
        "mean": 620.8571428571429,
        "n": 7,
        "first_step": 1012,
        "last_step": 1087,
        "missing": false
      },
      "w2_mean": {
        "mean": 0.9436139634677342,
        "n": 7,
        "first_step": 1012,
        "last_step": 1087,
        "missing": false
      },
      "pi_gap_sigma_median": {
        "mean": 0.018082955053874424,
        "n": 7,
        "first_step": 1012,
        "last_step": 1087,
        "missing": false
      },
      "stage2_entry_share": {
        "mean": 0.2348336981875556,
        "n": 7,
        "first_step": 1012,
        "last_step": 1087,
        "missing": false
      },
      "c_mean": {
        "mean": 0.18665496366364615,
        "n": 7,
        "first_step": 1012,
        "last_step": 1087,
        "missing": false
      },
      "increment_abs_mean": {
        "mean": 2.3751265832356045,
        "n": 7,
        "first_step": 1012,
        "last_step": 1087,
        "missing": false
      },
      "grad_cos": {
        "mean": 0.5321550369262695,
        "n": 1,
        "first_step": 1050,
        "last_step": 1050,
        "missing": false
      },
      "loss_share": {
        "mean": 0.3117410327707018,
        "n": 7,
        "first_step": 1012,
        "last_step": 1087,
        "missing": false
      },
      "w_mean": {
        "mean": 0.9284475445747375,
        "n": 7,
        "first_step": 1012,
        "last_step": 1087,
        "missing": false
      }
    }
  },
  "immediate_alert_seeds": []
}
```

## T2.5停止前机制对照（未完成、无部署判定）
```json
{
  "e_mve_s0": {
    "two_stage_share": {
      "mean": 0.7355928705182186,
      "n": 86,
      "first_step": 1012,
      "last_step": 2075
    },
    "decisive_share": {
      "mean": 0.16358070959185445,
      "n": 86,
      "first_step": 1012,
      "last_step": 2075
    },
    "c_mean": {
      "mean": 0.13626885232190752,
      "n": 86,
      "first_step": 1012,
      "last_step": 2075
    },
    "grad_cos": {
      "mean": 0.5031790605613163,
      "n": 21,
      "first_step": 1050,
      "last_step": 2050
    },
    "loss_share": {
      "mean": 0.3917485262418902,
      "n": 86,
      "first_step": 1012,
      "last_step": 2075
    },
    "w_mean": {
      "mean": 0.9415694149427636,
      "n": 86,
      "first_step": 1012,
      "last_step": 2075
    },
    "increment_abs_mean": {
      "mean": 1.3077922310940056,
      "n": 86,
      "first_step": 1012,
      "last_step": 2075
    }
  },
  "e_mve_s1": {
    "two_stage_share": {
      "mean": 0.7231578214067809,
      "n": 71,
      "first_step": 1012,
      "last_step": 1887
    },
    "decisive_share": {
      "mean": 0.15524413268750822,
      "n": 71,
      "first_step": 1012,
      "last_step": 1887
    },
    "c_mean": {
      "mean": 0.16997436874769103,
      "n": 71,
      "first_step": 1012,
      "last_step": 1887
    },
    "grad_cos": {
      "mean": 0.4635874217047411,
      "n": 17,
      "first_step": 1050,
      "last_step": 1850
    },
    "loss_share": {
      "mean": 0.31651912314791075,
      "n": 71,
      "first_step": 1012,
      "last_step": 1887
    },
    "w_mean": {
      "mean": 0.9352038334792768,
      "n": 71,
      "first_step": 1012,
      "last_step": 1887
    },
    "increment_abs_mean": {
      "mean": 1.4981451101706063,
      "n": 71,
      "first_step": 1012,
      "last_step": 1887
    }
  }
}
```
旧T2.5已按用户停止，最后落盘2084/1888步，原目录保全；上表旧c值仅机制对照，不是最终性能失败结论。本轮只新learner训练；SAC与6例标签审计按书复用，不跑新的标签审计。

## 历史对照、异常与墙钟
T2.4固定science632b633：MVE种子0的4k回报53.792454、种子1为55.238621，均捕获23/24、unsafe0；种子0相对SAC54.476604差−0.684149。T2.5.1原生v251按执行书阈值53.79，事后不改。T2.3对照来源、逐文件SHA和模型SHA随附，旧SAC恢复事件字段若缺失应标MISSING，不当0。
两个训练finished原件包含实际wall_s和worker_order，滚动吞吐与ETA区分；并行墙钟不相加。训练真实技术崩溃同配置从零最多一次且保全旧attempt，不拼Replay/RNG/优化器；纯发布失败只补报告，不重跑科学。可选T2.4早期曲线要求另一SHA，为不影响本轮先暂缓，不混版本执行。
模型ZIP/Replay/TB二进制本机保留，Git发布432必要原始评估、6审计、训练JSON与外部滚动日志/事件解析和验收。没有自动长训或后续分支，等待上层复核与用户批准。
