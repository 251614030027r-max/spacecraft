# T2 2000步固定8开局趋势读数

Fixed first8 of registered24, exploratory trends only; training continues unchanged

冻结3107420；仅294000–294007，未替换开局、未停止原3000步训练。相同标签和B_eval目录，完成JSON直接复用。
中性动作是近似0k参照，正式0k→3000曲线仍使用真实0k检查点；本报告另列已存在的真实0k，缺项不补造。
```json
{
  "jst": "2026-10-22T13:15:03.447386+09:00",
  "scientific_SHA": "31074206045cf62dba3d560e4442db4b00eea05a",
  "openings": [
    294000,
    294001,
    294002,
    294003,
    294004,
    294005,
    294006,
    294007
  ],
  "checkpoint_SHA256": {
    "e_sac": "581d2a9c64a02ef8275cfeff0ba12a7dee31e93128edea5a06dda8e6aa35b4e6",
    "e_mve": "10f1b9220806717e59ac646ff38ecbad3b74090906c0fd74d719c117e4ea38fd"
  },
  "arms": {
    "e_sac": {
      "2000": {
        "n": 8,
        "mean_return": -51.7456047763189,
        "capture": 4,
        "unsafe": 0
      },
      "neutral_c_0_0_proxy": {
        "n": 8,
        "mean_return": -74.24689679683215,
        "capture": 3,
        "unsafe": 0
      },
      "rally": {
        "n": 8,
        "mean_return": -43.437364289650695,
        "capture": 4,
        "unsafe": 0
      },
      "actual0k_available_only": {
        "n": 8,
        "mean_return": -80.15084891409073,
        "capture": 3,
        "unsafe": 0
      },
      "missing_or_error": [],
      "gain_vs_neutral_proxy": 22.501292020513247,
      "trend_gain20_capture_not_lower": true
    },
    "e_mve": {
      "2000": {
        "n": 8,
        "mean_return": -2.3313362777763267,
        "capture": 6,
        "unsafe": 0
      },
      "neutral_c_0_0_proxy": {
        "n": 8,
        "mean_return": -74.24689679683215,
        "capture": 3,
        "unsafe": 0
      },
      "rally": {
        "n": 8,
        "mean_return": -43.437364289650695,
        "capture": 4,
        "unsafe": 0
      },
      "actual0k_available_only": {
        "n": 0,
        "mean_return": null,
        "capture": 0,
        "unsafe": 0
      },
      "missing_or_error": [],
      "gain_vs_neutral_proxy": 71.91556051905582,
      "trend_gain20_capture_not_lower": true
    }
  },
  "scope": "Fixed first8 of registered24, exploratory trends only; training continues unchanged",
  "neutral_anchor": "c_0_0 is an approximate0k proxy, not substituted for actual0k in final curves",
  "paired_mve_minus_sac": {
    "n": 8,
    "mean": 49.414268498542576,
    "trend_gain5": true
  }
}
```

逐例对比见INTERIM_2000_8.csv；原始16例JSON及源检查点SHA随交付。若缺项或ERROR，不能用剩余成功例冒充完整8例趋势。训练噪声回合统计不用于部署成效判定；本小样本不能代替24例最终B_READOUT。
资源限制下按内存安全动态并行，最多每臂2工人/总4；每例后回收，训练PID/优化器/环境和科学参数不变。
