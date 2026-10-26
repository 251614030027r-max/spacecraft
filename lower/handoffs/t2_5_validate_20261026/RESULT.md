# T2.5 用户停止记录

停止时刻：2026-10-26T20:55:47.310952+09:00；最后约30秒落盘步数：{"e_mve_s0": 2084, "e_mve_s1": 1888}。**用户决定停止，T2.5.1优先。** 科学84ee9c1045f32bbbacfef6cd220fdad55c5765c8；原目录、检查点和日志全保留，不做部署评估，不执行任何判定，不将未完成等同最终机制失败。

步数是最后持久化计数，强制停止时的内存瞬时计数不可恢复，不编造精确终止步数。episodes/model_stats/heartbeat为冻结代码最后检查点落盘，可能早于停止；外部滚动与事件解析提供之后可核验的机制探针，逐字段样本数/窗口见STOP_MECHANISM_READOUT。部分训练不拼接到新实验。

## 停止前实际机制探针（严格1000之后）
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
上述读数仅作为低c对照；不是配对性能收益、因果证明或正式v25判定。已完成6例标签审计原件随附，未重跑。模型/Replay/TB二进制本机保留，不入Git。新T2.5.1有独立目录和冻结SHA。
