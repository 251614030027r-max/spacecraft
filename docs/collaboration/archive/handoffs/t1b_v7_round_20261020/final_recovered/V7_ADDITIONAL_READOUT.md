# V7补充读出

反事实QP必须每控制周期求解，配置stride值、事实误差分布与按经济/急迫划分的标签方向见MPC_FACTUAL_ERROR_AND_LABELS.json。四环节链仅在52维float32观测完全相同且实际存在采信配对时填标签；缺失不以历史模型标签或别处近邻替代。Q排序、Actor确定性选择与完整实际回报见FOUR_LINK_CHAIN.csv。292140逐臂逐检查点单列CASE_292140.csv；去掉该例的7例均值仅在7例齐时给出：

{
  "e_sac_2000": {
    "completed_without_292140": 7,
    "mean_without_292140": 60.30496646017032,
    "captures_without_292140": 7,
    "case_292140_complete": true,
    "partial_mean_not_reported": false
  },
  "e_sac_5000": {
    "completed_without_292140": 7,
    "mean_without_292140": 60.30496646017032,
    "captures_without_292140": 7,
    "case_292140_complete": true,
    "partial_mean_not_reported": false
  },
  "e_sac_10000": {
    "completed_without_292140": 7,
    "mean_without_292140": 60.30496646017032,
    "captures_without_292140": 7,
    "case_292140_complete": true,
    "partial_mean_not_reported": false
  },
  "e_tcpf_2000": {
    "completed_without_292140": 7,
    "mean_without_292140": 60.526466704269765,
    "captures_without_292140": 7,
    "case_292140_complete": true,
    "partial_mean_not_reported": false
  },
  "e_tcpf_5000": {
    "completed_without_292140": 7,
    "mean_without_292140": 60.26289689905352,
    "captures_without_292140": 7,
    "case_292140_complete": true,
    "partial_mean_not_reported": false
  },
  "e_tcpf_10000": {
    "completed_without_292140": 7,
    "mean_without_292140": 60.526466704269765,
    "captures_without_292140": 7,
    "case_292140_complete": true,
    "partial_mean_not_reported": false
  }
}

293100–293131是事先登记的独立开发块，10k后在SIDE队列执行并报告；只报告，不参与M/R/种子1或现场修改。292500–292531独立确认块及B离线试验台另见SIDE_BENCH_RESULT.md。
