# T2.2 种子1确定性评估：2k中期
 
科学冻结 `fa54ed1391602386063290b7d34222313a4a427e`，原执行书collab1745c28；用户2026-10-23最新指令取消重跑seed0，只完成seed1评估并推送，不再训练或自动进入P/K/L。生成时间 2026-10-23T13:44:26.196445+09:00。
本报告只统计种子1、两臂相同24开局294000–294023；两臂均真实完成3000次决策。原种子0模型和记录全部本机保全，退出本轮统计，不与重跑或种子1混合。
状态：完整24例的0k/2k配对趋势；1k/3k最终读出仍在继续。不是完整四点曲线或原双种子机械判定。

## 四点部署曲线（确定性；真实0k）
| 臂 | 步数 | n | 均值 | 捕获 | unsafe |
|---|---:|---:|---:|---:|---:|
| e_sac | 0 | 24 | -63.086319 | 9/24 | 0 |
| e_sac | 2000 | 24 | 51.136123 | 23/24 | 0 |
| e_mve | 0 | 24 | -63.086319 | 9/24 | 0 |
| e_mve | 2000 | 24 | 15.327238 | 19/24 | 0 |

## 学习与反馈配对读数
```json
{
  "learning_gain_from_actual_0k": {
    "e_sac": 114.22244156189448,
    "e_mve": 78.41355650519937
  },
  "paired": {
    "0": {
      "n": 24,
      "mean_difference": 0.0,
      "trim_largest2": 0.0,
      "urgent_mean": 0.0,
      "economy_mean": 0.0,
      "positive": 0,
      "negative": 0,
      "rows": [
        {
          "opening": 294000,
          "urgent": true,
          "difference": 0.0,
          "sac_event": "timeout",
          "mve_event": "timeout"
        },
        {
          "opening": 294001,
          "urgent": false,
          "difference": 0.0,
          "sac_event": "success",
          "mve_event": "success"
        },
        {
          "opening": 294002,
          "urgent": false,
          "difference": 0.0,
          "sac_event": "success",
          "mve_event": "success"
        },
        {
          "opening": 294003,
          "urgent": true,
          "difference": 0.0,
          "sac_event": "timeout",
          "mve_event": "timeout"
        },
        {
          "opening": 294004,
          "urgent": true,
          "difference": 0.0,
          "sac_event": "timeout",
          "mve_event": "timeout"
        },
        {
          "opening": 294005,
          "urgent": false,
          "difference": 0.0,
          "sac_event": "success",
          "mve_event": "success"
        },
        {
          "opening": 294006,
          "urgent": true,
          "difference": 0.0,
          "sac_event": "timeout",
          "mve_event": "timeout"
        },
        {
          "opening": 294007,
          "urgent": false,
          "difference": 0.0,
          "sac_event": "timeout",
          "mve_event": "timeout"
        },
        {
          "opening": 294008,
          "urgent": false,
          "difference": 0.0,
          "sac_event": "success",
          "mve_event": "success"
        },
        {
          "opening": 294009,
          "urgent": false,
          "difference": 0.0,
          "sac_event": "success",
          "mve_event": "success"
        },
        {
          "opening": 294010,
          "urgent": false,
          "difference": 0.0,
          "sac_event": "timeout",
          "mve_event": "timeout"
        },
        {
          "opening": 294011,
          "urgent": true,
          "difference": 0.0,
          "sac_event": "timeout",
          "mve_event": "timeout"
        },
        {
          "opening": 294012,
          "urgent": true,
          "difference": 0.0,
          "sac_event": "timeout",
          "mve_event": "timeout"
        },
        {
          "opening": 294013,
          "urgent": true,
          "difference": 0.0,
          "sac_event": "success",
          "mve_event": "success"
        },
        {
          "opening": 294014,
          "urgent": true,
          "difference": 0.0,
          "sac_event": "timeout",
          "mve_event": "timeout"
        },
        {
          "opening": 294015,
          "urgent": true,
          "difference": 0.0,
          "sac_event": "timeout",
          "mve_event": "timeout"
        },
        {
          "opening": 294016,
          "urgent": true,
          "difference": 0.0,
          "sac_event": "timeout",
          "mve_event": "timeout"
        },
        {
          "opening": 294017,
          "urgent": true,
          "difference": 0.0,
          "sac_event": "timeout",
          "mve_event": "timeout"
        },
        {
          "opening": 294018,
          "urgent": true,
          "difference": 0.0,
          "sac_event": "success",
          "mve_event": "success"
        },
        {
          "opening": 294019,
          "urgent": true,
          "difference": 0.0,
          "sac_event": "timeout",
          "mve_event": "timeout"
        },
        {
          "opening": 294020,
          "urgent": false,
          "difference": 0.0,
          "sac_event": "success",
          "mve_event": "success"
        },
        {
          "opening": 294021,
          "urgent": false,
          "difference": 0.0,
          "sac_event": "success",
          "mve_event": "success"
        },
        {
          "opening": 294022,
          "urgent": true,
          "difference": 0.0,
          "sac_event": "timeout",
          "mve_event": "timeout"
        },
        {
          "opening": 294023,
          "urgent": true,
          "difference": 0.0,
          "sac_event": "timeout",
          "mve_event": "timeout"
        }
      ]
    },
    "2000": {
      "n": 24,
      "mean_difference": -35.808885056695125,
      "trim_largest2": -48.919308428190014,
      "urgent_mean": -57.752830811674826,
      "economy_mean": 0.7643578682710408,
      "positive": 7,
      "negative": 17,
      "rows": [
        {
          "opening": 294000,
          "urgent": true,
          "difference": -219.71704440913058,
          "sac_event": "success",
          "mve_event": "timeout"
        },
        {
          "opening": 294001,
          "urgent": false,
          "difference": 2.2450419616170194,
          "sac_event": "success",
          "mve_event": "success"
        },
        {
          "opening": 294002,
          "urgent": false,
          "difference": 12.770981191415295,
          "sac_event": "success",
          "mve_event": "success"
        },
        {
          "opening": 294003,
          "urgent": true,
          "difference": -206.2517322907965,
          "sac_event": "success",
          "mve_event": "timeout"
        },
        {
          "opening": 294004,
          "urgent": true,
          "difference": -0.4874423926640219,
          "sac_event": "success",
          "mve_event": "success"
        },
        {
          "opening": 294005,
          "urgent": false,
          "difference": -7.791039731485377,
          "sac_event": "success",
          "mve_event": "success"
        },
        {
          "opening": 294006,
          "urgent": true,
          "difference": 0.3995004216344853,
          "sac_event": "success",
          "mve_event": "success"
        },
        {
          "opening": 294007,
          "urgent": false,
          "difference": -0.9513966351441709,
          "sac_event": "success",
          "mve_event": "success"
        },
        {
          "opening": 294008,
          "urgent": false,
          "difference": -0.5707590451521511,
          "sac_event": "success",
          "mve_event": "success"
        },
        {
          "opening": 294009,
          "urgent": false,
          "difference": -0.4759611733596074,
          "sac_event": "success",
          "mve_event": "success"
        },
        {
          "opening": 294010,
          "urgent": false,
          "difference": -1.0276603968196412,
          "sac_event": "success",
          "mve_event": "success"
        },
        {
          "opening": 294011,
          "urgent": true,
          "difference": -217.73302212140715,
          "sac_event": "success",
          "mve_event": "timeout"
        },
        {
          "opening": 294012,
          "urgent": true,
          "difference": -0.9476440652274505,
          "sac_event": "success",
          "mve_event": "success"
        },
        {
          "opening": 294013,
          "urgent": true,
          "difference": 0.2832288685952733,
          "sac_event": "success",
          "mve_event": "success"
        },
        {
          "opening": 294014,
          "urgent": true,
          "difference": -0.5748747561362393,
          "sac_event": "success",
          "mve_event": "success"
        },
        {
          "opening": 294015,
          "urgent": true,
          "difference": -0.09747816367243445,
          "sac_event": "success",
          "mve_event": "success"
        },
        {
          "opening": 294016,
          "urgent": true,
          "difference": -215.82189415234842,
          "sac_event": "success",
          "mve_event": "timeout"
        },
        {
          "opening": 294017,
          "urgent": true,
          "difference": 204.04056286808193,
          "sac_event": "timeout",
          "mve_event": "success"
        },
        {
          "opening": 294018,
          "urgent": true,
          "difference": 1.0727438192634793,
          "sac_event": "success",
          "mve_event": "success"
        },
        {
          "opening": 294019,
          "urgent": true,
          "difference": -0.5906431083403589,
          "sac_event": "success",
          "mve_event": "success"
        },
        {
          "opening": 294020,
          "urgent": false,
          "difference": -2.2094785739803555,
          "sac_event": "success",
          "mve_event": "success"
        },
        {
          "opening": 294021,
          "urgent": false,
          "difference": 4.889493217348356,
          "sac_event": "success",
          "mve_event": "success"
        },
        {
          "opening": 294022,
          "urgent": true,
          "difference": -0.09442993850757375,
          "sac_event": "success",
          "mve_event": "success"
        },
        {
          "opening": 294023,
          "urgent": true,
          "difference": -209.77229275446683,
          "sac_event": "success",
          "mve_event": "timeout"
        }
      ]
    }
  },
  "single_seed_reference_readings": {
    "not_original_dual_seed_DECISION_1": true,
    "no_branch_or_new_training": true,
    "reference_final_step": 2000,
    "MVE_vs_previous_23_421727": false,
    "paired_2k_positive": false,
    "paired_final_positive": false,
    "paired_final_at_least10": false,
    "urgent_pair_positive": false,
    "trim_largest2_positive": false,
    "capture_not_worse_than_SAC_minus1": false,
    "unsafe_not_worse_than_SAC_plus1": true
  }
}
```
所有负差、失败和unsafe均保留。配对以同版本同种子同开局为准；去最大两例和紧急任务读数用来检查收益是否集中，不据训练探索回合统计认定部署效果。原双种子C0-C5正式判定不再适用，不伪造DECISION_1或自动选分支。

## 固定策略和历史参照
基线筛选尚未完整，暂不把部分筛选结果当24例总体。最终交付补齐15策略全部360例。
上一轮3107420、24例、seed0：SAC3000均12.8969787684、19捕获/1unsafe；MVE均23.4217272911、20捕获/0unsafe；配对+10.5247485226、去最大两例−2.3966303800。固定证据science26fd421、补充28c96e0。跨版本历史仅供读数，不当本轮耦合因果对照。

## 机制、来源及限制
```json
{
  "e_sac": {
    "probe_n": 225,
    "lambda_eff": {
      "n": 225,
      "median": 0.0,
      "min": 0.0,
      "max": 0.0
    },
    "g_real": {
      "n": 225,
      "median": 0.0,
      "min": 0.0,
      "max": 0.0
    },
    "g_model_unscaled": {
      "n": 225,
      "median": 0.0,
      "min": 0.0,
      "max": 0.0
    },
    "weighted_model_gradient_ratio": {
      "median": null,
      "max": null
    },
    "note": "Mechanism probes alone do not establish deployment benefit"
  },
  "e_mve": {
    "probe_n": 225,
    "lambda_eff": {
      "n": 225,
      "median": 0.11226355333577122,
      "min": 0.0,
      "max": 0.6680984359134885
    },
    "g_real": {
      "n": 225,
      "median": 747.092529296875,
      "min": 0.0,
      "max": 2118.842529296875
    },
    "g_model_unscaled": {
      "n": 225,
      "median": 3080.409423828125,
      "min": 0.0,
      "max": 7836.06494140625
    },
    "weighted_model_gradient_ratio": {
      "median": 0.5,
      "max": 0.5000000000000001
    },
    "note": "Mechanism probes alone do not establish deployment benefit"
  }
}
```
完整manifest、finished、模型探针、replay门计数、必要原始JSON、源模型SHA和逐文件SHA随交付。模型ZIP及Replay不入Git。总体训练入口事件一致率仍MISSING，只能报告明确事件子集，不能当总体率；非零模型梯度不是独立收益证据。

## 中断、范围与运行状态
用户确认更新Codex导致应用绑定原进程树退出。种子1在此前已正常完整完成；未被重跑或续拼。原种子0只有旧检查点，用户已取消从零恢复并排除出本轮读数。受控后台迁移及取消记录保全，额外墙钟不隐藏。当前评估由Windows计划任务独立托管、无Codex祖先，科学代码/参数/任务/奖励/开局/门槛均未改。
只读加载时将不使用的Replay容量100000改为4；8个冻结模型的权重、动作、Actor分布参数和Q1/Q2逐位一致，模型文件SHA未改，验证表随附。仅改变空Replay分配，无学习更新或科学模块修改。
此项每模型减少约108.7MiB的Replay数组分配；核验进程的物理RSS未观察到同量级下降，不将其当作物理内存增加或科研收益。扩并行仍受实际提交余量和物理内存临界保护约束。
本轮仅单种子成效证据；是否有增长及同任务反馈收益以完整配对数值为准，不用少量样本替代总体。此次交付后停止，无自动训练。

原生曲线见SEED1_READOUT_INTERIM_2000.json；逐例与任务分层见SEED1_ANALYSIS_INTERIM_2000.json、CURVES.csv、PAIRED.csv、EPISODES.csv。实际覆盖及配置见RUNTIME_SNAPSHOT、SEED1_ONLY_SCOPE，哈希见SHA256SUMS。
