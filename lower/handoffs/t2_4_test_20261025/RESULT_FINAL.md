# T2.4 最终事实交付

生成 2026-10-26T12:44:23.713366+09:00；科学 `e13c8b1d52dafe7b90fac3eaaa3651294d253b0f`；执行书collab `57e4ba0`。本机26项测试通过，外部日志适配器单独验收；冻结科学文件不改，任务/奖励/候选K8/β0.5/执行层/步数/3worker不改。仅两组MVE seed0/1各4000，SAC复用T2.3，种子1SAC3k补评估；不跑后续分支。

## 日志和来源
用户明确授权外部日志监视；脚本随附，滚动JSON约30秒，TensorBoard逐回合真实/训练回报、成功率、原生critic损失/min-Q及模型反馈探针。日志验收含相同数据原生更新权重逐位一致、关闭日志不改变RNG状态、真实48决策异步冒烟事件生成和增长；机械数据不参与科研结果。异步调度不承诺跨次训练轨迹逐位相同。Q1/Q2分别值冻结探针未提供，使用实际q_pi_mean（min-Q），不额外前向、不伪造。
SAC逐文件SHA映射和旧模型SHA随附；复用SAC旧结果可能缺window_recoveries字段，明确计缺失，不能当恢复次数0。

## 判定与末段读数（用户主动缩减评估范围）
用户最新授权仅评估3k/4k：以下为范围修订后的描述性读数，不是原生t24正式判定：
```json
{
  "verdict": "scope_revised_descriptive_only",
  "formal_t24_verdict": "NOT_APPLICABLE: latest user cancelled0/1k/2k MVE evaluations, so original C0/full-curve C2 cannot be computed",
  "scope": "two MVE seeds3k/4k,24openings each96new;192historicalSAC retained;288total",
  "training_effect": "3k-to4k change is late-training change, not a measured0k-to4k deployment learning gain",
  "late_training": {
    "e_mve_s0": {
      "return_4k_minus_3k": 17.884596508761796,
      "capture_3k": 21,
      "capture_4k": 23,
      "unsafe_3k": 0,
      "unsafe_4k": 0
    },
    "e_mve_s1": {
      "return_4k_minus_3k": -0.5058662765663442,
      "capture_3k": 23,
      "capture_4k": 23,
      "unsafe_3k": 0,
      "unsafe_4k": 0
    }
  },
  "paired_vs_SAC": {
    "s0/step_3000": {
      "n": 24,
      "mean": -18.84797601697529,
      "without_top2": -21.071877734818717,
      "urgent": -29.756379330439174,
      "economy": -0.667303827868821,
      "positive": 9,
      "negative": 15,
      "capture_mve": 21,
      "capture_sac": 23,
      "unsafe_mve": 0,
      "unsafe_sac": 0
    },
    "s0/step_4000": {
      "n": 24,
      "mean": -0.6841493964997483,
      "without_top2": -10.330988504055172,
      "urgent": -2.7786318643226045,
      "economy": 2.8066547165383446,
      "positive": 14,
      "negative": 10,
      "capture_mve": 23,
      "capture_sac": 23,
      "unsafe_mve": 0,
      "unsafe_sac": 0
    },
    "s1/step_3000": {
      "n": 24,
      "mean": 10.81059733721729,
      "without_top2": 1.9131321051404095,
      "urgent": 14.059940496799369,
      "economy": 5.395025404580491,
      "positive": 18,
      "negative": 6,
      "capture_mve": 23,
      "capture_sac": 22,
      "unsafe_mve": 0,
      "unsafe_sac": 0
    }
  },
  "missing_control": "SACseed1step4000 absent; do not fabricate final paired effect forseed1",
  "mechanism_claim": "Audit and feedback probes describe mechanism behavior; performance evidence requires matched deployment comparisons and does not prove causal benefit alone",
  "no_branch": true
}
```
MVE两seed各3k/4k×24共96新评估，历史SAC192保留共288。按用户要求取消MVE0/1k/2k，原完整曲线C0/C2和正式t24总判定不适用；仅报告末段变化与匹配部署收益。seed1缺SAC4k，不伪造末点配对。训练滚动回报是带探索的训练统计，不替代部署增益。

|臂/种子|步数|任务|n|均回报|捕获|unsafe|
|---|---:|---|---:|---:|---:|---:|
|e_mve_s0|3000|all|24|35.907857653729494|21|0|
|e_mve_s0|3000|urgent|15|30.23402374844421|12|0|
|e_mve_s0|3000|economy|9|45.36424749587164|9|0|
|e_mve_s0|4000|all|24|53.79245416249129|23|0|
|e_mve_s0|4000|urgent|15|57.48838494565312|14|0|
|e_mve_s0|4000|economy|9|47.632569523888236|9|0|
|e_mve_s1|3000|all|24|55.74448726746232|23|0|
|e_mve_s1|3000|urgent|15|60.665838936972335|14|0|
|e_mve_s1|3000|economy|9|47.54223448494561|9|0|
|e_mve_s1|4000|all|24|55.238620990895974|23|0|
|e_mve_s1|4000|urgent|15|60.58109720106729|14|0|
|e_mve_s1|4000|economy|9|46.33449397394378|9|0|
|e_sac_s0|0|all|24|-14.590111878851593|15|0|
|e_sac_s0|0|urgent|15|-49.11830536798308|6|0|
|e_sac_s0|0|economy|9|42.95687726970089|9|0|
|e_sac_s0|1000|all|24|36.48854557639367|21|0|
|e_sac_s0|1000|urgent|15|31.311230045404276|12|0|
|e_sac_s0|1000|economy|9|45.11740479470933|9|0|
|e_sac_s0|2000|all|24|44.66488588401395|22|0|
|e_sac_s0|2000|urgent|15|44.8866178570459|13|0|
|e_sac_s0|2000|economy|9|44.295332595627364|9|0|
|e_sac_s0|3000|all|24|54.755833670704796|23|0|
|e_sac_s0|3000|urgent|15|59.990403078883396|14|0|
|e_sac_s0|3000|economy|9|46.03155132374046|9|0|
|e_sac_s0|4000|all|24|54.47660355899104|23|0|
|e_sac_s0|4000|urgent|15|60.26701680997573|14|0|
|e_sac_s0|4000|economy|9|44.825914807349896|9|0|
|e_sac_s1|1000|all|24|26.432544287573393|20|0|
|e_sac_s1|1000|urgent|15|17.205131828890696|11|0|
|e_sac_s1|1000|economy|9|41.811565052044536|9|0|
|e_sac_s1|2000|all|24|43.5895344494727|22|0|
|e_sac_s1|2000|urgent|15|44.16671838737794|13|0|
|e_sac_s1|2000|economy|9|42.62756121963066|9|0|
|e_sac_s1|3000|all|24|44.93388993024502|22|0|
|e_sac_s1|3000|urgent|15|46.60589844017298|13|0|
|e_sac_s1|3000|economy|9|42.147209080365116|9|0|

逐开局同任务配对、剔除最大两例、紧急/经济完整原生读数见B_READOUT.json。
```json
{
  "s0/step_3000": {
    "n": 24,
    "mean": -18.84797601697529,
    "without_top2": -21.071877734818717,
    "urgent": -29.756379330439174,
    "economy": -0.667303827868821,
    "positive": 9,
    "negative": 15,
    "capture_mve": 21,
    "capture_sac": 23,
    "unsafe_mve": 0,
    "unsafe_sac": 0
  },
  "s0/step_4000": {
    "n": 24,
    "mean": -0.6841493964997483,
    "without_top2": -10.330988504055172,
    "urgent": -2.7786318643226045,
    "economy": 2.8066547165383446,
    "positive": 14,
    "negative": 10,
    "capture_mve": 23,
    "capture_sac": 23,
    "unsafe_mve": 0,
    "unsafe_sac": 0
  },
  "s1/step_3000": {
    "n": 24,
    "mean": 10.81059733721729,
    "without_top2": 1.9131321051404095,
    "urgent": 14.059940496799369,
    "economy": 5.395025404580491,
    "positive": 18,
    "negative": 6,
    "capture_mve": 23,
    "capture_sac": 22,
    "unsafe_mve": 0,
    "unsafe_sac": 0
  }
}
```

## A机制与历史对照
本轮六开局2910000–2910005审计，已执行意图本阶段进/未进入口两组和raw/anchored误差在MECHANISM_READOUT.json（比较缺终止匹配的候选不强填误差）。原生A1_LABEL_AUDIT完整原件随附；若deadline尚未完成则在快照明确缺项。
T2.3固定对照science74945c71d5cdf883ab26aea812deeeb8d98f5e58：seed0 MVE4k42.364999、22/24、unsafe1；SAC4k54.476604、23/24、unsafe0；配对−12.111604，紧急−19.136867，去最大两例−22.569731；原生coupling_failure。历史门槛42.365只作参考；范围变更后原生完整曲线判定不适用。

## 机制指标与计算
MECHANISM_READOUT包含每1000步探针的entry_bias_now/entry_rel_mean/loss_share/grad_cos/Q/龄期分布及window恢复逐事件原件，缺失明确保留；候选误差估计尚未证明覆盖真误差，不宣称理论界成立。实际训练墙钟、worker_order、各run吞吐见finished；并行墙钟不相加。00点启动、09点目标按实测核查，deadline进度不能冒充最终判定。

## 交付边界
只有事实与原生判定，不自动调参或追加分支；等待研究层与用户下一授权。模型ZIP/Replay和TensorBoard二进制事件只本机保留并记摘要/来源，Git提供必要288原始评估（MVE96+历史SAC192）、6审计、原生读出/判定、训练JSON、外部日志源及验收证据。网络失败只补发布不重跑科学。
