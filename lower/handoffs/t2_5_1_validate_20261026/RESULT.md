# T2.5.1验证：final

生成2026-10-27T11:24:53.795750+09:00；科学`34f877e933dc78740674fc82ee0e36f918c541f4`；执行书collab`02c697d`；37tests入场门本机通过（实际数见PYTEST_VERIFIED）。双EACC(e_mve)种子0/1各4000、各3异步worker；SAC复用T2.3及T2.4补评估共192，不重训。只验证并交付，不运行任何后续分支。
滚动30秒准确步数/更新/回合，真实与训练回报、成功率、critic/min-Q、反馈探针；外部观察器不新增科学forward/backward/RNG，入场验收见LOGGING_VERIFIED。独立隐藏Windows后台、全CPU/BLAS1。Q1/Q2分别统计冻结源未提供，标MISSING，不捏造。

## 原生v251判定
```json
{
  "criteria": {
    "C0_complete": true,
    "C1_beyond_t23": true,
    "C2_coupling_s0": true,
    "C2_coupling_s1": false,
    "C3_end_s0": true,
    "C3_end_s1": true,
    "C4_converged": true,
    "C5_safety": true,
    "M_mechanism": true,
    "E1_s0": true,
    "E1_s1": false,
    "E2_final": true
  },
  "values": {
    "eacc_s0_4000": 54.57618756322554,
    "sac_s0_4000": 54.47660355899104,
    "paired_auc": {
      "0": 3.9426856623783015,
      "1": -0.8179112657257295
    },
    "paired_end": {
      "0": 0.0995840042344985,
      "1": -0.6404434773502162
    },
    "paired_end_urgent": {
      "0": -0.05360721888475458,
      "1": -2.176101263211639
    },
    "last_step_change_s0": {
      "mve": 2.177988841737161,
      "sac": -0.27923011171375833
    },
    "mechanism": {
      "TRAIN_e_mve_s0": {
        "two_stage_share": 0.5795348313534299,
        "decisive_share": 0.16367277373169467,
        "c_mean": 0.2340340716339246,
        "grad_cos": 0.333026313909986,
        "loss_share": 0.34619800544819745,
        "increment_abs_mean": 1.400289553824302,
        "w_mean": 0.9337378520193931,
        "w2_mean": 0.9596583116103999,
        "pi_gap_sigma_median": 0.026711353904633355,
        "stage2_rows": 898.8091286307053,
        "stage2_entry_share": 0.28258302702696
      },
      "TRAIN_e_mve_s1": {
        "two_stage_share": 0.6200291296772504,
        "decisive_share": 0.15622471779955868,
        "c_mean": 0.17367077105643838,
        "grad_cos": 0.33718396849991356,
        "loss_share": 0.32739992382115396,
        "increment_abs_mean": 1.2003580725044631,
        "w_mean": 0.933843589669936,
        "w2_mean": 0.942065725682682,
        "pi_gap_sigma_median": 0.03786112013354094,
        "stage2_rows": 998.7261410788382,
        "stage2_entry_share": 0.2509878826215554
      }
    }
  },
  "verdict": "mixed"
}
```
pass仅本轮开发验证，不是论文正式证据或收敛定理。SAC种子1只到3k，MVE种子1的4k无匹配SAC，不捏造对照。

|臂|步数|任务|n|回报|捕获|unsafe|
|---|---:|---|---:|---:|---:|---:|
|e_mve_s0|0|all|24|-14.590111878851593|15|0|
|e_mve_s0|0|urgent|15|-49.11830536798308|6|0|
|e_mve_s0|0|economy|9|42.95687726970089|9|0|
|e_mve_s0|1000|all|24|35.85875484645555|21|0|
|e_mve_s0|1000|urgent|15|29.588558463385585|12|0|
|e_mve_s0|1000|economy|9|46.30908215157215|9|0|
|e_mve_s0|2000|all|24|63.32347020844721|24|0|
|e_mve_s0|2000|urgent|15|73.62713267551575|15|0|
|e_mve_s0|2000|economy|9|46.15069942999963|9|0|
|e_mve_s0|3000|all|24|52.39819872148838|23|1|
|e_mve_s0|3000|urgent|15|57.37537238489953|14|1|
|e_mve_s0|3000|economy|9|44.10290928246978|9|0|
|e_mve_s0|4000|all|24|54.57618756322554|23|0|
|e_mve_s0|4000|urgent|15|60.21340959109096|14|0|
|e_mve_s0|4000|economy|9|45.18081751678315|9|0|
|e_mve_s1|0|all|24|-54.78874125776034|10|0|
|e_mve_s1|0|urgent|15|-96.97373714303497|2|0|
|e_mve_s1|0|economy|9|15.519585217697372|8|0|
|e_mve_s1|1000|all|24|25.396390513856648|20|0|
|e_mve_s1|1000|urgent|15|13.331198901467594|11|0|
|e_mve_s1|1000|economy|9|45.50504320117172|9|0|
|e_mve_s1|2000|all|24|42.812397903362466|22|0|
|e_mve_s1|2000|urgent|15|44.39350938787993|13|0|
|e_mve_s1|2000|economy|9|40.17721209583337|9|0|
|e_mve_s1|3000|all|24|44.29344645289481|22|0|
|e_mve_s1|3000|urgent|15|44.429797176961344|13|0|
|e_mve_s1|3000|economy|9|44.06619524611728|9|0|
|e_mve_s1|4000|all|24|53.03017341449411|23|0|
|e_mve_s1|4000|urgent|15|58.382953543427405|14|0|
|e_mve_s1|4000|economy|9|44.10887319960528|9|0|
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

## 同开局配对（EACC−SAC）
```json
{
  "s0/step_0": {
    "n": 24,
    "mean": 0.0,
    "without_top2": 0.0,
    "urgent": 0.0,
    "economy": 0.0,
    "positive": 0,
    "negative": 0,
    "capture_mve": 15,
    "capture_sac": 15,
    "unsafe_mve": 0,
    "unsafe_sac": 0
  },
  "s0/step_1000": {
    "n": 24,
    "mean": -0.6297907299381317,
    "without_top2": -10.201237292178952,
    "urgent": -1.7226715820186995,
    "economy": 1.1916773568628147,
    "positive": 11,
    "negative": 13,
    "capture_mve": 21,
    "capture_sac": 21,
    "unsafe_mve": 0,
    "unsafe_sac": 0
  },
  "s0/step_2000": {
    "n": 24,
    "mean": 18.658584324433253,
    "without_top2": 2.2028027189123254,
    "urgent": 28.74051481846985,
    "economy": 1.855366834372262,
    "positive": 16,
    "negative": 8,
    "capture_mve": 24,
    "capture_sac": 22,
    "unsafe_mve": 0,
    "unsafe_sac": 0
  },
  "s0/step_3000": {
    "n": 24,
    "mean": -2.3576349492164135,
    "without_top2": -3.219411151362267,
    "urgent": -2.6150306939838552,
    "economy": -1.9286420412706775,
    "positive": 9,
    "negative": 15,
    "capture_mve": 23,
    "capture_sac": 23,
    "unsafe_mve": 1,
    "unsafe_sac": 0
  },
  "s0/step_4000": {
    "n": 24,
    "mean": 0.0995840042344985,
    "without_top2": -0.40513024170437784,
    "urgent": -0.05360721888475458,
    "economy": 0.35490270943325364,
    "positive": 14,
    "negative": 10,
    "capture_mve": 23,
    "capture_sac": 23,
    "unsafe_mve": 0,
    "unsafe_sac": 0
  },
  "s1/step_1000": {
    "n": 24,
    "mean": -1.0361537737167434,
    "without_top2": -19.829572883558438,
    "urgent": -3.8739329274230956,
    "economy": 3.693478149127179,
    "positive": 16,
    "negative": 8,
    "capture_mve": 20,
    "capture_sac": 20,
    "unsafe_mve": 0,
    "unsafe_sac": 0
  },
  "s1/step_2000": {
    "n": 24,
    "mean": -0.777136546110229,
    "without_top2": -10.853163692510702,
    "urgent": 0.22679100050200607,
    "economy": -2.4503491237972845,
    "positive": 14,
    "negative": 10,
    "capture_mve": 22,
    "capture_sac": 22,
    "unsafe_mve": 0,
    "unsafe_sac": 0
  },
  "s1/step_3000": {
    "n": 24,
    "mean": -0.6404434773502162,
    "without_top2": -10.510468463561212,
    "urgent": -2.176101263211639,
    "economy": 1.918986165752159,
    "positive": 11,
    "negative": 13,
    "capture_mve": 22,
    "capture_sac": 22,
    "unsafe_mve": 0,
    "unsafe_sac": 0
  }
}
```
机制最终均值由原生v251按1000步之后model_stats计算，M/E1/E2/S逐项数值与verdict原件完整提供。two_stage_share、decisive_share、w2_mean、pi_gap_sigma_median、grad_cos正常只能说明机制工作；c_mean只报告不作门槛；是否有部署收益由匹配评估判定。六开局复用T2.5冻结84ee9c1审计包含阶段2标签、实际进入口与未进入口状态，不把缺匹配数据填0。
训练真实回报带探索，不能替代确定性部署曲线；两种子单独列报，不挑种子/检查点。

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
