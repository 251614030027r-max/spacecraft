用户最新授权仅评估1000/2000/3000/4000步共768例。取消点已有结果SHA保全，不补算；原完整t26w窗口不再适用，不宣称完整pass。
# T2.6W 固定140–200秒、共享校准与48开局验证：final

生成2026-10-28T21:16:27.265968+09:00，科学`e1166eb5daee069e7f14b6f48d24dcf43c36d66f`，执行书collab`69d49fa`。本机44项测试/真实deadline冒烟及日志入场验收见随附回执。SAC/EACC各两个种子均从零4000、每组2个异步worker；没有预筛选档位。EACC共享阶段/入口校准，SAC模型项关闭且不共享校准。任务动力学、执行器、奖励、动作、K、β、候选集合与阈值冻结。
原完整500–3000判据因用户缩减不再适用；保留原生命令输出DECISION_NATIVE和机制读数，最终标scope_revised_descriptive_only，不重新定义门槛。

## 千步机制早报（早期窗口）
```json
{
  "jst": "2026-10-27T23:58:46.145504+09:00",
  "source_SHA": "e1166eb5daee069e7f14b6f48d24dcf43c36d66f",
  "deadline": [
    140.0,
    200.0
  ],
  "readings": {
    "e_mve_s0": {
      "after1000": {
        "two_stage_share": {
          "mean": 0.8009995669126511,
          "n": 4,
          "first_step": 1012,
          "last_step": 1050,
          "MISSING": false
        },
        "decisive_share": {
          "mean": 0.13836846128106117,
          "n": 4,
          "first_step": 1012,
          "last_step": 1050,
          "MISSING": false
        },
        "stage2_rows": {
          "mean": 1050.25,
          "n": 4,
          "first_step": 1012,
          "last_step": 1050,
          "MISSING": false
        },
        "w2_mean": {
          "mean": 0.9074080586433411,
          "n": 4,
          "first_step": 1012,
          "last_step": 1050,
          "MISSING": false
        },
        "pi_gap_sigma_median": {
          "mean": 0.03541981987655163,
          "n": 4,
          "first_step": 1012,
          "last_step": 1050,
          "MISSING": false
        },
        "stage2_entry_share": {
          "mean": 0.1729729026556015,
          "n": 4,
          "first_step": 1012,
          "last_step": 1050,
          "MISSING": false
        },
        "c_mean": {
          "mean": 0.13838102295994759,
          "n": 4,
          "first_step": 1012,
          "last_step": 1050,
          "MISSING": false
        },
        "increment_abs_mean": {
          "mean": 1.8029332756996155,
          "n": 4,
          "first_step": 1012,
          "last_step": 1050,
          "MISSING": false
        },
        "grad_cos": {
          "mean": -0.23999178409576416,
          "n": 1,
          "first_step": 1050,
          "last_step": 1050,
          "MISSING": false
        },
        "loss_share": {
          "mean": 0.3795521855354309,
          "n": 4,
          "first_step": 1012,
          "last_step": 1050,
          "MISSING": false
        },
        "w_mean": {
          "mean": 0.948999360203743,
          "n": 4,
          "first_step": 1012,
          "last_step": 1050,
          "MISSING": false
        }
      },
      "first_model_step": 337,
      "T251_3worker_reference_step": 512,
      "sandbox_2worker_pooled_reference_step": 362,
      "reference_not_local_observation": true
    },
    "e_mve_s1": {
      "after1000": {
        "two_stage_share": {
          "mean": 0.7103412422266874,
          "n": 11,
          "first_step": 1012,
          "last_step": 1137,
          "MISSING": false
        },
        "decisive_share": {
          "mean": 0.10492671044035391,
          "n": 11,
          "first_step": 1012,
          "last_step": 1137,
          "MISSING": false
        },
        "stage2_rows": {
          "mean": 674.1818181818181,
          "n": 11,
          "first_step": 1012,
          "last_step": 1137,
          "MISSING": false
        },
        "w2_mean": {
          "mean": 0.9490993781523271,
          "n": 11,
          "first_step": 1012,
          "last_step": 1137,
          "MISSING": false
        },
        "pi_gap_sigma_median": {
          "mean": 0.031231849539009007,
          "n": 11,
          "first_step": 1012,
          "last_step": 1137,
          "MISSING": false
        },
        "stage2_entry_share": {
          "mean": 0.14782436937093735,
          "n": 11,
          "first_step": 1012,
          "last_step": 1137,
          "MISSING": false
        },
        "c_mean": {
          "mean": 0.1082228957252069,
          "n": 11,
          "first_step": 1012,
          "last_step": 1137,
          "MISSING": false
        },
        "increment_abs_mean": {
          "mean": 1.311318419196389,
          "n": 11,
          "first_step": 1012,
          "last_step": 1137,
          "MISSING": false
        },
        "grad_cos": {
          "mean": 0.18787720426917076,
          "n": 2,
          "first_step": 1050,
          "last_step": 1100,
          "MISSING": false
        },
        "loss_share": {
          "mean": 0.3808334605260329,
          "n": 11,
          "first_step": 1012,
          "last_step": 1137,
          "MISSING": false
        },
        "w_mean": {
          "mean": 0.9331225915388628,
          "n": 11,
          "first_step": 1012,
          "last_step": 1137,
          "MISSING": false
        }
      },
      "first_model_step": 512,
      "T251_3worker_reference_step": 537,
      "sandbox_2worker_pooled_reference_step": 362,
      "reference_not_local_observation": true
    }
  },
  "alerts": [
    "e_mve_s0"
  ],
  "scope": "Actual native probes steps>1000, early window not final M; c reported only; no performance claim"
}
```
严格after1000的原生TB探针，列样本数与首末步数；最终M按原生完整model_stats。c只报告，不作门。T251标准3worker的512/537和沙箱2worker共享362是参照，实际启动步数单独列；不能把不同任务/worker的历史差单独归因于共享校准。机制启用、w2高或梯度同向不等于部署收益。

## 最终完整训练机制读数（>=1000步，独立于取消的评估点）
```json
{
  "TRAIN_e_mve_s0": {
    "two_stage_share": 0.6290622858795527,
    "decisive_share": 0.13126095773891788,
    "c_mean": 0.13057767847143267,
    "grad_cos": 0.3771403348226039,
    "loss_share": 0.4517637190646067,
    "increment_abs_mean": 1.0906616968732652,
    "w_mean": 0.9539881271940049,
    "w2_mean": 0.9461876208356802,
    "pi_gap_sigma_median": 0.06025302969692159,
    "stage2_rows": 1200.0373443983403,
    "stage2_entry_share": 0.20608174856520292
  },
  "TRAIN_e_mve_s1": {
    "two_stage_share": 0.6318433771445946,
    "decisive_share": 0.12381505921359379,
    "c_mean": 0.1308445222521224,
    "grad_cos": 0.45221936488982106,
    "loss_share": 0.49996417442108665,
    "increment_abs_mean": 1.2541757782465195,
    "w_mean": 0.9436034899034935,
    "w2_mean": 0.9404733072177998,
    "pi_gap_sigma_median": 0.0517800538897267,
    "stage2_rows": 1022.7551867219917,
    "stage2_entry_share": 0.19303671202595304
  }
}
```
机制读数不替代配对回报收益；c仅报告。

## 现有证据的问题定位
# T2.6W 性能退化诊断：现有证据与源码检查

本分析读取冻结e1166eb已有结果和完整训练日志，不新增动力学仿真、模型forward、训练或调参。统计快照为1000/2000/3000两seed各48开局，4000当时尚未完成。重复检查点上的次数不是独立开局数。配对开局mission逐一相等已核验。

## 1. 已确认的性能损失机制
六个seed×检查点的EACC−SAC回报差全负，均值−12.70276。24次SAC成功/EACC失败，只有5次相反，净少19次捕获；全部翻转发生于紧急任务。经济任务各点均小幅正收益，但抵消不了每次成功/超时翻转约200回报的差距。不能解释成仅多花燃料或统计由一个开局决定：去掉最差294016后六点平均仍−9.3004。294016/294035/294045分别出现5/3/3次损失，包含旧24与新24中的开局。

在24次SAC成功/EACC失败中：18次EACC结束时未在bridge/descent宏段，6次处于宏段；6次有入口丢失记录。EACC平均多6.083个阶段决策。轨迹支持“接近/侧向动作偏离→近场徘徊或退离→迟进/未进→超时”；这些是已观察轨迹，不是反事实干预的因果证明。

例：s0/3000/294019，SAC在107.2s捕获、4阶段、Δv1.010；EACC198.3s超时、13阶段、Δv2.208。EACC48s到7.233m后出现收拢动作负值与方位反转，随后在7–8m附近徘徊；SAC连续推进到入口。s0/2000/294035，SAC93.4s捕获；EACC176s已到约3m但仍超时，表明进入和接近不等于在截止前达到最终捕获。s0/3000/294016曾丢失入口，后来径向动作接近−1并退离至13.907m。动作a_p映射v_c=.05+.15a_p，a_p接近−1代表退离意图，不能笼统解释为节省燃料。

## 2. 已确认的一处实现缺口：无入口预测不处理截止终止
冻结env/t2_stage.py predict_intent的无入口分支直接推进16s并done=False；predict_second_stage无入口分支同样推进16s且done=False。_scan会限制入口计划的剩余时间，但“没有可行入口”之后的普通阶段分支没有把deadline当终止。真实执行器在截止会timeout并扣终止罚分；学习器对done=False预测继续bootstrap，二阶段行又无AGE_CAP。

对冻结predict_intent原函数做AST提取的惰性stub分支检查（不是动力学仿真、不是实际训练标签）：t=146、deadline=150、无入口，函数返回t=162、done=False、reward=−1.6。证明此分支的时间终止语义不完整。具体训练中有多少标签实际越界、其权重及造成多大收益损失，现有汇总日志未提供，标MISSING；不将此缺口宣称为唯一原因。真实终止当步可能被not_comparable筛掉，因此不能推断每个临近截止转换都收到了错误标签；两阶段前瞻跨截止尤其值得下一轮专项验证。

## 3. 机制门通过为何仍可学坏
完整>=1000统计：w2约.946/.940、gap .060/.052σ、grad_cos .377/.452；但它们是内部一致性/模型损失信号，不是候选动作捕获正确率。w2主要用真实已执行动作的模型误差，不能逐候选保证终止和入口标签正确；gap比较同一个target critic上的两个动作，并以真实TD RMS σ归一化，critic共同误判时仍可很小。grad_cos是混合任务、全部critic参数上模型/真实TD梯度的相似性，不是紧急子集的actor策略梯度，更不是最终捕获收益。

模型项并非没起效：>=1000 loss_share约.452/.500，即β×模型损失约为真实损失的45%/50%，这是损失数值比，不是梯度范数比。模型梯度大部分同向，但1000–2000仍有20%/30%探针为负，2000–3000为15%/10%。不能凭均值正排除关键稀少终止样本被错误拟合或共享参数干扰。

## 4. 次级风险及已排除的简单解释
入口模型的实际−预测偏差在3000–4000均值为−8.38/−9.57，说明执行入口回报相对预测仍偏低；但这是当前worker滚动入口残差的探针均值，不是逐反事实候选的真值误差。源码的EntryCalibration.predict主要用共享历史入口的时长/燃料/成功和slack转换，并非状态充分条件下的精确捕获模型。平均偏差降权不能保证临界截止附近的成功排序正确。

模型标签在末段平均龄期约1596/1492个真实转换；AGE_CAP命中的候选比例约27.6%/22.6%，但只关闭Retrace修正，独立stage2 Bellman行仍使用。这不等价于数学错误的“旧动作不可离策略学习”；风险来自模型状态/入口结果标签的误差长期保留，现有日志无法拆出其净影响。

共享校准首模型337/512步：seed0确实较早，seed1仍约512。两seed均明显落后，且后段仍存在差距，因此“只因反馈太晚”不足以解释。训练探索回合也有相同方向：2000–3000紧急捕获率EACC67.5%/58.5%，SAC74.6%/69.6%；训练开局序列因异步/回合长度不同，这不是配对因果比较，但说明不是仅48个评估开局上的偶然局部现象。

近场直接捕获标签漏判属于已知未修项；本轮eval关闭候选生成，不能从这些轨迹直接判断某个动作当时被标签识别成什么。旧84ee六审计不能替代当前e116/紧急140–200工况的标签真值审计。不能把旧审计当本轮准确性证明。

## 5. 建议的下一步优先顺序（仅建议，未执行新实验）
1. 优先核查/修正普通阶段和二阶段预测的deadline/资源/捕获终止语义，在固定真实状态对候选标注进行单元检查；不以提高w2或放大β解决语义问题。
2. 对已确认“SAC能捕获、EACC超时”的代表开局294016/294035/294019，在关键48–96s决策点做冻结策略真值分支，核查入口/最终捕获/截止惩罚和候选排序。需要下一次明确执行授权，当前未跑。
3. 分别检查紧急/经济、临界slack、近场捕获/入口丢失的标签误差与梯度，避免全局平均M掩盖关键类别。
4. 若新版本要区分机制因果，只做预登记单因素对照（如终止语义修正或stage2行因素），不同时改任务、β、校准、标签模型后把改善归因于其中一项。未获授权不启动。

当前可以确认损失落在可达紧急任务的控制策略/终止学习上，不能确认唯一根因或承诺某个补丁必然有收益。原四点评估继续完成并发布。


## 缩减范围结果（原生完整判定不适用）
```json
{
  "criteria": {
    "C0_complete": false
  },
  "values": {},
  "verdict": "scope_revised_descriptive_only",
  "reported": {
    "late_s0": {
      "paired": {
        "3000": -8.787403681127538,
        "3500": null,
        "4000": -0.21721500849847372
      },
      "capture": {
        "3000": [
          41,
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
          0,
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
          29.516069632745868,
          null,
          38.68656528223579
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
        "3000": -21.85328460388506,
        "3500": null,
        "4000": -14.452282024820816
      },
      "capture": {
        "3000": [
          36,
          41
        ],
        "3500": null,
        "4000": [
          37,
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
          6.365156408630679,
          null,
          11.706624003230397
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
        "capture_mve": 45,
        "capture_sac": 52,
        "paired_mean": -16.76526333692045
      },
      "mid": {
        "openings": 7,
        "capture_mve": 33,
        "capture_sac": 43,
        "paired_mean": -38.551474447012815
      },
      "low": {
        "openings": 7,
        "capture_mve": 42,
        "capture_sac": 49,
        "paired_mean": -26.40081221718419
      }
    },
    "both_success": {
      "economy": {
        "checkpoints": 8,
        "delta_v": -0.06396327087063244,
        "time_s": -1.2755681818181817
      },
      "urgent": {
        "checkpoints": 8,
        "delta_v": 0.03599433758468969,
        "time_s": 1.6344060448065858
      }
    }
  },
  "native_verdict": "incomplete",
  "scope_revision": {
    "marks": [
      1000,
      2000,
      3000,
      4000
    ],
    "full_t26w_window": "MISSING; cancelled by user",
    "no_full_pass_claim": true
  }
}
```
C0为48/48。E1双seed都正且均值>=5；U跨检查点紧急捕获差双seed合计>=4且每seed非负，是重复检查点累计次数，不是独立开局数。E2/S取3000末点及登记窗口；不得用4000或最佳检查点替代。

|臂/种子|步数|任务|n|平均回报|捕获|unsafe|
|---|---:|---|---:|---:|---:|---:|
|e_mve_s0|1000|all|48|8.9807584685724|36|0|
|e_mve_s0|1000|urgent|25|-21.69667009201342|13|0|
|e_mve_s0|1000|economy|23|42.325789512687415|23|0|
|e_mve_s0|2000|all|48|29.019386361983255|41|0|
|e_mve_s0|2000|urgent|25|16.69660172237509|18|0|
|e_mve_s0|2000|economy|23|42.413717491992145|23|0|
|e_mve_s0|3000|all|48|29.516069632745868|41|0|
|e_mve_s0|3000|urgent|25|16.39987320485458|18|0|
|e_mve_s0|3000|economy|23|43.7728048804538|23|0|
|e_mve_s0|4000|all|48|38.68656528223579|43|0|
|e_mve_s0|4000|urgent|25|33.87999398914231|20|0|
|e_mve_s0|4000|economy|23|43.91109929646784|23|0|
|e_mve_s1|1000|all|48|-1.2281079052184147|34|0|
|e_mve_s1|1000|urgent|25|-40.77577194824742|11|0|
|e_mve_s1|1000|economy|23|41.75848344590007|23|0|
|e_mve_s1|2000|all|48|7.999696787241149|36|0|
|e_mve_s1|2000|urgent|25|-23.205780110968508|13|0|
|e_mve_s1|2000|economy|23|41.918693415729905|23|0|
|e_mve_s1|3000|all|48|6.365156408630679|36|1|
|e_mve_s1|3000|urgent|25|-25.425862147645304|13|1|
|e_mve_s1|3000|economy|23|40.92061136110459|23|0|
|e_mve_s1|4000|all|48|11.706624003230397|37|0|
|e_mve_s1|4000|urgent|25|-15.45756953019133|14|0|
|e_mve_s1|4000|economy|23|41.23292132216706|23|0|
|e_sac_s0|1000|all|48|29.04778947799677|41|0|
|e_sac_s0|1000|urgent|25|17.976626468971162|18|0|
|e_sac_s0|1000|economy|23|41.081662313894164|23|0|
|e_sac_s0|2000|all|48|38.39297527542198|43|0|
|e_sac_s0|2000|urgent|25|35.231727152355646|20|0|
|e_sac_s0|2000|economy|23|41.82911453962453|23|0|
|e_sac_s0|3000|all|48|38.303473313873404|43|0|
|e_sac_s0|3000|urgent|25|35.451885745877235|20|0|
|e_sac_s0|3000|economy|23|41.40302501821707|23|0|
|e_sac_s0|4000|all|48|38.90378029073427|43|0|
|e_sac_s0|4000|urgent|25|35.68354026332568|20|0|
|e_sac_s0|4000|economy|23|42.40404119009142|23|0|
|e_sac_s1|1000|all|48|5.12445096560766|36|0|
|e_sac_s1|1000|urgent|25|-23.683950427794535|13|0|
|e_sac_s1|1000|economy|23|36.43793074104483|23|0|
|e_sac_s1|2000|all|48|17.78238499022|39|0|
|e_sac_s1|2000|urgent|25|2.460977463053066|16|0|
|e_sac_s1|2000|economy|23|34.4360888240971|23|0|
|e_sac_s1|3000|all|48|28.218441012515743|41|0|
|e_sac_s1|3000|urgent|25|17.548290439785436|18|0|
|e_sac_s1|3000|economy|23|39.816430765483474|23|0|
|e_sac_s1|4000|all|48|26.158906028051216|41|0|
|e_sac_s1|4000|urgent|25|24.25256705549355|19|0|
|e_sac_s1|4000|economy|23|28.231013606918232|22|0|

## 全检查点配对、紧急/经济拆分
```json
{
  "s0/step_1000": {
    "n": 48,
    "mean": -20.067031009424365,
    "without_top2": -30.01222347709888,
    "urgent": -39.67329656098458,
    "economy": 1.2441271987932636,
    "positive": 25,
    "negative": 23,
    "capture_mve": 36,
    "capture_sac": 41,
    "unsafe_mve": 0,
    "unsafe_sac": 0,
    "urgent_capture_mve": 13,
    "urgent_capture_sac": 18,
    "urgent_bins": {
      "high": {
        "n": 11,
        "mean": -36.09501066020466,
        "capture_mve": 5,
        "capture_sac": 7
      },
      "mid": {
        "n": 7,
        "mean": -86.97870225035287,
        "capture_mve": 3,
        "capture_sac": 6
      },
      "low": {
        "n": 7,
        "mean": 2.0090884271581047,
        "capture_mve": 5,
        "capture_sac": 5
      }
    },
    "both_success": {
      "all": {
        "n": 34,
        "delta_v": -0.024204901935420704,
        "time_s": -1.4705882352941155
      },
      "economy": {
        "n": 23,
        "delta_v": -0.030240136491569855,
        "time_s": -1.7260869565217367
      },
      "urgent": {
        "n": 11,
        "delta_v": -0.011585775136199749,
        "time_s": -0.9363636363636348
      }
    }
  },
  "s0/step_2000": {
    "n": 48,
    "mean": -9.373588913438725,
    "without_top2": -15.186947155426893,
    "urgent": -18.535125429980564,
    "economy": 0.5846029523676192,
    "positive": 15,
    "negative": 33,
    "capture_mve": 41,
    "capture_sac": 43,
    "unsafe_mve": 0,
    "unsafe_sac": 0,
    "urgent_capture_mve": 18,
    "urgent_capture_sac": 20,
    "urgent_bins": {
      "high": {
        "n": 11,
        "mean": -19.452044644678956,
        "capture_mve": 6,
        "capture_sac": 7
      },
      "mid": {
        "n": 7,
        "mean": -32.71329052339813,
        "capture_mve": 5,
        "capture_sac": 6
      },
      "low": {
        "n": 7,
        "mean": -2.916087284894096,
        "capture_mve": 7,
        "capture_sac": 7
      }
    },
    "both_success": {
      "all": {
        "n": 40,
        "delta_v": 0.02240104473759666,
        "time_s": 3.6799999999999984
      },
      "economy": {
        "n": 23,
        "delta_v": -0.015167247722234023,
        "time_s": 1.1043478260869568
      },
      "urgent": {
        "n": 17,
        "delta_v": 0.07322873453619114,
        "time_s": 7.164705882352939
      }
    }
  },
  "s0/step_3000": {
    "n": 48,
    "mean": -8.787403681127538,
    "without_top2": -14.190549837996349,
    "urgent": -19.052012541022656,
    "economy": 2.3697798622367165,
    "positive": 18,
    "negative": 30,
    "capture_mve": 41,
    "capture_sac": 43,
    "unsafe_mve": 0,
    "unsafe_sac": 0,
    "urgent_capture_mve": 18,
    "urgent_capture_sac": 20,
    "urgent_bins": {
      "high": {
        "n": 11,
        "mean": 17.464147739573363,
        "capture_mve": 8,
        "capture_sac": 7
      },
      "mid": {
        "n": 7,
        "mean": -62.10573472952354,
        "capture_mve": 4,
        "capture_sac": 6
      },
      "low": {
        "n": 7,
        "mean": -33.3808279363155,
        "capture_mve": 6,
        "capture_sac": 7
      }
    },
    "both_success": {
      "all": {
        "n": 40,
        "delta_v": -0.015330208428907189,
        "time_s": 1.1425000000000007
      },
      "economy": {
        "n": 23,
        "delta_v": -0.05874884438200468,
        "time_s": -0.9913043478260862
      },
      "urgent": {
        "n": 17,
        "delta_v": 0.0434126519782247,
        "time_s": 4.029411764705883
      }
    }
  },
  "s0/step_4000": {
    "n": 48,
    "mean": -0.21721500849847372,
    "without_top2": -5.280683155489101,
    "urgent": -1.8035462741833799,
    "economy": 1.5070581063764243,
    "positive": 22,
    "negative": 26,
    "capture_mve": 43,
    "capture_sac": 43,
    "unsafe_mve": 0,
    "unsafe_sac": 0,
    "urgent_capture_mve": 20,
    "urgent_capture_sac": 20,
    "urgent_bins": {
      "high": {
        "n": 11,
        "mean": 17.555111498736064,
        "capture_mve": 8,
        "capture_sac": 7
      },
      "mid": {
        "n": 7,
        "mean": -31.99858525431514,
        "capture_mve": 5,
        "capture_sac": 6
      },
      "low": {
        "n": 7,
        "mean": -2.0292552229250345,
        "capture_mve": 7,
        "capture_sac": 7
      }
    },
    "both_success": {
      "all": {
        "n": 42,
        "delta_v": 0.003763040924972335,
        "time_s": 2.4785714285714295
      },
      "economy": {
        "n": 23,
        "delta_v": -0.038076452659410794,
        "time_s": 0.8000000000000008
      },
      "urgent": {
        "n": 19,
        "delta_v": 0.05441084894817296,
        "time_s": 4.510526315789474
      }
    }
  },
  "s1/step_1000": {
    "n": 48,
    "mean": -6.35255887082607,
    "without_top2": -7.736129538576168,
    "urgent": -17.09182152045288,
    "economy": 5.320552704855245,
    "positive": 28,
    "negative": 20,
    "capture_mve": 34,
    "capture_sac": 36,
    "unsafe_mve": 0,
    "unsafe_sac": 0,
    "urgent_capture_mve": 11,
    "urgent_capture_sac": 13,
    "urgent_bins": {
      "high": {
        "n": 11,
        "mean": -0.23264657601294336,
        "capture_mve": 4,
        "capture_sac": 4
      },
      "mid": {
        "n": 7,
        "mean": -30.68013380104822,
        "capture_mve": 4,
        "capture_sac": 5
      },
      "low": {
        "n": 7,
        "mean": -29.99649843826316,
        "capture_mve": 3,
        "capture_sac": 4
      }
    },
    "both_success": {
      "all": {
        "n": 34,
        "delta_v": -0.06877808235609162,
        "time_s": -2.211764705882354
      },
      "economy": {
        "n": 23,
        "delta_v": -0.13115512196920703,
        "time_s": -3.7173913043478275
      },
      "urgent": {
        "n": 11,
        "delta_v": 0.06164663683496787,
        "time_s": 0.9363636363636361
      }
    }
  },
  "s1/step_2000": {
    "n": 48,
    "mean": -9.782688202978852,
    "without_top2": -15.924945749032883,
    "urgent": -25.666757574021577,
    "economy": 7.482604591632806,
    "positive": 23,
    "negative": 25,
    "capture_mve": 36,
    "capture_sac": 39,
    "unsafe_mve": 0,
    "unsafe_sac": 0,
    "urgent_capture_mve": 13,
    "urgent_capture_sac": 16,
    "urgent_bins": {
      "high": {
        "n": 11,
        "mean": -1.098515402350437,
        "capture_mve": 5,
        "capture_sac": 5
      },
      "mid": {
        "n": 7,
        "mean": -30.594620481038557,
        "capture_mve": 4,
        "capture_sac": 5
      },
      "low": {
        "n": 7,
        "mean": -59.34613236534496,
        "capture_mve": 4,
        "capture_sac": 6
      }
    },
    "both_success": {
      "all": {
        "n": 35,
        "delta_v": -0.12601571053023025,
        "time_s": -5.371428571428573
      },
      "economy": {
        "n": 23,
        "delta_v": -0.18327163652994988,
        "time_s": -7.586956521739131
      },
      "urgent": {
        "n": 12,
        "delta_v": -0.016275185697434386,
        "time_s": -1.1250000000000024
      }
    }
  },
  "s1/step_3000": {
    "n": 48,
    "mean": -21.85328460388506,
    "without_top2": -23.980178027328115,
    "urgent": -42.974152587430744,
    "economy": 1.1041805956211141,
    "positive": 19,
    "negative": 29,
    "capture_mve": 36,
    "capture_sac": 41,
    "unsafe_mve": 1,
    "unsafe_sac": 0,
    "urgent_capture_mve": 13,
    "urgent_capture_sac": 18,
    "urgent_bins": {
      "high": {
        "n": 11,
        "mean": -56.066137454530754,
        "capture_mve": 4,
        "capture_sac": 7
      },
      "mid": {
        "n": 7,
        "mean": -35.36679457908186,
        "capture_mve": 4,
        "capture_sac": 5
      },
      "low": {
        "n": 7,
        "mean": -30.008391518908176,
        "capture_mve": 5,
        "capture_sac": 6
      }
    },
    "both_success": {
      "all": {
        "n": 36,
        "delta_v": -0.006370699805036604,
        "time_s": 0.8361111111111121
      },
      "economy": {
        "n": 23,
        "delta_v": -0.02816321054270169,
        "time_s": 1.1173913043478294
      },
      "urgent": {
        "n": 13,
        "delta_v": 0.03218528073083239,
        "time_s": 0.3384615384615356
      }
    }
  },
  "s1/step_4000": {
    "n": 48,
    "mean": -14.452282024820816,
    "without_top2": -22.080794443980107,
    "urgent": -39.71013658568489,
    "economy": 13.001907715248832,
    "positive": 25,
    "negative": 23,
    "capture_mve": 37,
    "capture_sac": 41,
    "unsafe_mve": 0,
    "unsafe_sac": 0,
    "urgent_capture_mve": 14,
    "urgent_capture_sac": 19,
    "urgent_bins": {
      "high": {
        "n": 11,
        "mean": -56.197011195895286,
        "capture_mve": 5,
        "capture_sac": 8
      },
      "mid": {
        "n": 7,
        "mean": 2.026066042655795,
        "capture_mve": 4,
        "capture_sac": 4
      },
      "low": {
        "n": 7,
        "mean": -55.53839339798067,
        "capture_mve": 5,
        "capture_sac": 7
      }
    },
    "both_success": {
      "all": {
        "n": 36,
        "delta_v": 0.0033778820017522576,
        "time_s": -0.23055555555555746
      },
      "economy": {
        "n": 22,
        "delta_v": -0.026883516667981576,
        "time_s": 0.7954545454545422
      },
      "urgent": {
        "n": 14,
        "delta_v": 0.05093150848276257,
        "time_s": -1.8428571428571427
      }
    }
  }
}
```

## 末期稳定性、压力分箱、双方都成功的燃料/时间（只报告）
```json
{
  "late_s0": {
    "paired": {
      "3000": -8.787403681127538,
      "3500": null,
      "4000": -0.21721500849847372
    },
    "capture": {
      "3000": [
        41,
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
        0,
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
        29.516069632745868,
        null,
        38.68656528223579
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
      "3000": -21.85328460388506,
      "3500": null,
      "4000": -14.452282024820816
    },
    "capture": {
      "3000": [
        36,
        41
      ],
      "3500": null,
      "4000": [
        37,
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
        6.365156408630679,
        null,
        11.706624003230397
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
      "capture_mve": 45,
      "capture_sac": 52,
      "paired_mean": -16.76526333692045
    },
    "mid": {
      "openings": 7,
      "capture_mve": 33,
      "capture_sac": 43,
      "paired_mean": -38.551474447012815
    },
    "low": {
      "openings": 7,
      "capture_mve": 42,
      "capture_sac": 49,
      "paired_mean": -26.40081221718419
    }
  },
  "both_success": {
    "economy": {
      "checkpoints": 8,
      "delta_v": -0.06396327087063244,
      "time_s": -1.2755681818181817
    },
    "urgent": {
      "checkpoints": 8,
      "delta_v": 0.03599433758468969,
      "time_s": 1.6344060448065858
    }
  }
}
```
只在双方均捕获的子集上比较Δv/时间，按紧急/经济区分；这是一项条件比较，不能外推为所有开局的经济优势。用户取消0/500/1500/2500/3500评估；完整窗口及3500读数MISSING，不填补。

## 反馈启动与实际计算开销
```json
{
  "activation": {
    "e_sac_s0": {
      "enabled": false,
      "pooled_calibration": false,
      "first_model_step": null,
      "model_probe_count": 0,
      "T251_reference_step": null,
      "sandbox_pooled_reference_step": null
    },
    "e_sac_s1": {
      "enabled": false,
      "pooled_calibration": false,
      "first_model_step": null,
      "model_probe_count": 0,
      "T251_reference_step": null,
      "sandbox_pooled_reference_step": null
    },
    "e_mve_s0": {
      "enabled": true,
      "pooled_calibration": true,
      "first_model_step": 337,
      "model_probe_count": 294,
      "T251_reference_step": 512,
      "sandbox_pooled_reference_step": 362
    },
    "e_mve_s1": {
      "enabled": true,
      "pooled_calibration": true,
      "first_model_step": 512,
      "model_probe_count": 280,
      "T251_reference_step": 537,
      "sandbox_pooled_reference_step": 362
    }
  },
  "wall": {
    "e_sac_s0": {
      "steps": 4000,
      "wall_s": 71740.09178972244,
      "seconds_per_accepted_decision": 17.93502294743061,
      "updates": 7602,
      "elapsed_parallel_wall_not_sum": true,
      "wrapper_wall_s": 71766.86169624329,
      "wrapper_wait_polling_resolution_s": 15,
      "attempt": 0
    },
    "e_sac_s1": {
      "steps": 4000,
      "wall_s": 69132.82302546501,
      "seconds_per_accepted_decision": 17.283205756366254,
      "updates": 7602,
      "elapsed_parallel_wall_not_sum": true,
      "wrapper_wall_s": 69151.39228963852,
      "wrapper_wait_polling_resolution_s": 15,
      "attempt": 0
    },
    "e_mve_s0": {
      "steps": 4000,
      "wall_s": 68744.96148896217,
      "seconds_per_accepted_decision": 17.186240372240544,
      "updates": 7602,
      "elapsed_parallel_wall_not_sum": true,
      "wrapper_wall_s": 68775.10679578781,
      "wrapper_wait_polling_resolution_s": 15,
      "attempt": 0
    },
    "e_mve_s1": {
      "steps": 4000,
      "wall_s": 62733.79897069931,
      "seconds_per_accepted_decision": 15.683449742674828,
      "updates": 7602,
      "elapsed_parallel_wall_not_sum": true,
      "wrapper_wall_s": 62760.84989333153,
      "wrapper_wait_polling_resolution_s": 15,
      "attempt": 0
    }
  }
}
```
每臂单列真实决策开销与完整wall，不把四个并行wall相加冒充整体用时。完整机制判据和各值在原生DECISION，非零模型损失和梯度不代替配对性能证据。G★仅内部系统参照，不作为阶段动作空间内的学习基线；参照结果不会用于调整本轮冻结训练。

## 标准任务历史固定对照
```json
{
  "science_publication": "62d2d72570b0c214d2c194a9106b237554d39805",
  "collab_publication": "e9cf679be63cd19e21f4db264892a357c1b58af7",
  "standard_4k": {
    "e_mve_s0": 54.57618756322554,
    "e_mve_s1": 53.03017341449411,
    "e_sac_s0": 54.47660355899104,
    "capture_each": 23,
    "n_each": 24,
    "unsafe_each": 0
  },
  "verdict": "mixed",
  "paired_checkpoint_mean_s0": 3.9426856623783,
  "paired_checkpoint_mean_s1": -0.81791126572573,
  "first_model_row_book_reference": [
    512,
    537
  ],
  "sources": [
    {
      "path": "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T2_5_1_VALIDATE_20261026\\B_READOUT.json",
      "sha256": "e2d69f48bdb0b960568b6d936732f9898a0c0b8bad48efaa530025598d762178"
    },
    {
      "path": "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T2_5_1_VALIDATE_20261026\\DECISION.json",
      "sha256": "d23bd38c1357930d9ffda0289b586b760599e1ac1269cfceedfd4542c42db5ba"
    },
    {
      "path": "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T2_5_1_VALIDATE_20261026\\TRAIN_e_mve_s0\\model_stats.json",
      "sha256": "0dd73a4f9c2e4c6fd8bdc07f0cfdcb6595dd9ae91e34f08ba86cc32b3ee23f22"
    },
    {
      "path": "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T2_5_1_VALIDATE_20261026\\TRAIN_e_mve_s1\\model_stats.json",
      "sha256": "88ac7ce3ac9fb8f621d4404a22cb87f64a6c039de7b1234272d66d2b38c7454e"
    }
  ]
}
```
旧标准结果仅作固定SHA参照，本轮SAC和EACC均新训。任务deadline和校准共享两项相对历史变化，不能仅据历史差认定单因素因果收益。近场直接捕获标签漏判本次未修，模型完备性不作保证。

## 日志、异常与交付
外部只读适配器不改变原_slim返回、原生IPC消息或原replay.add已接受转换，不额外forward/backward/RNG；本SHA同批native update权重逐位相等、RNG检查、双臂真实48步异步事件及额外12步非smoke共享校准IPC都在独立目录验证，全部不作科研性能证据。原生smoke关闭候选/共享校准，故另做非smoke12步并检查pooled manifest，不把smoke假充共享校准实证。正式约30秒准确计数、逐回合true/train和原生模型字段，Q1/Q2分值缺失不编造。
训练真崩溃同配置同seed最多一次从零，先保全attempt，不拼Replay/优化器/RNG；纯日志/发布失败不重训。单例错误原件在记录/episode_errors保全后仅同例补一次。模型ZIP/Replay/TB二进制本机保留不入Git。普通Git临时index/--no-filters保护活动HEAD/index，不强推；必要原始768+336+smoke与模型/源SHA、实际事件解析、原生判据随轮交付。无后续分支或长训授权。
用户运行中要求为桌面预留逻辑核，当前实际策略：[]；实验允许核：[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]。科学、worker数不改、无重启，历史策略在CPU_POLICY_HISTORY；实际墙钟包含系统资源政策及用户并行工作，不按独占12核估算。
用户电脑空闲后明确恢复全CPU；为利用约81%负载下的余量，原提前评估75%门槛改92%，仅单路Idle只读原生评估（训练BelowNormal）。行政助手在无在途case时替换，训练未停未重启，正式参数/48样本/路径未变，最终池前仍排空屏障。具体见EARLY_FAST_HANDOFF/POLICY。
