# T2.17＋T2.19 联合交付与审查入口

两轮均各4条独立训练、每条2000步/2worker，各192个固定开发评估；训练图来自真实逐回合日志近50回合滑动，不由过程评估点拼接。方法与参照同2000预算，正式判定以各轮原生JSON为准。

| 组别 | 训练曲线AUC | 末模型回报 | unsafe | 原生判定 |
|---|---:|---:|---:|---|
| e_sac | -0.5244 | -10.2716 | 0 | 固定对照 |
| e_acc | 22.7092 | 33.3026 | 0 | 固定对照 |
| e_priv | 25.4311 | 37.6908 | 0 | 固定对照 |
| e_pi | 14.3852 | 18.6922 | 0 | revert |
| e_pull | 15.1975 | 29.2294 | 0 | no_increment |
| e_cert | 15.2485 | 11.2347 | 0 | revert |
| e_full | 2.6322 | 6.4617 | 0 | revert |

## 上层应核对的结论

T217动作级局部/非局部反馈及T219机会图/完整反馈分别按原生判据列上表，不能挑好种子或把反馈覆盖当收益。两个开发种子不证明泛化，也不能唯一归因于标签、输入维度或执行模型。三份T219原生判定与事前选取结果如下；按原书执行，不因本报告改判：

```json
{
  "T219_method_selection": {
    "final_method": "U1+C1",
    "rule": 3,
    "stop_similar_development_iterations": true,
    "formal_experiments_require_new_user_authorization": true
  },
  "T219_e_full_vs_e_cert": {
    "criteria": {
      "S_safety": true,
      "N_not_below_T2_11": false,
      "I_increment_over_ablation": false,
      "G_above_sac": true,
      "complete": true
    },
    "verdict": "revert",
    "margin": 8.3,
    "values": {
      "e_full": {
        "curve_auc": 2.6322026873003876,
        "final_mean_return": 6.461746661096408,
        "late_return": 15.435017845231659,
        "late_urgent_capture": 0.6006976744186047,
        "final_urgent_capture": [
          14,
          12
        ],
        "final_economy_capture": [
          23,
          23
        ],
        "unsafe": 0,
        "runs": {
          "e_full_s0": {
            "curve_auc": 2.730550420290929,
            "final_mean_return": 11.765809822193466
          },
          "e_full_s1": {
            "curve_auc": 2.5338549543098456,
            "final_mean_return": 1.1576834999993484
          }
        }
      },
      "e_cert": {
        "curve_auc": 15.24845578549546,
        "final_mean_return": 11.234662726515944,
        "late_return": 19.920345685233563,
        "late_urgent_capture": 0.6530303030303031,
        "final_urgent_capture": [
          19,
          10
        ],
        "final_economy_capture": [
          23,
          23
        ],
        "unsafe": 0,
        "runs": {
          "e_cert_s0": {
            "curve_auc": 30.27384029036098,
            "final_mean_return": 32.059859755999376
          },
          "e_cert_s1": {
            "curve_auc": 0.2230712806299431,
            "final_mean_return": -9.590534302967487
          }
        }
      },
      "e_acc": {
        "curve_auc": 22.709208722553516,
        "final_mean_return": 33.302632996327304,
        "late_return": 38.43826689337975,
        "late_urgent_capture": 0.8166666666666667,
        "final_urgent_capture": [
          18,
          20
        ],
        "final_economy_capture": [
          23,
          23
        ],
        "unsafe": 0,
        "runs": {
          "e_acc_s0": {
            "curve_auc": 23.276024210379585,
            "final_mean_return": 28.473869494904196
          },
          "e_acc_s1": {
            "curve_auc": 22.142393234727447,
            "final_mean_return": 38.13139649775041
          }
        }
      },
      "e_sac": {
        "curve_auc": -0.5243532634724675,
        "final_mean_return": -10.271634188013277,
        "late_return": 12.576377153150437,
        "late_urgent_capture": 0.5764119601328903,
        "final_urgent_capture": [
          11,
          8
        ],
        "final_economy_capture": [
          23,
          23
        ],
        "unsafe": 0,
        "runs": {
          "e_sac_s0": {
            "curve_auc": 1.6101397940241866,
            "final_mean_return": -4.588178893568767
          },
          "e_sac_s1": {
            "curve_auc": -2.6588463209691215,
            "final_mean_return": -15.955089482457785
          }
        }
      }
    },
    "reported": {
      "reference_minus_ablation": {
        "curve_auc": 7.460752937058055,
        "final": 22.06797026981136
      }
    },
    "arms": {
      "method": "e_full",
      "ablation": "e_cert",
      "reference": "e_acc",
      "sac": "e_sac"
    }
  }
}
```

## 固定证据与不重复范围

- T217：已推送科学 `729b712820bc162c1b84d39cdbabe9c97c04e37f`，`evidence/T2_17_PULL/`；collab `53f1171b773052b7263104f778a16626380ed629`，`docs/collaboration/lower/handoffs/t2_17_pull_20261104/RESULT.md`。192例、四模型SHA和270项本机SHA已独立核验，本次不重新提交、不复制旧资产。
- T219：新增科学 `cb9f685b41fbd8cf17627795d9cee10eaa92797f`，`evidence/T2_19_CERT/`；本次collab提交内的 `docs/collaboration/lower/handoffs/t2_19_cert_20261104/RESULT.md` 含完整配置、三份判定、机会图/反馈走势、分段回报、墙钟和异常。
- 本联合入口的 `SUMMARY.json` 保留各种子数值及原生判定，原始逐例结果仅在各自科学固定路径，不重复打包。T217 A2按用户要求取消；T218条件false未运行；T219无额外审计或关键状态补算。

## 异常与完整性

T219评估父进程退出、遗留零CPU孤儿worker导致输出等待，未发现可证实的退出原因。已按创建时间和PID核验后接管，保全原114例哈希，只补齐其余78例；不重训、不改科学源/判据、不重算已有结果。原时钟及11.3h不开始新评估限制保留。输入模型、原始C、日志、SHA256SUMS、逐Git blob和远端回执由本轮独立核验覆盖。当前只交付已有开发证据，后续正式实验等待用户新授权。
