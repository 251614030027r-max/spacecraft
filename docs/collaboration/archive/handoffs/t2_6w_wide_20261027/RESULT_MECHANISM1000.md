# T2.6W 固定140–200秒、共享校准与48开局验证：mechanism1000

生成2026-10-27T23:58:47.050407+09:00，科学`e1166eb5daee069e7f14b6f48d24dcf43c36d66f`，执行书collab`69d49fa`。本机44项测试/真实deadline冒烟及日志入场验收见随附回执。SAC/EACC各两个种子均从零4000、每组2个异步worker；没有预筛选档位。EACC共享阶段/入口校准，SAC模型项关闭且不共享校准。任务动力学、执行器、奖励、动作、K、β、候选集合与阈值冻结。
判定窗口为500–3000；4000完成不等于收敛，不用末期最佳点替换预登记判据。原生pass仅本轮开发门通过，不能替代论文级证据。

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
训练真崩溃同配置同seed最多一次从零，先保全attempt，不拼Replay/优化器/RNG；纯日志/发布失败不重训。单例错误原件在记录/episode_errors保全后仅同例补一次。模型ZIP/Replay/TB二进制本机保留不入Git。普通Git临时index/--no-filters保护活动HEAD/index，不强推；必要原始1728+336+smoke与模型/源SHA、实际事件解析、原生判据随轮交付。无后续分支或长训授权。
用户运行中要求桌面预留1个逻辑核：本轮进程限制CPU1–11，逻辑CPU0留桌面；未改科学或worker、未重启。实际墙钟包含该系统资源政策和用户并行工作，不按独占12核估算，详见DESKTOP_CPU_RESERVATION。
