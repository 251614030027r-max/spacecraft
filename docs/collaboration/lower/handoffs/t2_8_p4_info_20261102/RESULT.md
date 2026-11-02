# T2.8 P4 事实交付

执行书 collab `a0b0b23`，detached科学 `8016b16df994ed2b206c3eea7096084494006288`；pytest 60 passed、退出码0。按原命令顺序执行：10进程真值分支，再对夜间四条训练500/1000/1500/2000的16个检查点打分，然后原生p4.py汇总。未训练，科学代码未修改。

覆盖：40开局2910000–2910039，紧急19、经济21；16检查点。obs_mismatch=67，nonterminal_branches=0。这两项若非0按执行书原样交付并注明，不改动原始结果。判定与解释由上层进行。

## 墙钟

```json
{
  "jst": "2026-11-02T19:48:38.197522+09:00",
  "stages_seconds": {
    "pytest": 120.06690764427185,
    "truth_branches": 17417.72495865822,
    "score": 30.035371780395508,
    "summary": 30.03070306777954
  },
  "total_seconds": 17597.857941150665
}
```

## 原生汇总

```json
{
  "decisions": 154,
  "phases": {
    "urgent_window": 52,
    "urgent_early": 34,
    "economy": 68
  },
  "decisive_pairs": {
    "urgent_window": 204,
    "urgent_early": 298,
    "economy": 367
  },
  "obs_mismatch": 67,
  "nonterminal_branches": 0,
  "critic_by_arm_phase": {
    "mve/urgent_window": {
      "checkpoints": 6,
      "accuracy_mean": 0.7058823529411765,
      "accuracy_min": 0.6519607843137255
    },
    "mve/urgent_early": {
      "checkpoints": 6,
      "accuracy_mean": 0.7628635346756152,
      "accuracy_min": 0.7315436241610739
    },
    "mve/economy": {
      "checkpoints": 6,
      "accuracy_mean": 0.6684831970935514,
      "accuracy_min": 0.6321525885558583
    },
    "sac/urgent_window": {
      "checkpoints": 6,
      "accuracy_mean": 0.6037581699346405,
      "accuracy_min": 0.5245098039215687
    },
    "sac/urgent_early": {
      "checkpoints": 6,
      "accuracy_mean": 0.7393736017897092,
      "accuracy_min": 0.714765100671141
    },
    "sac/economy": {
      "checkpoints": 6,
      "accuracy_mean": 0.6012715712988193,
      "accuracy_min": 0.5231607629427792
    }
  },
  "signals": {
    "label": {
      "selection": {
        "critic_right": 1439,
        "broken": 6,
        "net": 60,
        "pairs_urgent_early": 1116,
        "critic_wrong": 757,
        "fixed": 66,
        "pairs_urgent_window": 1080,
        "fix": 0.08718626155878467,
        "break": 0.004169562195969423,
        "pass": false
      },
      "heldout": {
        "critic_wrong": 978,
        "fixed": 134,
        "net": 134,
        "pairs_urgent_early": 2460,
        "critic_right": 2850,
        "broken": 0,
        "pairs_urgent_window": 1368,
        "fix": 0.13701431492842536,
        "break": 0.0,
        "pass": false
      },
      "checkpoints_net_positive": "12/12",
      "carries_missing_information": false
    },
    "opp": {
      "selection": {
        "critic_wrong": 1978,
        "fixed": 589,
        "net": 402,
        "pairs_economy": 2940,
        "critic_right": 3158,
        "broken": 187,
        "pairs_urgent_early": 1116,
        "pairs_urgent_window": 1080,
        "fix": 0.29777553083923153,
        "break": 0.05921469284357188,
        "pass": false
      },
      "heldout": {
        "critic_right": 3927,
        "broken": 69,
        "net": 225,
        "pairs_economy": 1464,
        "critic_wrong": 1365,
        "fixed": 294,
        "pairs_urgent_early": 2460,
        "pairs_urgent_window": 1368,
        "fix": 0.2153846153846154,
        "break": 0.01757066462948816,
        "pass": false
      },
      "checkpoints_net_positive": "12/12",
      "carries_missing_information": false
    }
  }
}
```

## 输入检查点SHA256

```json
{
  "e_mve_s0_500": "8d7216f0caff2f68242aa346ac14bab3ccdf45f01c65f8a7a163ae21f0296d38",
  "e_mve_s0_1000": "ceb957f58ea1a828f729dc730f8b50595aa189a4fccffe4c09e487590ae11169",
  "e_mve_s0_1500": "660acae641eb87fbca3687de326fc7a51604cfffc752e31e9d8f64578270dd3d",
  "e_mve_s0_2000": "aee66141e71933d766907e74497ad5ee3620126258b8419345afd3bd951a5323",
  "e_mve_s1_500": "906a79a3c08140fb372bbc671fa2a476b5e95f13feaf76d89dd380adcfd89957",
  "e_mve_s1_1000": "b70bce0cd4ede66998e8da1fa6aa4a74811f1bc8d872c65d348e09e80ed606bd",
  "e_mve_s1_1500": "c2b8ff11a3f3de9b2eb45ac8aa161a0d8dd308293668753679e71d96efabb394",
  "e_mve_s1_2000": "e0b324cc75ef43204bb48f4d7183ef436efb38a56192b5d3bb5c1e727edf076f",
  "e_sac_s0_500": "d7458ff380c40e57f085d1c11ff3e5d28d54e99dd9f8896672de0335ee2565f7",
  "e_sac_s0_1000": "ef72b2035c3e388f62dd3e81602dac2a4624f5734f25133b0d8f42a5f7ef20ee",
  "e_sac_s0_1500": "0fab5990c5af0b8a1c2fb03844ccb97b725ebc4dc01b561cad22f3e6a7d7d957",
  "e_sac_s0_2000": "de9be6f0c4c68a5fec4fdffcd1d68f9daee28f5710a29b22d2ea55147cd0ea0b",
  "e_sac_s1_500": "48e863908b12ebd951d7e9b7d393db41fada02d94d2fb7919e7bb7bacc92f61a",
  "e_sac_s1_1000": "ec7f1d1eaa4dbf5819a244f7c0832406b7b9f4a4082c035fdda8c62485bd8093",
  "e_sac_s1_1500": "f279eb9b39c3cc262fe97c61ae90ac1a118ee5f103c1e3b5d0688d9876586f54",
  "e_sac_s1_2000": "044972fdaa615dfbc1fc17f54741869d25b976d6319f57848f495a03c0c8f55f"
}
```

全部info逐开局、原始KEYSCORE_P4、P4_SUMMARY和四份日志及SHA256SUMS位于科学证据提交 `236fba1bef31618646dec2b9197e6b577612c144` 的 `evidence/T2_8_P4_INFO/`；Git blob已逐个核对原始SHA，未应用换行过滤。完整本机过程：`C:\Users\35884\Documents\Spacecraft\过程文件\T2_8_P4_INFO_20261102`。两个信号的原生汇总标志只按冻结p4.py报告，下层不另定门槛或结论。

资源调度：2026-11-02T16:29:08.440864+09:00按用户电脑空闲提速要求，将现有10个worker及行政父进程由低于正常提升到High（非Realtime），共享CPU0–11；不重启、不暂停、不改变科学配置或池内worker数。没有据此宣称实测速度提升。
