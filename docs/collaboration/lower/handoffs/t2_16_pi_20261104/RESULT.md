# T2.16 执行一致策略改进：本机事实交付

用户下发已批准执行书collab71ce5b5b，科学detached `323b8c87ad6833fc3c9754c8733e1f81f5162f29`，环境0.12.0/学习器6.1，87项pytest退出码0。A先真值方向审计，不训练；只有原生pass才允许进入B，正式B前另验收本SHA原生日志/TensorBoard和更新/RNG无扰动。科学代码、策略、开局、门槛均未修改。

## A 原生判定

```json
{
  "verdict": "pass",
  "criteria": {
    "D_data": true,
    "P1_wins": true,
    "P2_mean": true,
    "S_safety": true
  },
  "constants": {
    "delta": 0.25,
    "disagree_cos": 0.9,
    "tie": 1.0,
    "min_informative": 20,
    "win_share": 0.6
  },
  "episodes": 32,
  "decisions": 149,
  "branched": 137,
  "branched_share": 0.9194630872483222,
  "turned": 80,
  "informative": 20,
  "wins": 14,
  "win_share": 0.7,
  "mean_M_minus_Q_turned": 2.6422718198506785,
  "mean_M_minus_Q_turned_90ci": [
    -0.11326149060568413,
    7.6840193759128255
  ],
  "unsafe_M": 0,
  "unsafe_Q": 0,
  "reported": {
    "M_vs_Q_all_branched": {
      "n": 137,
      "informative": 25,
      "win_share": 0.64,
      "mean": 0.08013694783406944,
      "median": 0.0
    },
    "E_vs_Q_turned": {
      "n": 80,
      "informative": 34,
      "win_share": 0.6470588235294118,
      "mean": -2.5742039011699,
      "median": 0.1921765797263717
    },
    "M_vs_pi_turned": {
      "n": 80,
      "informative": 16,
      "win_share": 0.5,
      "mean": 2.3382613925240365,
      "median": -0.006591159791803136
    },
    "Q_vs_pi_turned": {
      "n": 80,
      "informative": 12,
      "win_share": 0.25,
      "mean": -0.30401042732664274,
      "median": -0.02927562022520469
    },
    "cos_g_gq_median": -0.8415816873113007,
    "g_over_gq_median": 0.9762694295114528
  },
  "by_label": {
    "e_priv_s0_1000": {
      "decisions": 31,
      "branched": 28,
      "M_vs_Q_turned": {
        "n": 12,
        "informative": 1,
        "win_share": 1.0,
        "mean": 0.09175876101105611,
        "median": 0.025262411823653963
      }
    },
    "e_priv_s0_500": {
      "decisions": 45,
      "branched": 41,
      "M_vs_Q_turned": {
        "n": 24,
        "informative": 8,
        "win_share": 0.625,
        "mean": 0.12764887617724593,
        "median": 0.05668298679493233
      }
    },
    "e_priv_s1_1000": {
      "decisions": 39,
      "branched": 37,
      "M_vs_Q_turned": {
        "n": 24,
        "informative": 9,
        "win_share": 0.7777777777777778,
        "mean": 8.795587113600241,
        "median": 0.1884305083398914
      }
    },
    "e_priv_s1_500": {
      "decisions": 34,
      "branched": 31,
      "M_vs_Q_turned": {
        "n": 20,
        "informative": 2,
        "win_share": 0.5,
        "mean": -0.1938511649369019,
        "median": -0.0007340008175873436
      }
    }
  }
}
```

A使用T2.12 e_priv两种子500/1000共四检查点，第一批固定2980000–2980007；仅insufficient时补2980008–2980015一次，总A上限3h。12进程、CPU0–11、Normal、OMP/MKL/OPENBLAS1。已完成开局数32；分支与门槛按原生summary，D有效差值>1至少20、P1胜率≥0.60、P2原生turned子集均值>0、S unsafe_M≤unsafe_Q。原生plot横轴是两个方向夹角的余弦，纵轴M减Q真实回报差，未经改绘。A3h若截停只用已完整写出的开局，不能冒称全部完成。

## B 是否执行

未执行。原生A判定为pass。A通过后，本SHA日志验收的48步 e_pi 冒烟在命令行参数解析时退出码2：冻结 experiments.t2 的 --arm 选项不包含 e_pi/e_pit，尚未进入训练。日志门槛因此失败，按执行书停报；没有修改科学源或绕过验收，没有正式B训练、模型评估或训练收益结论。原错误和退出回执随交付保存；需上层修复并重新冻结可运行入口后再授权推进。

## 版本、对照与解释边界

固定历史：T2.12 e_priv2000最终37.6908、曲线25.4311；e_rel17.4160、曲线10.4631。T2.14 e_term最终11.6430/revert，e_term2 2.3694/unsafe(1例)，曲线13.2548/9.2660。T2.11 e_acc最终33.3026、曲线22.7092，普通SAC最终−10.2716、曲线−0.5244。固定版本及哈希引用PRIOR_DELIVERIES、REFERENCE_INPUTS，不重复历史整包。B若进行，e_pi对e_rel、e_pit对e_term为既定机制对照；训练曲线与最终模型确定性回报不同，不能混用。两种子开发证据不作正式泛化。A即便pass也不保证训练收益；fail/insufficient只说明本次预注册真值证据未达到入训条件，不能强行推出唯一归因。

缺项：[]。执行时间线：
```text
start A 11/04/2026 13:43:49
audit 2980000-2980007 done at 2.05054248216667 h: verdict=pass

```

科学证据 `55a32e4a12a57f5540b4f68c491f228184ef2580`，`evidence/T2_16_PI/`，含完整原生A/RUNS、DIRECTION.json/png、日志、87测试、输入模型SHA与CAL哈希、SHA256SUMS，若B进行另有原生曲线/判定/C逐例/元数据/模型SHA及反馈覆盖走势；逐Git blob原字节校验通过。本机全部过程、模型、滚动日志和TB保留于 `C:\Users\35884\Documents\Spacecraft\过程文件\T2_16_PI_20261104`。下一实验等待上层新授权，不执行T215。
