# 最终方法与实验资料

**现状（2026-11-08）：** 方法由 EACH（U1 + C1）演进为 **ECC（执行机会证书耦合）**。

- RUN_ECC 按事先登记的判定顺序选中 **`ECC_deploy`**：critic 读过顶层证书，部署时由执行器认证意图；统一信息边界。
- 正式实验未授权。正式实验前还要再统筹优化一版，所以下表中 EACH 时期的文件需要随之更新。

| 文件 | 内容 | 状态 |
|---|---|---|
| `ECC_FINAL_SPEC.md` | **最终轮（2026-11-08 批准）**：执行机会调节器、通用反馈接口、理论命题 1–5、机会稀缺度条件与筛查门槛、A / B 步判据与选型顺序 | **现行（最终轮进行中）** |
| `ECC_SPEC.md` | ECC 算法（执行机会函数、认证集、认证映射、闭环）、性质 P1–P4、RUN_ECC 判定顺序 | RUN_ECC 版本；旧认证保留为回退 |
| `U2_CERT_SPEC.md` | U2 决策时认证（RUN_U2C）：机制、理论、两档判据、保底 | 历史（已并入 ECC） |
| `U2_PROBE_SPEC.md` | U2 执行器模式探询（作 critic 输入，RUN_U2）、离线判据 A1–A5 | 历史（主判定 `no_gain`，DEAD_ENDS D44） |
| `METHOD_MINIMAL.md` | EACH 六步最小闭环、信息边界、伪代码 | EACH 时期，待按 ECC 更新 |
| `METHOD_SPEC.md` | 宏观决策与真实 TD（T1）、C1、U1 理论（T2）、协议、参数表 | T1、T2 与 C1 部分仍有效；U1 部分待并入 ECC |
| `C1_THEORY.md` | 扫掠锥入口机会集（T3）、窗口承诺可行集包含（T4）、16 s 阶段论证 | 现行 |
| `ABLATION_MATRIX.md` | EACH 时期的训练组与对照 | 待按 ECC 与新接口更新 |
| `FIGURE_SPEC.md` | 图表来源与口径 | 待按 ECC 更新 |
| `CONFIG_LOCK.md` | EACH 冻结配置与等价性审计 | 待按 ECC 更新 |
| `METHOD_REGISTRY.json` | 展示名、arm、开关、阶段、种子、开局 | 已含阶段 `u2`、`u2c`、`ecc` 与配置 `SAC_C`、`EACH_C`、`ECC`；`formal` 阶段仍是 EACH 时期的 6 组，待更新 |

**执行书**（collab `docs/collaboration/upper/run_orders/`）：
- 已执行：`RUN_PRETEST.md`、`RUN_U2.md`、`RUN_U2C.md`、`RUN_ECC.md`；
- 草案、未授权：`RUN_FORMAL.md`，EACH 时期，待重写。

**证据目录：**
- `evidence/final_pretest/`
- `evidence/U2_PROBE_PRECHECK/`、`evidence/U2_TRAIN/`
- `evidence/U2C_PRECHECK/`、`evidence/U2C/`
- `evidence/ECC_PRECHECK/`、`evidence/ECC_DEV/`

**读出脚本：** `experiments/u2_readout.py`、`experiments/cert_readout.py`、`experiments/ecc_readout.py`；离线审计 `experiments/u2_probe_audit.py`。
