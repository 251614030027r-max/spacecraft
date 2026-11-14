# 证据目录索引

本目录只放已执行实验的事实证据（读出、判定、清单、汇总；模型 ZIP 与大量原始回合文件不入库，只登记 SHA-256）。2026-11-12 工程收尾时，整理前的完整目录在科学提交 `fcd8d2ee`（标签 `archive/pre-cleanup-20261112`）中原样保存；移出工作树的目录见 `docs/archive/PATH_MAP_20261112.csv`，取回方法：`git checkout fcd8d2ee -- evidence/<目录>`。

## A　现行方法（ECC）与论文直接相关

| 目录 | 内容 | 论文用途 |
|---|---|---|
| `FORMAL_PREP/` | 正式实验准备（RUN_FORMAL_PREP）：对 `e0a61753` 的环境与训练逐位回归、6 组开发冒烟、日志与补跑核验、读出与作图冒烟、资源条件预筛（R = 2.0 m/s） | 正式实验的软件准入与 R 的冻结依据 |
| `GOV_B/`、`GOV_B_UPPER/` | 开发期 B 步：在环 ECC 对 SAC_C（紧急 +36.0，捕获 145 / 111，翻转 37 : 3）、上层分解（学习侧反馈为主要来源） | 方法选型依据；附录"开发期结果"，不并入正式样本 |
| `GOV_A/` | 开发期 A 步：执行机会调节器（governor）被事先登记的判据否决 | DEAD_ENDS D45；回答"为何不用更强的执行期干预" |
| `ECC_DEV/` | RUN_ECC：完整 ECC（证书进 critic）与部署期认证 | DEAD_ENDS O6、O7（下降类认证的安全性，论文局限） |
| `ECC_FINAL_PRECHECK/` | 最终轮的等价性脚本与重放（`scripts/` 针对当时的检出运行） | 等价性方法的来源 |
| `ECC_PRECHECK/` | ECC 实现前的沙箱核验 | — |
| `U2_PROBE_PRECHECK/`、`U2_TRAIN/` | 执行器模式探询作 critic 输入（`no_gain`） | DEAD_ENDS D44 |
| `U2C/`、`U2C_PRECHECK/` | 决策时认证的首次检验（认证映射的来源） | 认证机制的演进 |
| `final_pretest/` | EACH 预实验：含 DUP（等宽 critic 对照）、NO_GEOM、窗口 / 刚性配对 | 附录：容量混杂（等宽对照）与旋转引导的开发期证据 |

## B　负结果台账（DEAD_ENDS D39–D43 引用；审稿质疑"为何不用反事实标签 / 动作反馈"时使用）

| 目录 | 对应 |
|---|---|
| `T2_7_PRECHECK/`、`T2_7_GUIDE/` | 执行器引导（DEAD_ENDS 引用） |
| `T2_8_PRECHECK/`、`T2_8_NIGHT/`、`T2_8_P4_INFO/`、`T2_8_P4_INFO_UPPER/` | P4 信息诊断：工程化观测已把规划器答案交给策略 |
| `T2_11_EACCA/`、`T2_11_EACCA_UPPER/`、`T2_11_SELFCHECK/` | EACC-A（反事实标签 + 决策级校准）；D43 |
| `T2_12_PRECHECK/`、`T2_12_EACC/` | 相对反事实标签 `e_rel`；D39 |
| `T2_13_PRECHECK/`、`T2_14_PRECHECK/`、`T2_14_TERM/` | 只保留可评估的终局反事实；D40 |
| `T2_16_PRECHECK/`、`T2_16_PI/`、`T2_17_PULL/` | 执行优势修正 actor 梯度；D41 |
| `T2_19_CERT/` | 校准机会图作 critic 输入；D42 |

## 已移出工作树（历史，按需从 `fcd8d2ee` 取回）

| 组 | 目录 | 说明 |
|---|---|---|
| 早期 T2（T2.0–T2.6W1） | `T2_PRECHECK`、`T2_STAGE_A/B`、`T2_2_TEST`、`T2_3_TEST`、`T2_4_TEST`、`T2_5_VALIDATE`、`T2_5_1_VALIDATE`、`T2_6_PRESSURE`、`T2_6W_WIDE_140_200`、`T2_6W1_CUTOFF` | 分阶段接口的建立、时间压力与截断语义；结论已并入 `docs/HISTORY.md` 与 DEAD_ENDS |
| T1 线 | `T1_*`、`T1B_*`、`T1N_*`、`T1O_*`、`T1P_*` | 单层审议接口、反事实 MPC、区间 Bellman 等；结论见 DEAD_ENDS D1x–D3x |
| F01 / V3 线 | `E01`–`E08`、`historical/` | 价值仲裁与停止选项；结论见 DEAD_ENDS D01–D1x |
| 开题 PPT 图 | `ppt_dev/` | 不用于论文 |

**正式实验**的证据将放在 `FORMAL/`（执行书 collab `RUN_FORMAL.md`）。
