# V3e 60k 第二轮价值验证：上层审查入口

本目录是**已完成实验的审查副本**，只放报告和小型原始结果，不包含大型模型或仿真数据。实验运行提交为 `c84ed7435f790198630ee7011fc7fd74b7307b48`，SAC 训练提交为 `a03632a304bf8af50c4be42cc9d3e450fb23d5ad`。本审查分支以当时远端上层文档提交 `9fed0394e879aeb5d127027e82051c7e85aad993` 为基础；上层随后新增的逐回合诊断指令不属于本轮已执行结果。

## 先看结论

Pure MPC 完成 37/48；60k learned-only 三模型完成 42/48、33/48、38/48。第二轮独立校准为 262420 **PASS（87.9%）**、262421 **STOP（77.1%）**、262422 **STOP（65.6%）**。仅 262420 进入正式 48 回合仲裁，完成 **40/48**、救回 Pure 失败 5 回合、丢失 Pure 成功 2 回合，但有 **1 个真值违规回合**；按预先固定的安全门 L1 不通过，L2 通过。另两模型 STOP，官方总体结论为 **does not hold**，不能用第一轮结果或挑选种子替代第二轮主判定。

从 50k 到 60k，同一 48 开局的 learned-only 完成数分别为 39→42、32→33、30→38。这是策略单独运行的变化，不是仲裁有效性结论。第二轮每模型新增 192 个独立 learned_full 开局，网络、超参数、效用定义、V_B 掩码和仲裁规则保持固定；扩大数据后仍为一过两停。两轮校准开局不同，不能把一致率差当作同样本配对提升，也不能认定数据量是唯一原因。

## 审查顺序

1. [正式报告](REPORT.md)与 [combined_results.json](combined_results.json)：完整策略表、第一轮补充与第二轮主结果的边界、两层判定。
2. [官方判读原始 JSON](readout_r2.json)；各模型的 `m6_r2_<seed>.json` 和 `m3_report_<seed>.json`：校准门、留出误差及原始检查点。
3. [50k 对 60k 原始对照](50k_vs_60k_learned_readout.json)及 [本地交付回执](delivery_receipt.json)：固定版本、包哈希和生成信息。
4. [GitHub Release 完整证据 ZIP](https://github.com/251614030027r-max/spacecraft/releases/tag/v3e-60k-value-r2-20260930)：318 个包内文件，包括 118 个 JSON、日志、执行快照与复现脚本。ZIP 的 SHA-256 为 `96aa157c6529c539c8884468f7b33ef5c444d4452d444ab95a54ff020d1bf960`，大小 42,676,423 字节；`ALL_FILES_SHA256.txt` 与 `JSON_SHA256.txt` 可逐文件核对。

模型 ZIP、M2/M2x NPZ、价值 PT 共 33 个外部大文件按原执行边界未装入证据 ZIP；[路径、大小、SHA 清单](large_artifact_locations.json)在此。远端材料支持审查本轮结论；重新运行仿真仍需取得这些大文件。工程已整理为 `D:/py/DRL2`，原 `D:/py/DRL2_v3e` 为兼容联接；报告中的 Ve 路径是实验发生时的路径，原始科学 JSON 与交付 ZIP 未因迁移改写。

本目录文件的 SHA-256 见 `REVIEW_FILES_SHA256.txt`。并行执行耗时不作为正式串行实时性结论。
