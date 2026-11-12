# RUN_CLEANUP 合并交付

**收尾完成，等待上层复核与 RUN_FORMAL 用户授权。**

## A 与 B

- 5 个 annotated tag 已推送并回读。见 [TAGS.json](TAGS.json)。
- Windows 冻结 21730075ecee9d9bbd06f34febf981121db87e68 对参照 cd8d2ee。全量 pytest 158 项通过、0 xfailed、退出码 0。环境 7 种情形共 14 回合、6 配置单进程训练、旧模型及 GOV_B 模型两组评估全部逐位一致。科学证据位于 vidence/CLEANUP_EQUIV/windows/。
- 原冒烟根目录完成训练、评估、固定策略筛选、轨迹、几何扫描与 check。resolve 拒绝 40 步预算且退出码 1，符合 collab cf8edd57 的修订验收口径。补跑 formal_readout formal、formal_readout episodes 和 formal_figures --smoke，三项退出码均为 0。见 [B_CONTINUATION.json](B_CONTINUATION.json)。

## C

- 登记后清理 200 个旧产物目标、63105 个文件，逻辑大小 5.84 GiB。删除计划含 Git 汇总证据。完整逐文件清单见 [本机路径与 SHA-256 回执](DELETED_FILES_20261112_SHA256.json)。
- 保留 FORMAL_PREP、GOV_B、GOV_A、RUN_ECC 全部过程目录，复核 239 个模型 ZIP 哈希。保留现用 venv、协作检出及必要的历史交付。旧审查副本中仍有本地改动，历史 Git 元数据中仍有归档标签未覆盖的提交，这两类取证材料继续保留。
- Windows 核验临时参照检出已移除，科学工作区干净并停在 21730075。已更新项目说明和 AGENTS，原文与前后哈希见 [LOCAL_DOCS_SHA256.json](LOCAL_DOCS_SHA256.json)。已注销 67 个退役项目计划任务，并保存其定义。
- C: 剩余 41.13 GiB，较清理前增加 4.85 GiB。
- D: 剩余 60.14 GiB，较清理前增加 2.12 GiB。
- 正式输出目录不存在，项目 Python 进程及计划任务均无残留。[READY.json](READY.json)

本轮未启动正式实验，未使用正式训练种子或正式评估开局。
