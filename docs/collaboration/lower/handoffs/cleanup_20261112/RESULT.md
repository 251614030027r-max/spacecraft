# RUN_CLEANUP Windows 核验交付

**状态：A 已通过。B 在 B4 的 resolve 停止。C 未启动。**

- 执行书 collab `7a5dd220374a91bfa19e3d46797447451b083dc3`。Windows 科学检出 `21730075ecee9d9bbd06f34febf981121db87e68`，参照 `fcd8d2eec2d78faf43ae6522772589d6494cb7e2`，同机比较。
- A 的 5 个 annotated tag 已推送并回读，目标及标签对象见 [TAGS.json](TAGS.json)。
- B2 全量 pytest 158 passed、0 xfailed，退出码 0，耗时 322.24 秒。见 [PYTEST.json](PYTEST.json) 与 [pytest.log](pytest.log)。
- B3 四组通过。7 种环境情形各 2 个开发开局，逐步记录一致。6 个配置各以种子 10、单进程 40 决策训练，所有检查点张量、保留的回合和训练统计字段、worker_order 一致。旧检出训练模型在 294300/294301、GOV_B 模型在 294800/294801 各两回合保留字段逐决策一致。后两回合均成功，决策数分别为 3 和 4。GOV_B 使用执行书指定脚本读取 final.zip，核对过该 ZIP 与登记的 step_3000.zip 的 4 个参数和优化器成员字节一致。各组回执在科学分支 `evidence/CLEANUP_EQUIV/windows/`。
- B4 在 `resolve` 前完成 29 个入口调用，包括 6 个两进程 40 决策训练、17 个评估调用、3 个固定策略筛选、traj、geoscan 和 check。check 对开发冒烟输出返回 `complete=false`。完整前缀见科学分支 `SMOKE_SUMMARY.json`。

## 停止点

`python -B -m experiments.final resolve --stage formal --root <B/DRY> --output <B/DRY/resolve.json>` 的原生退出码为 1，回执为 `budgets [40] differ from the stage budget 5000`。第一处不一致是冒烟 manifest 的 `steps=40` 与登记阶段的 `steps=5000`。冻结源码 `experiments/final.py:450` 对有运行记录的输入执行这项检查。见 [FIRST_DIFFERENCE.json](FIRST_DIFFERENCE.json) 和 [原生日志](smoke_resolve.log)。

依执行书停止。后续读出和作图未运行。C 的资产盘点、删除、文档更新及 READY 未执行，参照检出和 Windows 原始核验结果保存在本机 `过程文件/cleanup_20261112/`。正式评估块和正式训练种子未使用。待上层复核 resolve 的冒烟验收口径。
