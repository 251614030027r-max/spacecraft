# 最新工程已回到DRL2：整理完成与边界

用户授权直接整理本地DRL2及Spacecraft，并从Git找回已有代码/文档/证据位置，不再补剩余评估。最新科学工程为D:/py/DRL2，科学工作分支claude/sac-mpc-coupling-design-ns7g6i，4795910102cf9fbf1dfec74b77e9449623a7b51e，Git工作树干净。临时D:/py/spacecraft不替代这个工程入口，原.venv、PyCharm配置及三个final2模型SHA均未改变。

## 实际执行

- 归档前4608个logs/eval/local_artifacts文件、3,999,974,773字节复制到D:/spacecraft_archive/return_drl2_20261010/pre_clean_source并逐文件SHA核验，旧未跟踪/忽略输出在residual_outputs。Git正常切换科学分支，无reset/force/历史重写，无删除原始证据；三个有效模型目录仍在logs/final2_262460–262462，eval只留代码。
- 62个保留科学Python文件的Git blob与F01固定f2f8169相同；新目录下docs/current、HISTORY、DEAD_ENDS、evidence、docs/archive由已整理科学分支恢复。ASSET_LOCATION_INDEX.csv填入4608个原资产的当前有效路径、备份位置及SHA；GIT_ASSET_ENTRIES.csv给出核验过的历史标签证据路径与索引。旧JSON里的历史绝对路径不改造冒充当时路径，从外部索引找资产。
- 代码测试合计354通过、3预期失败、0最终错误。初次完整运行341通过、3预期失败，13项因未创建basetemp父目录在setup报错；修正目录后仅重跑这13项，13通过。科学代码未改，日志和命令附TEST_VERIFICATION及pytest_setup_retry.log。不能说一次整套运行无报错。
- 本机Spacecraft项目入口、最新交付和主题README已更新。旧两模型阶段交付移入上层交付/历史/F01_STAGE_20261009，迁移前后SHA一致；本轮完整三模型读数和现有650个结果在F01_closed_records_20261010，不含模型或嵌套ZIP。

## 收口而不是补造

271000七个正式行336回合完整（另nominal48仅描述），冻结f2f8169官方三模型METHOD_DOES_NOT_HOLD，problems=[]。现存全部评估650/768，272000缺118，重放12/0/11未齐。Git中没有本机未执行的结果，缺失保留编号，不能用其他开局/实验替代；不统计不完整块排行榜或完整误差指标。

用户已取消剩余记录仿真，本次不补旧六回合行为复现，不能把源码一致或单元测试当成该复现通过。新方法/奖励/接口/阈值没有改，新研究执行仍等上层与用户明确。所有已知旧评估和监督进程已消失，保全中断现场，不重启、不清锁。

## 保留的事项

main未合并，科学/协作分支不互混；7个历史归档标签及6个旧分支处理此前已完成，没有重复改远端引用。旧DRL2_v3e、历史工作树、ORPHAN及769源文件归属歧义保留，没以“无价值”为由删除。新归档与旧C段备份有部分重复，按索引可查，未来确认后再去重。外置离线备份未提供目的地，尚未建立；同盘归档不是离线备份。
