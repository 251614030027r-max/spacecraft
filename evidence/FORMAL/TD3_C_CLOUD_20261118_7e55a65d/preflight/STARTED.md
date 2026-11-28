# TD3_C 云端执行：启动事实（非最终交付）

用户 2026-11-18 明确授权：仅 TD3_C，种子 20、21、22，各 5000 决策，注册表的 2 个采集 worker。固定科学提交 7e55a65d373666bdb901357e07f5c53713647a4c。本次授权覆盖旧 RUN_FORMAL 的 TD3 停止项与 RUN_CLOSEOUT 的等待条件，仅执行该 TD3 范围；不恢复 ECC2_GATE0 / ECC_EVT。

已读取协作提交 90d2ed06377ff047bec26ad077ae74b762abc250 的 HANDOFF、PROJECT_STATE、RUN_FORMAL、RUN_CLOSEOUT 与 RUN_ECC_LE。后两份有效评估范围为每种子 urgent 300000–300095 与验证 295200–295223 的 1000/2000/3000/4000/5000 检查点；不运行其他组或固定策略。

实际资源：3 个逻辑 CPU，但 cgroup 配额 200000/100000（2 核）；内存上限 8 GiB；安装后磁盘剩余约 5.7 GiB。种子串行，每个种子训练后立即评估再转下一个。没有访问本机 ECC_LE/SAC_LE 的进程、代码或资源。

冻结工作区干净。python -B -m pytest -q：161 passed，退出码 0。依赖按冻结 requirements-dev.txt 安装，Torch 2.9.1+cpu；完整解析依赖另附。未修改科学代码、配置、预算、随机流或判据。

原生训练仅每 1000 步保存进度，未打开 TensorBoard。仓库外 /workspace/td3-cloud-ops/sitecustomize.py 使用 Python 3.12 sys.monitoring 读取 train 已存在的局部变量，逐决策记录 progress.jsonl，逐完成回合记录 episodes.live.jsonl，并写 TensorBoard。它不替换科学函数或写入科学对象；会带来日志开销，注册的异步到达顺序本来依赖时序。原生 episodes.json 在结束后与滚动记录逐项核对。操作脚本及 SHA 另附，不属于科学源码。

后台探针已跨命令退出存活并由 PID 1 接管；监督器 PID 1140，训练学习进程 PID 1163，采集 worker 1166/1167。此快照 2026-11-18T06:03:20.796460+00:00：种子 20 实际 47/5000 决策，5 个完成回合；种子 21、22 尚未启动，已在串行监督器排队。manifest.steps=5000 是预算，不是已完成数。

云平台未提供容器生命周期/会话结束后保活保证；脱离命令存活的实测不等于跨容器回收持续运行。容器回收会中断，禁止据此声称完成，也不从检查点续训。若中断，应保留 aborted 目录并按同种子从零重跑。

云端产物根：/workspace/td3_cloud_runs/TD3_C_7e55a65d_20261118。状态 STATUS.json，调度 SCHEDULE.jsonl，原始日志 logs/；监督器每阶段检查冻结文件，训练结束核对 manifest、5000 条转移、两个 worker、模型 ZIP、原生/滚动回合一致、TensorBoard 与 SHA256；评估核对开局集合、无 error、预算与审计字段，保留 numerical_failure。完整授权范围是 15000 决策、288 个 urgent 和 360 个验证回合；不能称全量多方法矩阵 complete。

本目录只追加 TD3 证据。未更新或覆盖任何 CURRENT，未强推，未操作本机。
