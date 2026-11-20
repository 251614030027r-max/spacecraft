# 复现：环境、测试、命令与开局号段

## 环境

```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt
```

`requirements.txt`（数值栈与学习库）+ `requirements-mpc.txt`（CVXPY 与求解器）+ `requirements-dev.txt`（pytest）。PyPI 的 torch 即可。

## 测试

```bash
python -B -m pytest -q
```

以退出码判定（0 = 全部通过）。`tests/test_ecc.py` 覆盖方法本身（RWG 几何与运动学、阶段目标与安全投影、窗口重定时、信息边界、17 项观测、任务、机会可达性势函数、模式预测与协调映射、学习器设置、TensorBoard 日志）；`tests/test_round2_entry.py`、`tests/test_formal_audit.py`、`tests/test_td3_formal.py` 覆盖注册表、开局与种子边界、评估开关、只读日志与 TD3；其余覆盖动力学、MPC、任务判据等执行器与物理模块。

## 第二轮实验

唯一入口 `experiments/final.py`；所有配置、阶段、种子与开局来自 `docs/final_method/METHOD_REGISTRY.json`；规格 `docs/final_method/ROUND2.md`；执行书在 collab 分支 `docs/collaboration/upper/run_orders/`。

```bash
python -B -m experiments.final train   --stage dev --config ECC --seed 0 --output ROOT/B/TRAIN_ECC_s0
python -B -m experiments.final eval    --stage dev --run ROOT/B/TRAIN_ECC_s0 --output ROOT
python -B -m experiments.final screen  --stage STAGE --policy pure --output ROOT          # 阶段登记了 pure 时
python -B -m experiments.final eval    --stage STAGE --run RUN --switch no_long_entry --output ROOT   # 阶段登记了该开关时
python -B -m experiments.final traj    --entry ECC=RUN --openings A,B --output DIR
python -B -m experiments.final check   --stage dev --root ROOT --output ROOT/CHECK.json
python -B -m experiments.final resolve --stage dev --root ROOT --output ROOT/REGISTRY_RESOLVED.json
tensorboard --logdir ROOT/B                                                              # 学习曲线（每个运行的 tb/）
```

冒烟（不用任何登记种子）：`train --stage dev --config ECC --seed 7 --smoke-steps 40 --workers 1 --output ...`，`eval --openings 294300-294301`。

第一稿（第一轮）的命令与读出脚本在科学提交 `72a8d41`；论文第一稿数据仍可由 `python -B -m experiments.paper_readout --output evidence/PAPER_READOUT` 从 `evidence/FORMAL/` 重新生成。

**可复现性：** 评估确定、逐回合串行（并行只在回合之间）。训练用异步采集：单进程逐位可复现；多进程时转移到达顺序取决于各进程快慢（`finished.json` 的 `worker_order`），种子相同的两次多进程训练不保证逐位相同。同一提交在不同机器上可能有末位差异，跨机器比较判定而不比较比特。

## 开局号段

| 号段 | 用途 |
|---|---|
| 290xxx–293xxx、2995xx–2999xx | T1 线的筛选与核验（历史） |
| 294000–294999 | T2 / EACH / ECC 开发期评估（含冒烟用 294300–294301、等价性用 294020 / 294031 与 294302–294305、几何扫描） |
| 295000–295099、296xxx–299xxx | T1 线开发集（历史） |
| 295100–295147 | 经济任务预算的预筛（`formal_prescreen`，只用非学习策略；已完成，2.0 m/s） |
| 295200–295223 | 第一稿验证曲线（强制紧急） |
| **295300–295395** | **第二轮反馈机制开发（`dev` 阶段）** |
| 300000–300295 | 第一稿正式块（U / E / N），第二轮起不再使用 |
| 300296–999999 | 正式块的未用部分，从未访问 |
| ≥ 1,000,000 | 训练开局：1,000,000·(seed + 1) + 进程序号 × 100,000 + 回合序号；开发期种子 0、1、10、11，第一稿正式种子 20、21、22（不再使用），第二轮开发种子 0、1、2 |
