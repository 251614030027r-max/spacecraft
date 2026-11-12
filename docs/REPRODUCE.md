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

以退出码判定（0 = 全部通过）。`tests/test_ecc.py` 覆盖方法本身（RWG 几何与运动学、阶段目标与安全投影、窗口重定时、信息边界、观测视图、任务分布、模式预测与认证映射、学习器设置与视图分离）；`tests/test_formal_*.py`、`tests/test_td3_formal.py` 覆盖注册表、评估选项、只读日志、读出与作图；其余覆盖动力学、MPC、任务判据等执行器与物理模块。

## 正式实验

唯一入口 `experiments/final.py`，所有配置与开局来自 `docs/final_method/METHOD_REGISTRY.json`；执行书在 collab 分支 `docs/collaboration/upper/run_orders/RUN_FORMAL.md`。

```bash
python -B -m experiments.final train  --stage formal --config ECC_CERT_LOOP --seed 20 --output ROOT/B/TRAIN_ECC_CERT_LOOP_s20
python -B -m experiments.final eval   --stage formal --run ROOT/B/TRAIN_ECC_CERT_LOOP_s20 --condition urgent --output ROOT
python -B -m experiments.final eval   --stage formal --run ROOT/B/TRAIN_ECC_CERT_LOOP_s20 --condition urgent --no-certify --output ROOT
python -B -m experiments.final eval   --stage formal --run ROOT/B/TRAIN_ECC_CERT_LOOP_s20 --condition validation --checkpoint-step 1000 --output ROOT
python -B -m experiments.final screen --stage formal --policy pure --condition urgent --output ROOT
python -B -m experiments.final check   --stage formal --root ROOT --output ROOT/CHECK.json
python -B -m experiments.final resolve --stage formal --root ROOT --output ROOT/REGISTRY_RESOLVED.json
python -B -m experiments.formal_readout formal   --root ROOT --output ROOT/FORMAL_READOUT.json
python -B -m experiments.formal_readout episodes --root ROOT --output ROOT/EPISODES.csv
python -B -m experiments.formal_figures --root ROOT --readout ROOT/FORMAL_READOUT.json --output ROOT/FIGURES
```

开发冒烟（不用正式种子与正式块）：`train --smoke-steps 40 --seed 10 --workers 1`，`eval --openings 294300-294301`。

**可复现性：** 评估确定、逐回合串行（并行只在回合之间）。训练用异步采集：单进程逐位可复现；多进程时转移到达顺序取决于各进程快慢（`finished.json` 的 `worker_order`），种子相同的两次多进程训练不保证逐位相同。同一提交在不同机器上可能有末位差异，跨机器比较判定而不比较比特。

## 开局号段

| 号段 | 用途 |
|---|---|
| 290xxx–293xxx、2995xx–2999xx | T1 线的筛选与核验（历史） |
| 294000–294999 | T2 / EACH / ECC 开发期评估（含冒烟用 294300–294301、等价性用 294020 / 294031、几何扫描） |
| 295000–295099、296xxx–299xxx | T1 线开发集（历史） |
| 295100–295147 | 资源条件预筛（`formal_prescreen`，只用非学习策略） |
| 295200–295223 | 正式实验验证曲线（强制紧急） |
| **300000–300095** | **正式 U（强制紧急）** |
| **300100–300195** | **正式 E（强制经济），R 复用（预算 2.0 m/s）** |
| **300200–300295** | **正式 N（自然 1 : 1）** |
| 300296–999999 | 正式块的未用部分，从未访问 |
| ≥ 1,000,000 | 训练开局：1,000,000·(seed + 1) + 进程序号 × 100,000 + 回合序号；开发期种子 0、1、10、11，正式种子 20、21、22 |
