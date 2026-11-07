# 复现说明（当前有效，2026-10-10）

## 环境

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-mpc.txt -r requirements-dev.txt
python -B -m pytest -q          # 仓库没有 conftest.py，也不作为包安装，必须从仓库根目录运行
```

新工作副本建议用部分克隆，历史中约 1.5 GB 的模型 ZIP 不会被下载：

```bash
git clone --filter=blob:limit=5m -b claude/sac-mpc-coupling-design-ns7g6i <repo>
```

## 主线命令

**训练**（每个种子一个进程，从零开始，60k 外层决策）：

```bash
python -B -m train.train_stopping --steps 60000 --seed 262460 --run-name final2_262460 --regime w2.36_r15
```

**评估**（确定性，串行）：

```bash
python -B -m experiments.v3_stopping evaluate --run-dir logs/final2_262460 --row stopping --seeds 271000-271047 --output-dir eval/final2/stopping_262460_271000
python -B -m experiments.v3_stopping evaluate --run-dir logs/final2_262460 --row pure     --seeds 271000-271047 --output-dir eval/final2/pure_271000
python -B -m experiments.regime_screen evaluate ...   # nominal 行
```

**判定与表格：**

```bash
python -B -m experiments.v3_stopping readout --pure <dir> --stopping M=<dir> --learned M=<dir> --seeds 271000-271047 --output ...
python -B -m experiments.final_tables ...
python -B -m experiments.stopping_replay replay ... ; python -B -m experiments.stopping_replay summarize ...
```

**训练健康检查**（只读，训练中可以运行）：

```bash
python -B -m experiments.v3_stopping_training_health --run-dir logs/final2_262460
```

F01 的逐步命令以协作分支 `docs/collaboration/upper/run_orders/FINAL_RERUN_20261007.md` 为准，本页不重复。

## 产物与证据

- 训练产物（模型、检查点、监控文件、逐开局 JSON）被 `.gitignore` 忽略，保存在本机，并按实验 ID 归档。
- 进入决策或论文的文件，用 `git add -f` 单独发布到 `evidence/<实验ID>/`，并在总账中登记 SHA-256。
- **不提交整个产物目录。**
- 实验总账：协作分支 `docs/collaboration/registry/EXPERIMENT_REGISTRY_20261007.csv`。
