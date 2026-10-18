# 复现（2026-10-18）

## 1. 环境

- Python 3.12。依赖：`numpy scipy cvxpy gymnasium stable-baselines3 torch pytest`（PyPI 的 torch 可用）。
- 自 `166244b` 以来 `requirements*.txt` 没有变化：下层本机 T1 / T1-L / T1-O 共用同一个 Python 环境。
- 线程：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`。
- 项目没有安装成包，也没有 `conftest.py`：一律在仓库根目录运行 `python -B -m ...`。

## 2. 测试

```bash
python -B -m pytest -q                       # 全部（约 23 min）
python -B -m pytest -q tests/test_t1_o.py tests/test_t1_l_s.py tests/test_t1_l.py tests/test_t1_counterfactual.py \
    tests/test_t1_gym.py tests/test_t1_termination.py tests/test_t1_fixes.py tests/test_t1_m1.py   # T1 相关：106 项
```

## 3. 常用命令

```bash
# 开局（只按初始几何）
python -B -m experiments.t1o_openings --profile p2 --set discovery            # 或 confirmation、development
python -B -m experiments.t1o_screen openings --profile p1 --output <file>     # 物理筛选开局

# 固定候选闭环（逐开局 JSON，含逐步 Δv 与四档 λ_f 回报）
python -B -m experiments.t1o_screen run --profile p2 --policy A15 --openings <file> --output-dir <dir> --workers 4
python -B -m experiments.t1o_screen table --openings <file> --root <dir> --output per_opening.csv

# T1-O V2 读出与串联
python -B -m experiments.t1o_v2 gate-a --openings <discovery> --root <dir> --output gate_a.json
python -B -m experiments.t1o_v2 state-rule --openings <discovery> --root <dir> --fuel-weight 20 --output state_rule.json
python -B -m experiments.t1o_gates gate-b --openings <discovery> --uniform A15 --fuel-weight 20 --output gate_b.json
python -B -m experiments.t1o_v2_chain --root logs/t1o_v2 --workers W        # A → B → C → 零步 → E，遇 FAIL 即停

# 已有证据的诊断（不新增回合）
python -B -m experiments.t1o_diagnose --profile p2 --episodes <dir> --output diagnose.json

# 训练（T1-O 四臂需 --profile 与 --fuel-cost；拒绝从修改过的工作树训练）
python -B -m train.train_t1_option --arm or3 --profile p2 --fuel-cost 20 --steps 12000 --seed 0 --run-name <name> --n-envs 4
```

## 4. 产物

- 训练与评估产物写在 git 忽略的 `logs/` 下。
- 证据只以单个文件强制加入 `evidence/<实验ID>/`，并附 `SHA256SUMS`。
- 下层大包放在 collab 的 `docs/collaboration/lower/handoffs/<主题>/`；超过 50 MB 的留本机，只交 SHA。
- 模型 zip 一律不上传，只交 SHA。
