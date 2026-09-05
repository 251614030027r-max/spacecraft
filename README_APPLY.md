# Commit 5b 修复包 — 应用说明（v2，含 h20/h50 视界行）

基线 commit：`c564690`（分支 `claude/sac-mpc-coupling-design-ns7g6i`）。
本包在该 commit 之上产生三个提交。**代码只改了两个文件**，其余是测试、文档和证据。

## 方式 A：打补丁（推荐）

```powershell
Set-Location D:\py\DRL2
git checkout claude/sac-mpc-coupling-design-ns7g6i
git status                 # 工作树干净，HEAD = c564690
git am patches\0001-*.patch
git am patches\0002-*.patch
git am patches\0003-*.patch
python -B -m pytest -q     # 期望 167 passed
```

## 方式 B：直接覆盖

把 `files\` 原样覆盖到 `D:\py\DRL2\`，自行 commit。

| 文件 | 性质 |
|---|---|
| `env/se3_rendezvous_env.py` | **唯一的行为修改**：precapture 分支设 `_episode_tumble_scale` |
| `experiments/evaluate_precapture_oracle.py` | 重写（计划 + 控制律） |
| `tests/test_precapture_task.py` | 新增转速回归用例 |
| `docs/PRECAPTURE_TASK_SPEC.md` | 更新到 V0.2 |
| `HANDOFF_COMMIT5B_RESOLVED_20260904.md` | 本轮结论，**先读这个** |
| `logs/precapture_planning_oracle/*.json` | 5 份证据（可不入库） |

## 必须先读的两点

1. **§2**：`precapture_planning` 之前一直以 0.10308 rad/s 运行，是配置写的
   0.041231 rad/s 的 2.5 倍，外区在物理上不可解。
   所有 `logs/precapture_planning_baseline/*.json` 作废。
2. **§5**：Pure MPC 的前沿测出来了，但 **Go/No-Go 判不过**——
   h20 的失败是 0.264 m 稳态偏差（容差 0.25 m），可能是代价权重的产物；
   h50 只富余 0.004 m。下一步是权重敏感性扫描，不是训练。

## 验证

```powershell
python -B -m pytest -q
# 167 passed

python -B -m experiments.evaluate_precapture_oracle --episodes 5 --seed 262000 --output OUT.json
# acceptance.passed = true, completion_rate = 1.0
```

沙箱环境 Python 3.11 + `requirements-dev.txt` 锁定版本。
`.venv` 启动器指向不存在的 3.12.6，与本包无关，按 `docs/REPRODUCIBILITY.md` 走。
