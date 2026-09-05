# Commit 5b 修复包 v4 — 直接覆盖到 D:\py\DRL2\ 根目录

本包已按仓库根目录布局，解压后与 `D:\py\DRL2\` 一一对应，直接覆盖即可
（`_patches/` 和本文件不属于仓库，删掉）。
基线 commit `c564690`，本包在其上 6 个提交，`python -B -m pytest -q` = **168 passed**。

## 一句话结论

**缺口成立，在完成率和约束满足上，基线保持满血。** seed block 262000，5 episodes：

| 行 | 完成 | 零违约 | 最差裕度 | 成功者最差裕度 | Δv | 算力预算 |
|---|---|---|---|---|---|---|
| Oracle（离线、真值） | **5/5** | **5/5** | **+0.148** | +0.148 | 1.844 | — |
| MPC h20 `tw1000` | 3/5 | 3/5 | −0.337 | +0.145 | 1.932 | 0.82× |
| MPC h50 `tw1000` | 4/5 | 4/5 | −0.368 | **+0.018** | 2.247 | 1.95× |

全部失败都是 `corridor_lateral`——**无准备进入旋转终端走廊**。
**加长视界解决不了**：h50 超预算 1.95× 仍在同一 seed 上同样失败，
多完成的那一个靠贴边界换来（裕度薄了 8 倍）。
燃料不是杠杆（1.932 对 1.844，持平）。n=5，正式上报要 20 episodes。

先读 `HANDOFF_COMMIT5B_RESOLVED_20260904.md`。

## 覆盖的文件

| 文件 | 性质 |
|---|---|
| `env/se3_rendezvous_env.py` | **行为修改**：precapture 分支设 `_episode_tumble_scale`（转速缺陷） |
| `controllers/mpc/controller.py` | **行为修改**：`external_local` 参考改用状态自己的坐标（速度帧缺陷） |
| `experiments/evaluate_precapture_oracle.py` | 重写（计划 + 控制律） |
| `experiments/evaluate_mpc.py` | 新增 `--external-guidance oracle_plan`、`--external-hold-steps` |
| `tests/test_precapture_task.py` | 转速回归用例 |
| `tests/test_mpc.py` | 航点接口用例 |
| `docs/PRECAPTURE_TASK_SPEC.md` | V0.2 + 公平基线 |
| `HANDOFF_COMMIT5B_RESOLVED_20260904.md` | 本轮结论 |
| `logs/precapture_planning_oracle/*.json` | 9 份证据（不想入库可删） |

## 或者用补丁

```powershell
Set-Location D:\py\DRL2
git status                    # 干净、HEAD = c564690
Get-ChildItem _patches\*.patch | ForEach-Object { git am $_.FullName }
python -B -m pytest -q        # 168 passed
```

## 验证

```powershell
python -B -m pytest -q
python -B -m experiments.evaluate_precapture_oracle --episodes 5 --seed 262000 --output OUT.json
python -B -m experiments.evaluate_mpc --task precapture_planning --episodes 5 `
    --seed 262000 --horizon 20 --terminal-weight 1000 --reference-source fixed --output OUT_MPC.json
```
