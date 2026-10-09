# 归档说明

本目录保存 2026-10-10 整理前的全部文档，**内容未作任何修改**，只按研究阶段分组。每个阶段的结论摘要见 `../HISTORY.md`，被否定的方向见 `../DEAD_ENDS.md`。

| 目录 | 阶段 | 篇数 |
|---|---|---|
| `00_single_phase_perception/` | 单阶段任务、A1–A3、G0 感知线 | 7 篇，另含配图子目录 |
| `01_precapture_task_pure_mpc/` | 预捕获任务规格、Pure MPC 时域、终端门、P0–P2 扫描 | 20 |
| `02_coupling_interface_T6_T12/` | 接口单因素实验、T7–T12、非合作探针 | 31 |
| `03_adaptive_round1_v2/` | 自适应任务、第一轮、V2（评估无效，见勘误） | 22 |
| `04_v3_value_arbitration/` | V3b–V3e、价值仲裁 | 24 |
| `05_handoff_stopping_final/` | B/C 阶段、停止头、价值规则、工况筛选 | 14 |
| `handoffs/` | 历代窗口交接与开场提示 | 17 |
| `index_pages_pre_cleanup/` | 整理前的 INDEX、HISTORY、EVIDENCE_INDEX、README、REPRODUCIBILITY | 5 |

## 索引文件

| 文件 | 内容 |
|---|---|
| `PATH_MAP.csv` | 每篇文档的旧路径到新路径 |
| `ARTIFACT_INDEX.csv` | 从工作树移除的 1618 个训练产物（`logs/`、`local_artifacts/`）：路径、blob SHA、大小、取回命令 |
| `CODE_INDEX.csv` | 从工作树移除的 82 个脚本和测试：最后提交、所属阶段、对应实验、取回命令 |

## 取回

```bash
git show archive/pre-cleanup-20261010:<旧路径> > <文件>   # 单个文件
git checkout archive/pre-cleanup-20261010 -- <旧路径>      # 恢复到工作树
```

旧文档中引用的提交 SHA 全部有效，因为历史没有重写。

标签 `archive/pre-cleanup-20261010` 推到远端之前，把命令中的标签名换成提交 `813e27f` 即可，两者指向同一提交。

几个旧 review 分支的证据，保存在标签 `archive/review/...` 下。
