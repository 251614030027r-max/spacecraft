# 论文稿件

本目录只放论文稿件及其直接附属文件，由写作窗口维护。

| 路径 | 内容 |
|---|---|
| `outline.md` | 目标期刊、语言、章节结构与每节的事实来源 |
| `draft/` | 正文（LaTeX 或 Markdown，按目标期刊模板） |
| `refs.bib` | 参考文献；每条都有可核对的 DOI 或正式出处 |
| `literature/notes.md` | 文献调研笔记：每篇读了什么、用于哪一节、写作规范与措辞参考 |
| `figures/` | 只放由 `experiments/formal_figures.py` 生成的正式图，或注明来源的示意图 |

## 写作规则

1. **事实来源唯一：**
   - 方法：`docs/paper/METHOD.md`（符号、命题、伪代码、"不声称"清单）；
   - 论文对象与源码的对应：`docs/paper/FACT_MAP.md`；
   - 实验设计：`docs/paper/EXPERIMENTS.md`；
   - 背景、命名与贡献：`docs/paper/BASELINE.md` 第 0–3 节；
   - 历史与负结果：`docs/HISTORY.md`、`docs/DEAD_ENDS.md`。

   稿件与这些文件冲突时，以它们为准；认为它们有误时，提给上层修改，不在稿件里自行改口径。
2. **数字：**
   - 正式结果只取自 `evidence/FORMAL/`（正式实验交付后才有）的 `FORMAL_READOUT.json`、`EPISODES.csv`、图表；
   - 交付前，结果位置一律写占位符 `{{FORMAL:键名}}`，不填估计值；
   - 开发期数字（B 步等）只能作为"开发期"出现，注明证据目录，不并入正式样本。
3. **不声称：** 遵守 `METHOD.md` 的"不声称"清单与 `EXPERIMENTS.md` 的写法注意。尤其注意以下几点：
   - 0 次不安全只写"本测试样本中未观察到"；
   - 训练不写"逐位可复现"（多进程异步）；
   - U 与 E 不称单参数对照；
   - 容量混杂如实说明。
4. **文献：**
   - 只引用读过并核对过出处的文献，不编造引用；
   - `References/` 里的 PDF 与 `References/README.md`（每篇的用途与不要声称的内容）优先；
   - 新增文献写进 `refs.bib` 与 `literature/notes.md`。
5. **改动范围：** 写作窗口只改本目录。代码、注册表、证据、`docs/paper/` 与 `docs/final_method/` 只读；需要改时提给上层。
6. **Git：**
   - 只在科学分支 `claude/sac-mpc-coupling-design-ns7g6i` 上提交，提交前先 fetch 并 rebase，不强推；
   - 提交末尾附协作尾注；
   - 下层运行正式实验期间不影响其检出（下层不拉取）。
