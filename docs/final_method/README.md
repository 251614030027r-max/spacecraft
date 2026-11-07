# EACH 最终方法与实验资料

| 文件 | 内容 |
|---|---|
| `METHOD_MINIMAL.md` | 六步最小闭环、模块分类、信息边界、默认开关、伪代码 |
| `METHOD_SPEC.md` | 宏观决策与真实 TD（T1）、C1 执行器、U1 理论（T2）、学习与评估协议、参数表；逐式对应源码 |
| `C1_THEORY.md` | 扫掠锥入口机会集（T3）、窗口承诺可行集包含（T4）、16 s 阶段论证 |
| `ABLATION_MATRIX.md` | 训练组与无需训练的组、正式实验目标矩阵、历史开发证据 |
| `FIGURE_SPEC.md` | 每张图的数据来源、坐标、分组与统计口径 |
| `CONFIG_LOCK.md` | 冻结配置、与 `2e7d838` 的等价性审计、阶段与开局、依赖 |
| `METHOD_REGISTRY.json` | 展示名 ↔ 内部 arm ↔ 开关 ↔ 阶段、种子、开局；`experiments/final.py` 与 `experiments/final_figures.py` 的唯一配置来源 |

执行书在 collab `docs/collaboration/upper/run_orders/`：`RUN_PRETEST.md`（预实验）与 `RUN_FORMAL.md`（正式实验草案，未授权）。证据目录：`evidence/final_pretest/`、`evidence/final_formal/`。
