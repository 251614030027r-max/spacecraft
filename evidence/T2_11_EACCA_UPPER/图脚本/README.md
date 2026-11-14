# 图脚本草稿（上层沙箱，2026-11-01 至 11-03）

- `fig_a.py`：学习曲线。
  - 输入：`t2_readout curve` 输出的 READOUT.json。
  - 两个面板：平均回报、紧急捕获数。细线为单条训练，粗线为臂均值。
  - 用法：`python fig_a.py READOUT.json OUT.png 标题`
- `fig_b.py`：关键状态因果图。
  - 输入：关键状态目录与 `t2_keyscore` 输出。
  - 内容：critic 对可分候选对的判对率、策略选择、示例状态的 3 × 3 真值 / 标签 / critic 对照。
  - 示例状态按事先规则选取。脚本目前只取标签含 `_ng_` 的检查点（T2.8 命名），换数据时要改这一处筛选。
  - 用法：`python fig_b.py STATES_DIR KEYSCORE.json OUT.png [步数]`
- **配色：** EACC 系蓝 `#2a78d6`，SAC 橙 `#eb6834`。中文字体按可用字体自动选择。
- **定位：** 草稿，不是论文终稿。正式出图以正式实验的数据为准。
