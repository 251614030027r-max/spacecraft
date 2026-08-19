# 历史模型资产

本目录是从活动`models/`可逆迁出的模型历史，不代表当前候选：

- 11个旧S1单因素run保留各自`final_model.zip`和周期checkpoint；
- 两个S1-v2目录保存从活动run移出的非代表checkpoint；
- 活动`models/`仅保留空间接近最佳、速度控制最佳及各自final遗忘对照共4个模型。

模型解释必须结合原run的`logs/manifest.json`、周期evaluation和`docs/history/`报告；旧schema不应强行用当前canonical入口加载。所有历史模型都未取得可进入S2的独立20-seed合格结论。
