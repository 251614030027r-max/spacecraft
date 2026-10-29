# GIT_CLEANUP_20261029：A/B完成；C1/C2未执行

依据执行书collab `6bb1d664fb638901c5398f930048a3b20fbc3e01`，仅普通新提交，不改写历史、不强推、不改既有标签、未触碰main。没有训练或评估；pytest为代码验收。

## 保全与提交

- collab保全标签：`archive/collab-pre-cleanup-20261029`；annotated tag对象`ccfc02467aceb5ad0e64b6455aadb1f346686b01`，实际指向`6bb1d664fb638901c5398f930048a3b20fbc3e01`，远端已推送核验。
- `archive/science-pre-cleanup-20261029`：**未创建**。执行书将它作为可选C1删除前置条件，C1尚未获逐项确认；B不删任何科学资产，不冒充已执行第二个标签步骤。
- A1删除旧副本提交：`fceb72e58def49ce6a18e0b99e0ecd362908854d`。
- A2七目录迁移提交：`09a4063a113d52a60a4babcde1760c4e03da042e`。
- A3现场状态提交：`7ac53022edb6f75ad5c16d1516cd25d27b5c2d70`。
- B科学轻量提交：`05bf44a0b2fc0b9281c32654528bfb9ec409ed7f`，基于`8761ff7dddb924a9ceb41e39287c4303a45b7f5b`。本报告与总账收尾为后续独立collab提交；最终推送SHA在本机PUSH_RECEIPT.json中记录，避免自引用。

## 最新Git树统计（不含.git对象库、虚拟环境或忽略的本机资产）

|分支|整理前文件数|整理前字节|整理后文件数|整理后字节|
|---|---:|---:|---:|---:|
|collab|3747|1947706081|1658|139658647|
|science|9531|664545837|9542|664574582|

collab仅保留`docs/collaboration/`，加执行书明确保留的根README指针和.gitignore。删除2095个旧文件，原字节1808506857；每项路径、blob、大小、科学或标签恢复位置见REMOVED.csv。三类旧图目录均已在科学archive找到相同blob，补迁移独有文件0项。克隆历史体积不会因此变小，没有宣称Git对象库释放空间。

## 交付路径与现场

七个T2目录共110文件使用git mv统一进入docs/collaboration/lower/handoffs；全部新旧blob相同，原报告/交付包内部内容与SHA没有重写。HANDOFF、PROJECT_STATE、README、研究演进及lower入口已更新；已执行书不改，旧→新路径表在lower/README.md及MOVED.csv。

CURRENT已同步T2.6W完成关闭、768评估+336参照、science605f3e4/collab4ad969a、训练/评估均false、自动跟进关闭、本地资产清理完毕。T2_6W1执行书仅待用户批准，没有开跑。保留旧CURRENT快照。registry只追加T2.2、T2.3、T2.4、T2.5、T2.5.1、T2.6、T2.6W七轮下层事实，既有行不变，上层审查/论文角色/主张列不代填。

## 科学侧B：保留ZIP与原始数据

实际evidence内ZIP为40项（执行书估计约35），均不删，原blob与大小全相同，补各目录SHA与docs/archive/MODEL_ZIP_EXCEPTIONS_20261029.json，在GIT_ASSETS中登记历史例外；以后新模型只记SHA。未来过程资料目录统一脚本/记录/输入，不批量改写既有证据目录。复用SAC评估副本保留。

**C1/C2均未执行：** T1-B逐决策/模型发布日志保留；collab T1历史交付ZIP保留（本次逐一核验7个既有ZIP）。上层自己的DEAD_ENDS/HISTORY/T2证据总表/docs-current注记不由下层代做。

## 自检

- 最新科学源`8761ff7dddb924a9ceb41e39287c4303a45b7f5b`导出到D:/py/DRL2内部测试快照（不是新克隆），主科学检出HEAD和工作树未改；B只新增清单/文档、未改科学代码或旧证据。
- `python -B -m pytest -q tests/test_t2_stage.py`：**46 passed，1 warning，原生退出码0**。pytest自身报告103.90s，外部完整wall 107.26s；唯一warning为测试中requires_grad tensor转标量。268个导出源文件SHA前后不变。pytest.log与VALIDATION随附。
- 现行五份入口Markdown相对文件链接40项均存在，无缺失目标；只读核验路径，不宣称联网验证外部URL可达性。已执行书41项blob全不变。
- 模型ZIP40项、C1逐决策/发布日志19项、C2历史ZIP7项均核验原blob保留。

本次未授权任何实验；等待上层与用户后续指令。
