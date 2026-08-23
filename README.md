# DRL2：六自由度翻滚目标预捕获控制

高保真 SE(3) 六自由度预捕获控制,目标为自由翻滚的非合作航天器。真值传播含 J2、
引力梯度、二阶矩与刚体耦合,RK45 积分,0.1 s 控制周期。

研究主线是三方对照:**Pure SAC / Pure MPC / SAC-MPC 分层混合**,共同基准是单相位
的 `gate_free` 任务——十到十四米起始,五条任务约束全程激活,一条控制律跑到底。

## 唯一权威文档

**`CLAUDE.md`** 是这个仓库的交接文档:研究主线、已定结论、实测参考数值、纪律规则
和陷阱全在里面。开始任何工作前先读它。

`docs/history/` 下是历史材料,记录当时的判断,其中不少已被后续实测推翻,**不能用来
支持任何当前主张**。

## 快速开始

```bash
python -B -m pytest -q                       # 仓库无 conftest.py,必须这样跑
python -B -m eval.validate_gatefree_semantics --output logs/gatefree_validation/scripted.json
python -B -m train.train --steps 400000 --seed <seed> --run-name <name> --mode gate_free
python -B -m eval.digest_run --run logs/<name>
```

入口清单见 `CLAUDE.md` 的 *Trusted entry points*。

## 目录

活动源码只在 `dynamics/`、`env/`、`controllers/`、`train/`、`eval/`、`experiments/`。
`References/` 是本工作对标的文献,是方法的一部分。`models/` 刻意不入版本控制——
checkpoint 可由 manifest 加种子复现。
