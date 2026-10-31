# T2.5验证：mechanism1000

生成2026-10-26T18:25:03.224915+09:00；科学`84ee9c1045f32bbbacfef6cd220fdad55c5765c8`；执行书collab`4c52ece`；33tests本机通过。双EACC(e_mve)种子0/1各4000、各3异步worker；SAC复用T2.3及T2.4补评估共192，不重训。只验证并交付，不运行任何后续分支。
滚动30秒准确步数/更新/回合，真实与训练回报、成功率、critic/min-Q、反馈探针；外部观察器不新增科学forward/backward/RNG，入场验收见LOGGING_VERIFIED。独立隐藏Windows后台、全CPU/BLAS1。Q1/Q2分别统计冻结源未提供，标MISSING，不捏造。

## 1000步机制早报（非最终判定）
统计窗口严格在1000之后，原生探针经事件文件读取，各字段列明实际样本数和首末步数，不以目录存在或合成数据代替。decisive_share为0或two_stage_share低于0.5即列警报；按书只报告、不停止训练。
```json
{
  "jst": "2026-10-26T18:25:02.792337+09:00",
  "source": "Actual native probes parsed from external TensorBoard; rows strictly after1000; provisional,not final criteria",
  "readings": {
    "e_mve_s0": {
      "two_stage_share": {
        "mean": 0.7299040101823353,
        "n": 21,
        "first_step": 1012,
        "last_step": 1262,
        "missing": false
      },
      "decisive_share": {
        "mean": 0.15348440266790844,
        "n": 21,
        "first_step": 1012,
        "last_step": 1262,
        "missing": false
      },
      "c_mean": {
        "mean": 0.14840936731724513,
        "n": 21,
        "first_step": 1012,
        "last_step": 1262,
        "missing": false
      },
      "grad_cos": {
        "mean": 0.1847626894712448,
        "n": 5,
        "first_step": 1050,
        "last_step": 1250,
        "missing": false
      }
    },
    "e_mve_s1": {
      "two_stage_share": {
        "mean": 0.7340984642505646,
        "n": 4,
        "first_step": 1012,
        "last_step": 1050,
        "missing": false
      },
      "decisive_share": {
        "mean": 0.14234961941838264,
        "n": 4,
        "first_step": 1012,
        "last_step": 1050,
        "missing": false
      },
      "c_mean": {
        "mean": 0.17136888578534126,
        "n": 4,
        "first_step": 1012,
        "last_step": 1050,
        "missing": false
      },
      "grad_cos": {
        "mean": 0.7442177534103394,
        "n": 1,
        "first_step": 1050,
        "last_step": 1050,
        "missing": false
      }
    }
  },
  "immediate_alert_seeds": []
}
```

## 历史对照、异常与墙钟
T2.4固定science632b633：MVE种子0的4k回报53.792454、种子1为55.238621，均捕获23/24、unsafe0；种子0相对SAC54.476604差−0.684149。T2.5原生v25按执行书阈值53.79，事后不改。T2.3对照来源、逐文件SHA和模型SHA随附，旧SAC恢复事件字段若缺失应标MISSING，不当0。
两个训练finished原件包含实际wall_s和worker_order，滚动吞吐与ETA区分；并行墙钟不相加。训练真实技术崩溃同配置从零最多一次且保全旧attempt，不拼Replay/RNG/优化器；纯发布失败只补报告，不重跑科学。可选T2.4早期曲线要求另一SHA，为不影响本轮先暂缓，不混版本执行。
模型ZIP/Replay/TB二进制本机保留，Git发布432必要原始评估、6审计、训练JSON与外部滚动日志/事件解析和验收。没有自动长训或后续分支，等待上层复核与用户批准。
