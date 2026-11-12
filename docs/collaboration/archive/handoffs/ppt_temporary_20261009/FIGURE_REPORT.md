# 最新汇报图表与科学审核

本版替换原30k图片。科学提交 f2f8169acd580ea96a578d45022dd8952447eca5；工况w2.36_r15。三种子训练曲线来自追加日志的独立完整行快照，累计episode长度表示外层决策数，全部展示至各自当前进度，横轴预留到60k，绝不延长数据。原始回报浅线，50回合尾随均值；完成率也是50回合窗口，任务完成不等同零违规，不宣称已收敛。

轨迹使用262460、262462的冻结50k模型，在已固定266019开局各做一次确定性重放；没有换案例或额外48开局评估。Pure复用此前同提交同开局已核验物理数据；两新模型采用原科学环境、加载器和官方决策函数，以低优先级单线程逐一重放。仅给当前重放实例加只读采样，真实0.1s数据，不平滑，不改控制。50k没有对应原开发JSON，不能套用30k的k30/143.6s结论；本版依据原评估函数返回、物理终点、相同初态、首次价值阈值与模型前后SHA核验。详细检查见plot_data。

三维轨迹是目标本体系，弧线含目标旋转效应，不能直接解释为更优路径。预捕获完成不是物理接触。价值越零只支持部署规则执行，不证明价值已校准或交接贡献。没有learned-only同模型同块对照，不能宣称协调增益；单案例也不能代替预注册正式性能。30k结构统计保留在历史版，不放进本版冒充50k结构验证。

## 资产与结果

```json
{
  "models": {
    "262460": {
      "checkpoint": "D:\\py\\DRL2\\logs\\final2_262460\\checkpoints\\stopping_50000_outer_decisions.zip",
      "sha256": "0ce77af635f8a314c7d9cba7b62dec7f32de0f28831bf52576f55c287df2e036",
      "outer_decisions": 50000,
      "checks": {
        "same_initial_condition": true,
        "physical_time_matches_result": true,
        "checkpoint_unchanged": true,
        "first_value_threshold_matches_handoff": true
      }
    },
    "262462": {
      "checkpoint": "D:\\py\\DRL2\\logs\\final2_262462\\checkpoints\\stopping_50000_outer_decisions.zip",
      "sha256": "5a343ca15206ad363b8e8a269009a159fef42ab1b6867e86bbe5d123c07c58aa",
      "outer_decisions": 50000,
      "checks": {
        "same_initial_condition": true,
        "physical_time_matches_result": true,
        "checkpoint_unchanged": true,
        "first_value_threshold_matches_handoff": true
      }
    }
  },
  "training_sources": {
    "262460": {
      "source": "D:\\py\\DRL2\\logs\\final2_262460\\train.monitor.csv",
      "snapshot_sha256": "c3b11a771e29820c07c1fe423921bea5f80248779d225ed6f02fe7f03b67d869",
      "episodes_shown": 536,
      "last_outer_decision": 53388
    },
    "262461": {
      "source": "D:\\py\\DRL2\\logs\\final2_262461\\train.monitor.csv",
      "snapshot_sha256": "4da01f6dd40a5e2a9d7b449cfcebdbcc4d63e1de395c5589b0f755d2fc9e2f30",
      "episodes_shown": 531,
      "last_outer_decision": 49945
    },
    "262462": {
      "source": "D:\\py\\DRL2\\logs\\final2_262462\\train.monitor.csv",
      "snapshot_sha256": "96251961d2fb673a6a2d3fde5f34586feacd7a3a4f7256483b59097a50d1527d",
      "episodes_shown": 588,
      "last_outer_decision": 52348
    }
  },
  "results": {
    "pure": {
      "completed": true,
      "zero_violation": true,
      "clean_completion": true,
      "survival_s": 106.80000000000001,
      "equivalent_delta_v_m_s": 1.3405265577779548,
      "handoff_k": 0,
      "final_position_error_m": 0.21529943273877916,
      "final_attitude_error_deg": 7.312941235712258,
      "minimum_keepout_margin_m": 1.1817931298171858,
      "minimum_fov_margin_deg": 32.08756213394502
    },
    "262460": {
      "completed": true,
      "zero_violation": true,
      "clean_completion": true,
      "survival_s": 133.9,
      "equivalent_delta_v_m_s": 1.7196381784351449,
      "handoff_k": 21,
      "final_position_error_m": 0.20607231652299246,
      "final_attitude_error_deg": 9.016608630735641,
      "minimum_keepout_margin_m": 1.1992462630887935,
      "minimum_fov_margin_deg": 35.314823956243856
    },
    "262462": {
      "completed": true,
      "zero_violation": true,
      "clean_completion": true,
      "survival_s": 161.10000000000002,
      "equivalent_delta_v_m_s": 1.9805159539932125,
      "handoff_k": 39,
      "final_position_error_m": 0.21283292642859525,
      "final_attitude_error_deg": 5.1015274634416485,
      "minimum_keepout_margin_m": 1.21225368525596,
      "minimum_fov_margin_deg": 33.036218244243074
    }
  }
}
```

所有PNG400dpi，另附PDF/SVG。脚本和快照随版保存。训练进程、主工程及正式结果未改。


## 展示目录调整

全部图表重新审核见[TRAINING_AND_FIGURE_REVIEW.md](TRAINING_AND_FIGURE_REVIEW.md)。训练曲线已移入诊断材料，约束图移入支撑材料，重复双图归档；当前入口只保留轨迹、状态响应和价值触发三张，不宣称已学好或性能优势。文件名称与哈希位置以README和FILES_SHA256为准。
