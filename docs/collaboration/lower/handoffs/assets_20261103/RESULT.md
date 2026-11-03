# ASSETS_20261103 事实交付

执行书collab `f2b066c`；科学只读检出 `19278093c3b4bfb5a954280becf68ae15f1b3aec`，远端包含1927809；自检无experiments.t2/info_probe/opp2_probe运行，无训练、评估或测试执行。没有修改科学代码，没有删除任何文件，没有科学分支提交。

## 盘点

6个实际目录，1个可选T2_8_KEYSCORE目录不存在。总文件 21754，总字节 662907234；162个SB3模型ZIP（含保全的门槛冒烟模型，不能当正式训练数）。INVENTORY.json逐目录登记大小、文件数、所有ZIP SHA、B/TRAIN_*原生manifest关键字段、逐例评估JSON及ERROR_*。评估计数包含保全副本，分目录列数，不当独立样本数。

T2.11与科学evidence/T2_11_EACCA/MODEL_SHA256.json的24个ZIP逐项一致；P4使用夜间16与P3使用旧32检查点共48项全部存在。P4 info/opp2原件80份登记SHA。与Git模型证据不一致项：0；盘点首尾源实验文件大小/mtime变化：0；ZIP读取错误：0。具体其他历史ERROR和JSON读取情况以INVENTORY.json分目录事实为准，未清理或抹掉历史故障。

## 归档标签推送回执

```json
{
  "jst": "2026-11-03T11:25:54.176746+09:00",
  "status": "pushed_verified",
  "tags": [
    {
      "name": "archive/t2-11-eacca-code-20261103",
      "target": "ec394efcaaeadbb5305f4f9e289addb58517dca7",
      "object": "4d60f32bc2fe4c23c5c54c2a4e08f7986e58692d",
      "existing_remote_before": "",
      "reachable_from_science_remote": true,
      "remote_after": "4d60f32bc2fe4c23c5c54c2a4e08f7986e58692d\trefs/tags/archive/t2-11-eacca-code-20261103\nec394efcaaeadbb5305f4f9e289addb58517dca7\trefs/tags/archive/t2-11-eacca-code-20261103^{}"
    },
    {
      "name": "archive/t2-11-eacca-evidence-20261103",
      "target": "5e42c47032d4e6811451a6794f6254fe9fde2063",
      "object": "45e45552ea648a5ee9f5a1f4ff99596618260e9f",
      "existing_remote_before": "",
      "reachable_from_science_remote": true,
      "remote_after": "45e45552ea648a5ee9f5a1f4ff99596618260e9f\trefs/tags/archive/t2-11-eacca-evidence-20261103\n5e42c47032d4e6811451a6794f6254fe9fde2063\trefs/tags/archive/t2-11-eacca-evidence-20261103^{}"
    }
  ],
  "push_stdout": "",
  "science_branch_unchanged": true
}
```

两个目标均由rev-parse核对并可从科学远端到达；标签不存在时创建annotated tag，已有同名标签只核对，不移动或删除。科学分支头保持原样。

## 保留、入口与本机说明

所有实验目录、模型、P4原件、TensorBoard与30秒日志原位保留，未压缩或去重。未来清理候选仅登记于INVENTORY，不代表删除批准。

上层交付/最新仅保留未改字节的T2.11 RESULT.md和README导航；894份配套文件迁入`C:\Users\35884\Documents\Spacecraft\上层交付\历史\T211_SUPPORT_before_ASSETS_20261103_112554`，逐文件SHA一致。没有把资产报告冒充新的科学验证。项目说明与AGENTS已按CLAUDE入口表加入EACC-A、e_acc/e_sac、reduced_priv、评估读取manifest、68项冻结测试基准及“正式实验需用户单独批准”；原文备份在本轮输入目录。

本机文档哈希：
```json
{
  "C:\\Users\\35884\\Documents\\Spacecraft\\项目说明.md": {
    "before_SHA256": "bab8fcdbe0d91ec68ad3b91cdf2e061850d1dcda048824f7e9eb512ca5b1ca2d",
    "after_SHA256": "f334b7032972e065b06caaf8ec42ff80d4f5ee82ff5db31fe737f8e62313aeec"
  },
  "C:\\Users\\35884\\Documents\\Spacecraft\\AGENTS.md": {
    "before_SHA256": "889557339b0e698ea3c68f67a3edb942c30c5a56fd25ffc922c5ef0b1e705233",
    "after_SHA256": "ae7b91b9818982d97d484cb36a955884c8098a2815a64cc2c65727cd5b4dde3d"
  }
}
```

本次原始权限拒绝日志保留在记录/inventory.log；经只读权限恢复执行同一盘点脚本，成功回执为记录/inventory_attempt1.log。全程未运行科学。
