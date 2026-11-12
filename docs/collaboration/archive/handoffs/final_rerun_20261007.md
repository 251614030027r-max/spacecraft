# 最终主线重训准备交接（2026-10-07）

状态：ready_for_manual_training；科学提交f2f8169acd580ea96a578d45022dd8952447eca5；执行单19aa467。跟踪工作树干净，物理/控制相对2e5c236差异为空；四组规定测试28 passed in 150.57s，脚本静态语法及CheckOnly核验通过。Codex未启动训练。已选工况w2.36_r15沿用，不重跑筛选。

由用户从独立Windows PowerShell手动启动：过程文件/最终主线重训/脚本/Start-FinalRerun.ps1。新种子262460–262462，各60k，final2_*；value停止规则、soft Bellman继续值。启动脚本拒绝Codex祖先进程、核验冻结SHA、记录PID，关闭插电睡眠/休眠并保存旧设置；进度脚本Show-FinalRerun.ps1只读，不杀进程、不重启、不清理。旧三种子原样保留，登记无效；真实中断时刻未知，最后监督快照16:35，18:52核验所有进程消失。

后续依新执行单完成30k结构门控、271000官方判读、272000描述性扩展、nominal、确定性及反事实重放、完整交付；不新增门槛。此准备不表示后续已完成或已有自动推进监督器。模型和科学原始文件未修改；正式审查分支留待交付。资产盘点在交付后。

旧运行只读登记（SHA/最后检查点/monitor行数）：

```json
{
  "262450": {
    "classification": "INVALID_INTERRUPTED_DO_NOT_USE",
    "directory": "D:\\py\\DRL2\\logs\\final_262450",
    "manifest_original_status": "running",
    "last_checkpoint": "D:\\py\\DRL2\\logs\\final_262450\\checkpoints\\stopping_15000_outer_decisions.zip",
    "checkpoint_sha256": "00ee1a558e216edeb3a78f8651c6ff768411d09237642aed5cbd0511141ef0ec",
    "checkpoint_modified_at": "2026-10-07T16:24:12.534909+09:00",
    "monitor_rows": 160,
    "monitor_sha256": "1f470667e8431b929957303e6bb69f8a7b0e64fa6fec1c673942f65550855e3a",
    "manifest_sha256": "5b8eec81e27234eb726c11cd83795d887e9e9b70c20a1a34f816f85de3da6647",
    "interruption_observed": "2026-10-07T18:52+09:00",
    "last_supervisor_snapshot": "2026-10-07T16:35+09:00"
  },
  "262451": {
    "classification": "INVALID_INTERRUPTED_DO_NOT_USE",
    "directory": "D:\\py\\DRL2\\logs\\final_262451",
    "manifest_original_status": "running",
    "last_checkpoint": "D:\\py\\DRL2\\logs\\final_262451\\checkpoints\\stopping_15000_outer_decisions.zip",
    "checkpoint_sha256": "45060d9dbf99d2500cbbe8ed90995c6ff0d40d5d31c93d14f2d342e3893dcae1",
    "checkpoint_modified_at": "2026-10-07T16:25:33.666903+09:00",
    "monitor_rows": 166,
    "monitor_sha256": "d2508bc964823efab497864adb8d7cb843b0c652f88f3db3547d5856baa5d5de",
    "manifest_sha256": "d82a2510b6bfde9984cc50788a55656b6f6da694b6928e1e336d5f61271853ac",
    "interruption_observed": "2026-10-07T18:52+09:00",
    "last_supervisor_snapshot": "2026-10-07T16:35+09:00"
  },
  "262452": {
    "classification": "INVALID_INTERRUPTED_DO_NOT_USE",
    "directory": "D:\\py\\DRL2\\logs\\final_262452",
    "manifest_original_status": "running",
    "last_checkpoint": "D:\\py\\DRL2\\logs\\final_262452\\checkpoints\\stopping_10000_outer_decisions.zip",
    "checkpoint_sha256": "1a1909814f8452aa4064ded0c4f08178321ead17c2da0787be56ef05abd57877",
    "checkpoint_modified_at": "2026-10-07T14:12:09.607801+09:00",
    "monitor_rows": 153,
    "monitor_sha256": "0ef89aadad19db01a249a764886619cd66680b68b0bcd1d4e32264bacb3b36c8",
    "manifest_sha256": "07176a47d947d2ae4d02392fb5912d5a43fd69d447091e21a38e63e74410d814",
    "interruption_observed": "2026-10-07T18:52+09:00",
    "last_supervisor_snapshot": "2026-10-07T16:35+09:00"
  }
}

```
