# 实测吞吐口径

快照UTC：2026-10-18T23:52:29.444619+00:00。每进程BLAS/OMP/MKL为1线程；证据任务按约12个科学worker的资源上限排队。

以下为并行、共享本机负载条件下的单回合 elapsed wall_s；Σworker wall_s不是OS CPU秒，也不是整批历时。不得称正式串行性能基准。

|阶段|完整回合|真实基本决策步|Σworker wall_s|均值wall_s/回合|
|---|---:|---:|---:|---:|
|deadline_v1_probe|16|1273|3743.587|233.974|
|mission_v3_confirmation|56|3497|8668.723|154.799|
|mission_v3_discovery|30|2128|7788.285|259.609|
|mission_v3_refined_confirmation|48|3089|6427.507|133.906|
|t1b_v0|48|3490|9536.773|198.683|
|t1b_v0_strong|36|2538|7978.218|221.617|
|t1b_v0_zero|12|836|2519.800|209.983|
|t1b_v1_confirmation|80|5276|17761.083|222.014|
|t1b_v1_confirmation_extra_times|16|1444|4734.116|295.882|
|t1b_v1_discovery|96|6789|21816.950|227.260|
|t1b_v1_extra_times|24|2169|7509.318|312.888|
