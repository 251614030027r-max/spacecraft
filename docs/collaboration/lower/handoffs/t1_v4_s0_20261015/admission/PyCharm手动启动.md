> **准入及校准已通过：λ_mc=0.03，λ_eq=0.1，校准2004决策正常结束。正式M0/M2尚未启动；直接从下面终端一、终端二开始，不要再次训练校准。**

# T1 v4：PyCharm手动启动与滚动监视

执行单第2.1版875c39c；科学提交1d432c8aa0e26e31354b8d43771e2a45c8d1caf5。已删除本轮自动编排脚本及定时跟进。正式M0/M2尚未启动，所有命令由用户在PyCharm现有PowerShell终端手动执行，不创建外部窗口。

六套规则全部完成、96/96捕获；强固定参照κ1.0、入口速度0.02，回报82.469507625659、Δv1.300345495748704、时间116.1125秒。指定71测试、配置、同机C通过。2k校准已正常完成，冻结λ如上。

## 0. 校准已完成，无需重做

401个有效MC探针；MC/TD梯度比中位3.7789574629，按预注册选择λ_mc=0.03，λ_eq固定0.1，组合占比0.1125878182。2004决策为6环境跨过2000的正常计数。原1104中断数据已归档，本次重做由用户明确授权。

开训前可在任一终端执行防睡眠设置：

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```

## 1. 终端一：M0，前台运行，不要关闭终端

```powershell
Set-Location D:\py\DRL2_T1
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:PYTHONUNBUFFERED='1'
$c=Get-Content 'logs\t1\_v4\calibration.json' -Raw | ConvertFrom-Json
if ($c.PSObject.Properties.Name -contains 'defect' -or $c.frozen.lambda_mc -notin @(0.03,0.1,0.3) -or $c.frozen.lambda_eq -ne 0.1) { throw '校准未通过' }
if (Test-Path 'logs\t1\t1v4_m0_s0') { throw 'M0目录已存在，先核查，禁止重跑' }
& 'D:\py\DRL2\.venv\Scripts\python.exe' -B -m train.train_t1 --method m0 --run-name t1v4_m0_s0 --steps 40000 --seed 0 --n-envs 6 --log-root logs/t1 --checkpoint-freq 5000 --extra-checkpoints 2000 --lambda-mc $c.frozen.lambda_mc --lambda-eq $c.frozen.lambda_eq --audit-every 100 2>&1 | Tee-Object -FilePath 'logs\t1\_v4\m0_manual_console.txt'
```

## 2. 终端二：M2，与M0同时运行

```powershell
Set-Location D:\py\DRL2_T1
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:PYTHONUNBUFFERED='1'
$c=Get-Content 'logs\t1\_v4\calibration.json' -Raw | ConvertFrom-Json
if ($c.PSObject.Properties.Name -contains 'defect' -or $c.frozen.lambda_mc -notin @(0.03,0.1,0.3) -or $c.frozen.lambda_eq -ne 0.1) { throw '校准未通过' }
if (Test-Path 'logs\t1\t1v4_m2_s0') { throw 'M2目录已存在，先核查，禁止重跑' }
& 'D:\py\DRL2\.venv\Scripts\python.exe' -B -m train.train_t1 --method m2 --run-name t1v4_m2_s0 --steps 40000 --seed 0 --n-envs 6 --log-root logs/t1 --checkpoint-freq 5000 --extra-checkpoints 2000 --lambda-mc $c.frozen.lambda_mc --lambda-eq $c.frozen.lambda_eq --audit-every 100 2>&1 | Tee-Object -FilePath 'logs\t1\_v4\m2_manual_console.txt'
```

## 3. 终端三、四：开发评估（分别各运行一条）

每个新终端先执行环境设置，然后选择M0或M2对应的一条命令。

```powershell
Set-Location D:\py\DRL2_T1
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:PYTHONUNBUFFERED='1'
```

终端三：

```powershell
& 'D:\py\DRL2\.venv\Scripts\python.exe' -B -m experiments.t1_dev_eval --watch logs/t1/t1v4_m0_s0 --openings experiments/t1_dev_openings_295000.json --checkpoints 0,2000,5000,10000,15000,20000,25000,30000,35000,40000 --workers 1 2>&1 | Tee-Object -FilePath 'logs\t1\_v4\watch_m0_manual_console.txt'
```

终端四：

```powershell
& 'D:\py\DRL2\.venv\Scripts\python.exe' -B -m experiments.t1_dev_eval --watch logs/t1/t1v4_m2_s0 --openings experiments/t1_dev_openings_295000.json --checkpoints 0,2000,5000,10000,15000,20000,25000,30000,35000,40000 --workers 1 2>&1 | Tee-Object -FilePath 'logs\t1\_v4\watch_m2_manual_console.txt'
```

不加`--stop-on-plateau`。两个watch会依次评估已生成的检查点，每点16个开局；允许落后于训练。

## 4. 终端五：两条训练进度滚动显示

冻结训练器verbose=0，不逐步打印。下面手动监视心跳，每当决策数变化便输出一行；Ctrl+C仅退出此监视终端，不影响另两个训练终端。

```powershell
Set-Location D:\py\DRL2_T1
$seen=@{}
while ($true) {
  foreach ($m in @('m0','m2')) {
    $p="logs\t1\t1v4_${m}_s0\heartbeat.json"
    if (Test-Path $p) {
      try {
        $h=Get-Content $p -Raw | ConvertFrom-Json
        if ($seen[$m] -ne $h.decisions) {
          $seen[$m]=$h.decisions
          '{0:HH:mm:ss}  {1}  {2}/40000  有限={3}  平均={4:N3}s/决策' -f (Get-Date),$m,$h.decisions,$h.parameters_finite,($h.wall_s/[Math]::Max(1,$h.decisions))
        }
      } catch {}
    }
  }
  Start-Sleep -Seconds 5
}
```

若要逐决策原始日志，在另一个终端任选一条（文件生成后执行）：

```powershell
Get-Content 'D:\py\DRL2_T1\logs\t1\t1v4_m0_s0\decisions.jsonl' -Tail 3 -Wait
# M2把路径中的m0改成m2
```

## 5. 10k技术门与交付节点

自动编排已依用户要求删除，因此不会再自动执行10k技术检查、20k打包或最终打包。M2写出10k检查点后需执行：

```powershell
Set-Location D:\py\DRL2_T1
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'
& 'D:\py\DRL2\.venv\Scripts\python.exe' -B -m experiments.t1_fix_readout --m0 logs/t1/t1v4_m0_s0 --m2 logs/t1/t1v4_m2_s0 --output logs/t1/_v4/cf_check_10k.json
if ($LASTEXITCODE -ne 0) { throw '10k读出失败，先核查，不据此停训练' }
$q=Get-Content 'logs\t1\_v4\cf_check_10k.json' -Raw | ConvertFrom-Json
$valid=$q.curve4.m2.cf_by_kind_total.entry_speed.gate_valid
if ($null -eq $valid -or $valid -eq 0) {
  '{"rule":"T1_FIX r2 sec. 6: entry-speed counterfactual never valid by 10k"}' | Set-Content 'logs\t1\t1v4_m2_s0\STOP_REQUESTED' -Encoding ascii
  '技术门未通过：仅请求M2干净停止，M0继续；保留结果并报告'
} else { "入口速度反事实技术门通过，有效样本=$valid" }
```

两条20k评估完成后由下层只读快照交付；40k及全部评估结束后下层完整交付。需回到本聊天通知节点，当前没有定时自动跟进。不要按成绩手动停止、不要续训或重跑，也不要在运行时切换科学树。
