param([switch]$CheckOnly)
$ErrorActionPreference='Stop'
$main='D:\py\DRL2'
$py=Join-Path $main '.venv\Scripts\python.exe'
$statePath='C:\Users\35884\Documents\Spacecraft\过程文件\阶段C\记录\STAGE_C_EXECUTION.json'
$s=Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
if ($s.phase -ne 'await_user_training' -or $s.verdict -ne 'A_GO_TO_E') { throw 'Stage C尚未官方判A且核验交付，禁止启动新训练' }
$head=& git -c "safe.directory=$main" -C $main rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $head.Trim() -ne 'e6694268634be63d5d66a4da637128ac2c5123a1') { throw '版本改变，先交上层核对' }
$dirty=@(& git -c "safe.directory=$main" -C $main status --porcelain --untracked-files=no)
if ($LASTEXITCODE -ne 0 -or $dirty.Count) { throw '工作树不干净' }
$topic='C:\Users\35884\Documents\Spacecraft\过程文件\阶段E训练'
$records=Join-Path $topic '记录'
foreach ($seed in 262430,262431,262432) {
    if ((Test-Path -LiteralPath (Join-Path $main "logs\v3e_$seed")) -or (Test-Path -LiteralPath (Join-Path $records "train_$seed.stdout.log"))) { throw '新种子目录或日志已存在，禁止重复训练' }
}
if (Test-Path -LiteralPath (Join-Path $records 'E_MANUAL_TRAIN_LAUNCH.json')) { throw '训练启动记录已存在' }
if ($CheckOnly) { Write-Output 'E_TRAIN_CHECK_OK: 仅预检；尚未启动'; exit 0 }
New-Item -ItemType Directory -Force -Path $records | Out-Null
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'
$env:PYTHONPATH=$main
$record=[ordered]@{at=(Get-Date -Format o);status='launching';code_commit=$head.Trim();workers=@()}
foreach ($seed in 262430,262431,262432) {
    $p=Start-Process -FilePath $py -WorkingDirectory $main -WindowStyle Hidden -PassThru -ArgumentList '-u','-B','-m','train.train_hybrid','--steps','60000','--seed',"$seed",'--run-name',"v3e_$seed",'--horizon','35','--parametrization','task_state_v3','--adaptive-task','--device','cpu' -RedirectStandardOutput (Join-Path $records "train_$seed.stdout.log") -RedirectStandardError (Join-Path $records "train_$seed.stderr.log")
    $record.workers+=@{seed=$seed;pid=$p.Id}; $record | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $records 'E_MANUAL_TRAIN_LAUNCH.json') -Encoding utf8
    Write-Output "E训练种子$seed PID=$($p.Id)"; Start-Sleep -Seconds 20
}
$record.status='started'; $record | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $records 'E_MANUAL_TRAIN_LAUNCH.json') -Encoding utf8
Write-Output 'E_TRAIN_STARTED: 训练期间不要拉代码；须核对三模型manifest主线配置与训练健康门。'
