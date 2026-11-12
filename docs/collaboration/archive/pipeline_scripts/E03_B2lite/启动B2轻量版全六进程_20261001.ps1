param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'

$repo = 'D:\py\DRL2'
$py = 'D:\py\DRL2\.venv\Scripts\python.exe'
$records = 'C:\Users\35884\Documents\Spacecraft\过程文件\阶段B2\记录'
$launchRecord = Join-Path $records 'B2_LITE_MANUAL_SIX_LAUNCH_20261001.json'
$oldFirstWave = Join-Path $records 'B2_LITE_4B_FIRST_WAVE_20261001.json'
$b1State = 'C:\Users\35884\Documents\Spacecraft\过程文件\阶段B1\记录\STAGE_B1_EXECUTION_20260930.json'
$expectedCommit = 'a714c61cc3327525ef6c103ecfcb9dc46e552a07'

if ((Test-Path -LiteralPath $launchRecord) -or (Test-Path -LiteralPath $oldFirstWave)) {
    throw 'B2 启动记录已存在；不要重复启动。'
}
$head = & git -c "safe.directory=$repo" -C $repo rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $head.Trim() -ne $expectedCommit) { throw 'Git 提交不是 B1 固定版本；不要启动。' }
$trackedChanges = @(& git -c "safe.directory=$repo" -C $repo status --porcelain --untracked-files=no)
if ($LASTEXITCODE -ne 0 -or $trackedChanges.Count -ne 0) { throw '已跟踪工作树不干净；不要启动。' }
$state = Get-Content -LiteralPath $b1State -Raw | ConvertFrom-Json
if ($state.status -ne 'b1_block_complete_wait_user') { throw 'B1 尚未按 4A 完整收口。' }
foreach ($model in @('262420', '262421', '262422')) {
    $b1Dir = Join-Path $repo "eval\v3e\stage_b\$model"
    if (@(Get-ChildItem -LiteralPath $b1Dir -Filter 'seed_262*.json' -File).Count -ne 48) {
        throw "$model 的 B1 262000 块不足 48/48。"
    }
    if (@(Get-ChildItem -LiteralPath $b1Dir -Filter '*.lock' -File).Count -ne 0) {
        throw "$model 的 B1 目录仍有活动锁。"
    }
    $b2Dir = Join-Path $repo "eval\v3e\stage_b2\$model"
    if ((Test-Path -LiteralPath $b2Dir) -and @(Get-ChildItem -LiteralPath $b2Dir -Force).Count -gt 0) {
        throw "$model 的 B2 输出目录已有文件；不要重复启动。"
    }
}
foreach ($oldPid in @(23096, 28608, 24224, 8736, 15204, 6872, 10560, 28292, 25384, 19544, 21464, 3412)) {
    if (Get-Process -Id $oldPid -ErrorAction SilentlyContinue) { throw "旧 B1 进程 $oldPid 仍存在，请先核对。" }
}
if ($CheckOnly) {
    Write-Output 'CHECK_OK: B1 已 48/48/48 收口，旧提交干净，可以手动启动六个 B2-lite 进程。'
    exit 0
}

New-Item -ItemType Directory -Force -Path $records | Out-Null
foreach ($model in @('262420', '262421', '262422')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $repo "eval\v3e\stage_b2\$model") | Out-Null
}
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$record = [ordered]@{status='launching';at=(Get-Date -Format o);code_commit=$expectedCommit;launched=@()}
$record | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $launchRecord -Encoding utf8

foreach ($model in @('262420', '262421', '262422')) {
    for ($index=1; $index -le 2; $index++) {
        $name = "b2_${model}_$index"
        $stdout = Join-Path $records "$name.stdout.log"
        $stderr = Join-Path $records "$name.stderr.log"
        if ((Test-Path -LiteralPath $stdout) -or (Test-Path -LiteralPath $stderr)) {
            throw "日志 $name 已存在；保留已启动进程并报告，不要重复运行。"
        }
        $args = @('-u','-B','-m','experiments.v3_handoff_scan',
            '--run-dir',"logs/v3e_$model",'--seeds','262000-262047',
            '--scan','successes','--stride','10','--verify-prefix','middle',
            '--output-dir',"eval/v3e/stage_b2/$model")
        $process = Start-Process -FilePath $py -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
            -ArgumentList $args -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $record['launched'] += [ordered]@{name=$name;model=$model;wrapper_pid=$process.Id;stdout=$stdout;stderr=$stderr}
        $record | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $launchRecord -Encoding utf8
        Write-Output "$name started: PID $($process.Id)"
        Start-Sleep -Seconds 20
        $process.Refresh()
        if ($process.HasExited -and $process.ExitCode -ne 0) {
            $record['status'] = 'launch_failed'
            $record | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $launchRecord -Encoding utf8
            throw "$name 提前退出，退出码 $($process.ExitCode)；查看错误日志，不要重复运行脚本。"
        }
    }
}
$record['status']='started'
$record['completed_at']=Get-Date -Format o
$record | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $launchRecord -Encoding utf8
Write-Output 'B2_LITE_SIX_STARTED; B1 判定前只看 B2 文件个数，不读结果内容。'
