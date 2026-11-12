param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'

$repo = 'D:\py\DRL2'
$py = 'D:\py\DRL2\.venv\Scripts\python.exe'
$records = 'C:\Users\35884\Documents\Spacecraft\过程文件\阶段B2\记录'
$launchRecord = Join-Path $records 'B2_LITE_4B_FIRST_WAVE_20261001.json'
$expectedCommit = 'a714c61cc3327525ef6c103ecfcb9dc46e552a07'

if (Test-Path -LiteralPath $launchRecord) { throw '第一批已有启动记录；不要重复运行。' }
$head = & git -c "safe.directory=$repo" -C $repo rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $head.Trim() -ne $expectedCommit) { throw 'Git 提交不是 B1 固定版本；不要启动。' }
$trackedChanges = @(& git -c "safe.directory=$repo" -C $repo status --porcelain --untracked-files=no)
if ($LASTEXITCODE -ne 0 -or $trackedChanges.Count -ne 0) { throw '已跟踪工作树不干净；不要启动。' }
if (Get-Process -Id 26180 -ErrorAction SilentlyContinue) { throw '旧 B1 监督进程仍在；不要启动。' }

foreach ($model in @('262420', '262422')) {
    $b1Dir = Join-Path $repo "eval\v3e\stage_b\$model"
    $done = @(Get-ChildItem -LiteralPath $b1Dir -Filter 'seed_262*.json' -File).Count
    $active = @(Get-ChildItem -LiteralPath $b1Dir -Filter 'seed_262*.lock' -File).Count
    if ($done -ne 48 -or $active -ne 0) { throw "$model 的 B1 262000 块尚未完成 48/48。" }
    $b2Dir = Join-Path $repo "eval\v3e\stage_b2\$model"
    if ((Test-Path -LiteralPath $b2Dir) -and @(Get-ChildItem -LiteralPath $b2Dir -Force).Count -gt 0) {
        throw "$model 的 B2 输出目录已有文件；不要重复启动。"
    }
}

foreach ($workerPid in @(8736, 15204, 6872)) {
    if (-not (Get-Process -Id $workerPid -ErrorAction SilentlyContinue)) {
        throw "262421 的 B1 工作进程 $workerPid 已变化；先由我核对槽位。"
    }
}
foreach ($oldPid in @(23096, 28608, 24224, 10560, 28292, 25384)) {
    if (Get-Process -Id $oldPid -ErrorAction SilentlyContinue) { throw "待腾出的旧进程 $oldPid 仍在。" }
}

if ($CheckOnly) {
    Write-Output 'CHECK_OK: 可以在当前固定提交和三个空闲槽位启动 B2-lite 第一批。'
    exit 0
}

New-Item -ItemType Directory -Force -Path $records | Out-Null
foreach ($model in @('262420', '262422')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $repo "eval\v3e\stage_b2\$model") | Out-Null
}

$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$record = [ordered]@{
    status = 'launching'
    at = (Get-Date -Format o)
    code_commit = $expectedCommit
    b1_262421_retained_wrapper_pids = @(8736, 15204, 6872)
    launched = @()
}
$record | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $launchRecord -Encoding utf8

foreach ($spec in @(@{model = '262420'; count = 1}, @{model = '262422'; count = 2})) {
    $model = [string]$spec.model
    for ($index = 1; $index -le $spec.count; $index++) {
        $name = "b2_${model}_$index"
        $stdout = Join-Path $records "$name.stdout.log"
        $stderr = Join-Path $records "$name.stderr.log"
        if ((Test-Path -LiteralPath $stdout) -or (Test-Path -LiteralPath $stderr)) {
            throw "日志 $name 已存在；保留已启动进程并报告，不要重复运行脚本。"
        }
        $args = @(
            '-u', '-B', '-m', 'experiments.v3_handoff_scan',
            '--run-dir', "logs/v3e_$model",
            '--seeds', '262000-262047',
            '--scan', 'successes',
            '--stride', '10',
            '--verify-prefix', 'middle',
            '--output-dir', "eval/v3e/stage_b2/$model"
        )
        $process = Start-Process -FilePath $py -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
            -ArgumentList $args -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $record.launched += [ordered]@{name = $name; model = $model; wrapper_pid = $process.Id; stdout = $stdout; stderr = $stderr}
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
$record['status'] = 'started'
$record['completed_at'] = Get-Date -Format o
$record | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $launchRecord -Encoding utf8
Write-Output 'B2_LITE_FIRST_WAVE_STARTED; 现在总计算槽位为 B1 三个 + B2 三个。B1 判定前只看 B2 文件个数，不读内容。'
