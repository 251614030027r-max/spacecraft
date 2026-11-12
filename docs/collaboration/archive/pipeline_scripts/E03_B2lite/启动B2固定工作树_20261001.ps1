param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$main = 'D:\py\DRL2'
$scanRoot = 'C:\Users\35884\Documents\Spacecraft\过程文件\阶段B2\扫描工作树_a714c61'
$records = 'C:\Users\35884\Documents\Spacecraft\过程文件\阶段B2\记录'
$py = Join-Path $main '.venv\Scripts\python.exe'
$commit = 'a714c61cc3327525ef6c103ecfcb9dc46e552a07'
$launchPath = Join-Path $records 'B2_FIXED_WORKTREE_LAUNCH_20261001.json'
$receiptPath = 'C:\Users\35884\Documents\Spacecraft\上层交付\最新\STAGE_B1_DELIVERY_RECEIPT_20261001.json'
function Assert-Git($root, $expected) {
    $head = & git -c "safe.directory=$root" -C $root rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $head.Trim() -ne $expected) { throw "版本不符: $root" }
    $dirty = @(& git -c "safe.directory=$root" -C $root status --porcelain --untracked-files=no)
    if ($LASTEXITCODE -ne 0 -or $dirty.Count) { throw "工作树不干净: $root" }
}
Assert-Git $scanRoot $commit
Assert-Git $main 'd24b00a86d2320a3ff31ebe3841f536efdb566c4'
if (-not (Test-Path -LiteralPath $py)) { throw '项目解释器不存在' }
$receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
$readoutPath = 'C:\Users\35884\Documents\Spacecraft\上层交付\最新\readout_b1.json'
if ((Get-FileHash -LiteralPath $readoutPath -Algorithm SHA256).Hash.ToLower() -ne $receipt.readout_sha256) { throw 'B1 readout 哈希不符' }
$readout = Get-Content -LiteralPath $readoutPath -Raw | ConvertFrom-Json
if ($readout.verdict -ne 'PROCEED' -or @($readout.fidelity_problems).Count -ne 0) { throw 'B1 尚未通过' }
if (Test-Path -LiteralPath $launchPath) { throw '启动记录已存在，不要重复运行；续跑另行核对' }
$active = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*experiments.v3_handoff_scan*' })
if ($active.Count) { throw '扫描进程仍在运行，不要重复启动' }
foreach ($model in $receipt.models) {
    $id = [string]$model.model
    if ((Get-FileHash -LiteralPath $model.path -Algorithm SHA256).Hash.ToLower() -ne $model.sha256) { throw "$id 模型哈希不符" }
    $b1Dir = Join-Path $main "eval\v3e\stage_b\$id"
    $files = @(Get-ChildItem -LiteralPath $b1Dir -Filter 'seed_262*.json' -File)
    if ($files.Count -ne 48) { throw "$id B1 文件不足48" }
    foreach ($seed in 262000..262047) {
        $b1 = Get-Content -LiteralPath (Join-Path $b1Dir "seed_$seed.json") -Raw | ConvertFrom-Json
        if ($b1.code_commit -ne $commit -or $b1.code_dirty -or $b1.model_sha256 -ne $model.sha256) { throw "$id/$seed B1 版本或模型不符" }
    }
    $b2Dir = Join-Path $main "eval\v3e\stage_b2\$id"
    if ((Test-Path -LiteralPath $b2Dir) -and @(Get-ChildItem -LiteralPath $b2Dir -Force).Count) { throw "$id B2目录已有文件，需核对后续跑" }
    foreach ($i in 1..2) {
        foreach ($suffix in 'stdout','stderr') {
            if (Test-Path -LiteralPath (Join-Path $records "b2_fixed_${id}_$i.$suffix.log")) { throw '日志已存在，禁止覆盖' }
        }
    }
}
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'
$env:PYTHONPATH=$scanRoot
Push-Location $scanRoot
try {
    & $py -B -c "from pathlib import Path; import experiments.v3_handoff_scan as s; from train.train_hybrid import _code_provenance; p=_code_provenance(); assert Path(s.__file__).resolve().parent.parent == Path.cwd().resolve(); assert p['code_commit']=='a714c61cc3327525ef6c103ecfcb9dc46e552a07' and not p['code_dirty']; print('IMPORT_AND_PROVENANCE_OK')"
    if ($LASTEXITCODE -ne 0) { throw '导入路径或代码来源预检失败' }
} finally { Pop-Location }
if ($CheckOnly) { Write-Output 'CHECK_OK: 三模型、144个B1文件、固定工作树与启动前条件核对通过；未启动B2'; exit 0 }
New-Item -ItemType Directory -Force -Path $records | Out-Null
$record = [ordered]@{status='launching';at=(Get-Date -Format o);scan_root=$scanRoot;code_commit=$commit;python=$py;seeds='262000-262047';scan='successes';stride=10;verify_prefix='middle';deviation='User delayed B2 until after B1 judgment and main checkout update; isolated original scan commit';workers=@()}
$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $launchPath -Encoding utf8
try {
    foreach ($id in '262420','262421','262422') {
        for ($i=1; $i -le 2; $i++) {
            $stdout=Join-Path $records "b2_fixed_${id}_$i.stdout.log"
            $stderr=Join-Path $records "b2_fixed_${id}_$i.stderr.log"
            $output=Join-Path $main "eval\v3e\stage_b2\$id"
            $arguments=@('-u','-B','-m','experiments.v3_handoff_scan','--run-dir',"$main\logs\v3e_$id",'--seeds','262000-262047','--scan','successes','--stride','10','--verify-prefix','middle','--output-dir',$output)
            $p=Start-Process -FilePath $py -WorkingDirectory $scanRoot -WindowStyle Hidden -PassThru -ArgumentList $arguments -RedirectStandardOutput $stdout -RedirectStandardError $stderr
            $record.workers += [ordered]@{model=$id;index=$i;pid=$p.Id;stdout=$stdout;stderr=$stderr;output=$output}
            $record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $launchPath -Encoding utf8
            Write-Output "B2 $id/$i PID=$($p.Id)"
            Start-Sleep -Seconds 20
            $p.Refresh()
            if ($p.HasExited -and $p.ExitCode -ne 0) { throw "进程提前失败: $id/$i，保留日志与已启动进程，禁止重复运行" }
        }
    }
    $record.status='started'
    $record.completed_at=Get-Date -Format o
} catch {
    $record.status='launch_failed'
    $record.error=$_.Exception.Message
    throw
} finally {
    $record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $launchPath -Encoding utf8
}
Write-Output 'B2_FIXED_SIX_STARTED: 可关闭启动终端；完成后需新版官方判读与独立交付。'
