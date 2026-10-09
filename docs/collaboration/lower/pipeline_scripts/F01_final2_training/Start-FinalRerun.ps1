param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$repo = 'D:\py\DRL2'
$py = Join-Path $repo '.venv\Scripts\python.exe'
$record = Join-Path (Split-Path -Parent $PSScriptRoot) ([string][char]0x8bb0+[char]0x5f55)
$commit = 'f2f8169acd580ea96a578d45022dd8952447eca5'
$receiptPath = Join-Path $record 'PREPARATION.json'
function Get-GitText([string[]]$Arguments) {
    $output = & git -c "safe.directory=$repo" -C $repo @Arguments
    if ($LASTEXITCODE -ne 0) { throw ('Git failed: '+($Arguments -join ' ')) }
    return ($output -join "`n").Trim()
}
if (-not (Test-Path -LiteralPath $receiptPath)) { throw 'Missing verified preparation receipt.' }
$receipt = Get-Content -LiteralPath $receiptPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($receipt.commit -ne $commit -or -not $receipt.tests_passed) { throw 'Preparation is not verified.' }
if ((Get-GitText @('rev-parse','HEAD')) -ne $commit) { throw 'Frozen science commit changed.' }
if (Get-GitText @('status','--porcelain','--untracked-files=no')) { throw 'Tracked worktree is dirty.' }
if (Get-GitText @('diff','2e5c236','HEAD','--stat','--','env','controllers','dynamics')) { throw 'Physics/control code changed.' }
foreach ($item in $receipt.files) {
    if ((Get-FileHash -LiteralPath $item.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $item.sha256) { throw ('Prepared file hash changed: '+$item.path) }
}
foreach ($seed in 262460,262461,262462) {
    if (Test-Path -LiteralPath (Join-Path $repo "logs\final2_$seed")) { throw "Run already exists: final2_$seed. Never resume or overwrite." }
    foreach ($suffix in 'stdout','stderr') {
        if (Test-Path -LiteralPath (Join-Path $record "train_$seed.$suffix.log")) { throw "Training log already exists for $seed." }
    }
}
if (Test-Path -LiteralPath (Join-Path $record 'MANUAL_TRAINING_LAUNCH.json')) { throw 'A launch receipt already exists; no duplicate launch.' }
$existing = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'train\.train_stopping|run_final_mainline\.py' }
if ($existing) { throw ('Existing related training/supervisor processes: '+(($existing.ProcessId) -join ',')) }
if ($CheckOnly) { Write-Host 'MANUAL_LAUNCH_PRECHECK_OK; training not started.'; return }
# Refuse launching through Codex's process tree. Start from Explorer/Win+R instead.
$ancestor = Get-CimInstance Win32_Process -Filter "ProcessId=$PID"
for ($i=0; $i -lt 16 -and $ancestor; $i++) {
    if ($ancestor.Name -match '^codex|^electron' -or $ancestor.CommandLine -match 'codex-runtimes') { throw 'Launch from a separate Windows PowerShell window opened by Win+R, not Codex.' }
    $ancestor = Get-CimInstance Win32_Process -Filter "ProcessId=$($ancestor.ParentProcessId)"
}
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'
$env:PYTHONPATH=$repo; $env:PYTHONIOENCODING='utf-8'
& powercfg /query SCHEME_CURRENT SUB_SLEEP | Out-File -LiteralPath (Join-Path $record 'POWER_SETTINGS_BEFORE.txt') -Encoding UTF8
if ($LASTEXITCODE -ne 0) { throw 'Could not record power settings; nothing started.' }
& powercfg /change standby-timeout-ac 0
if ($LASTEXITCODE -ne 0) { throw 'Could not disable AC sleep; nothing started.' }
& powercfg /change hibernate-timeout-ac 0
if ($LASTEXITCODE -ne 0) { throw 'Could not disable AC hibernation; nothing started.' }
$launch = [ordered]@{ commit=$commit; started_at=(Get-Date -Format o); manual_parent_pid=$PID; runs=@(); status='launching'; ac_sleep_disabled=$true; ac_hibernation_disabled=$true }
$launchPath = Join-Path $record 'MANUAL_TRAINING_LAUNCH.json'
$launch | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $launchPath -Encoding UTF8
foreach ($seed in 262460,262461,262462) {
    $args_ = @('-u','-B','-m','train.train_stopping','--steps','60000','--seed',"$seed",'--run-name',"final2_$seed",'--regime','w2.36_r15','--device','cpu')
    $process = Start-Process -FilePath $py -WorkingDirectory $repo -WindowStyle Hidden -PassThru -ArgumentList $args_ -RedirectStandardOutput (Join-Path $record "train_$seed.stdout.log") -RedirectStandardError (Join-Path $record "train_$seed.stderr.log")
    $launch.runs += [ordered]@{ seed=$seed; wrapper_pid=$process.Id; arguments=$args_; launched_at=(Get-Date -Format o) }
    $launch | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $launchPath -Encoding UTF8
    Write-Host "TRAIN $seed PID=$($process.Id)"
}
$launch.status='started'
$launch | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $launchPath -Encoding UTF8
# Wait only for startup manifests. Never stop or restart any process here.
$deadline=(Get-Date).AddMinutes(3)
do {
    $ready=$true
    foreach ($seed in 262460,262461,262462) {
        if (-not (Test-Path -LiteralPath (Join-Path $repo "logs\final2_$seed\manifest.json"))) { $ready=$false }
    }
    if (-not $ready) { Start-Sleep -Seconds 3 }
} until ($ready -or (Get-Date) -ge $deadline)
if (-not $ready) { throw 'Startup manifests not all ready; inspect the saved PIDs/logs. Do not run launcher again.' }
foreach ($seed in 262460,262461,262462) {
    $m=Get-Content -LiteralPath (Join-Path $repo "logs\final2_$seed\manifest.json") -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($m.code_commit -ne $commit -or $m.code_dirty -or $m.method -ne 'learned_stopping_option' -or $m.regime.name -ne 'w2.36_r15' -or $m.stopping.stop_rule -ne 'value' -or $m.stopping.bellman_stop_value -ne 'soft' -or -not $m.fresh_initialization) { throw "Manifest fidelity mismatch for $seed; preserve evidence and report." }
    Write-Host "$seed w2.36_r15 value soft False MANIFEST_OK"
}
$launch.status='manifest_verified'; $launch | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $launchPath -Encoding UTF8
Write-Host 'THREE_FRESH_RUNS_STARTED. No Codex supervisor owns these processes. Keep the PC powered on and connected to AC. Do not log off Windows or reboot.'
