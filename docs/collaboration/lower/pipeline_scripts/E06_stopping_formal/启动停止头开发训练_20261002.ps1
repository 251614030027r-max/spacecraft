[CmdletBinding()]
param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$repo = 'D:\py\DRL2'
$topic = 'C:\Users\35884\Documents\Spacecraft\过程文件\停止头'
$expected = '2e5c236f7412caea3b203519619ef7a48bb16df9'
$py = Join-Path $repo '.venv\Scripts\python.exe'
$testLog = Join-Path $topic '记录\pytest_preflight.log'
if (!(Test-Path -LiteralPath $testLog)) { throw '缺少已通过的测试日志。' }
if ((Get-Content -LiteralPath $testLog -Tail 1) -notmatch '^30 passed in ') {
    throw '指定测试回执不完整。'
}
$head = (& git -C $repo rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $head -ne $expected) { throw "固定提交不符：$head" }
$dirty = @(& git -C $repo status --porcelain --untracked-files=no)
if ($LASTEXITCODE -ne 0 -or $dirty.Count) { throw '跟踪工作树不干净。' }
$physicalDiff = @(& git -C $repo diff a714c61 HEAD --stat -- env controllers dynamics)
if ($LASTEXITCODE -ne 0 -or $physicalDiff.Count) { throw '物理或控制代码相对a714c61发生变化。' }
function Get-LfSha256([string]$Path) {
    $text = [IO.File]::ReadAllText($Path).Replace("`r`n", "`n")
    $sha = [Security.Cryptography.SHA256]::Create()
    return ([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($text))) -replace '-', '').ToLowerInvariant()
}
$labelHashes = @{
    'label_check_280000.json' = 'f0165e8e508c422c80e64a35122f95fc31d2fa2181c818ca384ccbe9c8c6ebbd'
    'label_check_280001.json' = 'd362a3a778054e85c1673dfdf207440e1ae798c03c4add80c2ce305c09264f46'
    'label_check_280002.json' = 'b7cebf20b99e9692266ff20642d2cdd553c4635148217536abb8178d15f0609a'
    'label_check_280003.json' = '4678e905376917a8b58cc34838005c726179e2886cb3e080d219e4083097f13f'
}
$labelDir = Join-Path $repo 'eval\stopping\label_check'
foreach ($name in $labelHashes.Keys) {
    $path = Join-Path $labelDir $name
    if (!(Test-Path -LiteralPath $path) -or (Get-LfSha256 $path) -ne $labelHashes[$name]) {
        throw "冻结标签哈希不符：$name"
    }
}
$run = Join-Path $repo 'logs\stop_dev_262440'
$launch = Join-Path $topic '记录\DEV_LAUNCH.json'
$outLog = Join-Path $topic '记录\dev_train.stdout.log'
$errLog = Join-Path $topic '记录\dev_train.stderr.log'
foreach ($path in @($run,$launch,$outLog,$errLog)) {
    if (Test-Path -LiteralPath $path) { throw "已有运行资产，禁止覆盖或重复启动：$path" }
}
$active = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'train\.(train_stopping|train_hybrid)'
})
if ($active.Count) { throw "已有训练进程，禁止重复启动：$($active.ProcessId -join ',')" }
if ($CheckOnly) { Write-Output 'READY: 固定提交、工作树、标签和测试回执已核对；尚未启动训练。'; return }
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:PYTHONPATH = $repo
$env:PYTHONIOENCODING = 'utf-8'
$args_ = @('-u','-B','-m','train.train_stopping','--steps','30000','--seed','262440','--run-name','stop_dev_262440','--device','cpu')
$proc = Start-Process -FilePath $py -WorkingDirectory $repo -WindowStyle Hidden -PassThru -ArgumentList $args_ -RedirectStandardOutput $outLog -RedirectStandardError $errLog
[ordered]@{
    started_at = (Get-Date).ToString('o')
    status = 'started_not_yet_verified_completed'
    wrapper_pid = $proc.Id
    commit = $head
    python = $py
    arguments = $args_
    run_directory = $run
    stdout = $outLog
    stderr = $errLog
    budget = '30000 outer decisions; MPC suffix excluded and recorded separately'
} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $launch -Encoding utf8
Write-Output "DEV_STARTED: 种子262440，30000外层决策，启动PID=$($proc.Id)。可关闭启动终端；不要重复启动或拉代码。"
Write-Output "日志：$outLog"
