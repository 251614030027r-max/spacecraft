param(
    [string]$EvidenceDirectory = 'D:\py\DRL2\logs\a3_deployable_planning\horizon_profile_analytic'
)

$ErrorActionPreference = 'Stop'
$runtimePython = 'C:\Users\35884\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:PYTHONPATH = 'D:\py\DRL2\.venv\Lib\site-packages'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$horizons = @(10, 15, 20)

New-Item -ItemType Directory -Path $EvidenceDirectory -Force | Out-Null
Set-Location 'D:\py\DRL2'
foreach ($horizon in $horizons) {
    $output = Join-Path $EvidenceDirectory ("deployable_h{0}.json" -f $horizon)
    if (Test-Path -LiteralPath $output) {
        Write-Output ("SKIP existing horizon={0}" -f $horizon)
        continue
    }
    & $runtimePython -B -m experiments.evaluate_a3 `
        --method deployable_online `
        --deployable-horizon $horizon `
        --episodes 1 `
        --seed 262000 `
        --max-time 10 `
        --serial-compute-profile `
        --output $output
    if ($LASTEXITCODE -ne 0) {
        throw "A3 horizon profile failed: h$horizon"
    }
}
