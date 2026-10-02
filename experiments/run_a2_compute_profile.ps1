param(
    [string]$EvidenceDirectory = 'D:\py\DRL2\logs\a2_guidance_free\compute_serial'
)

$ErrorActionPreference = 'Stop'
$runtimePython = 'C:\Users\35884\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:PYTHONPATH = 'D:\py\DRL2\.venv\Lib\site-packages'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$methods = @(
    'guided_a1_control',
    'terminal_short',
    'receding_plan_long',
    'planning_tracking',
    'planning_tracking_oracle'
)

New-Item -ItemType Directory -Path $EvidenceDirectory -Force | Out-Null
Set-Location 'D:\py\DRL2'
foreach ($method in $methods) {
    $output = Join-Path $EvidenceDirectory ("compute_{0}.json" -f $method)
    if (Test-Path -LiteralPath $output) {
        Write-Output ("SKIP existing method={0}" -f $method)
        continue
    }
    Write-Output ("PROFILE method={0}" -f $method)
    & $runtimePython -B -m experiments.evaluate_a2 `
        --episodes 1 `
        --seed 262000 `
        --method $method `
        --serial-compute-profile `
        --output $output
    if ($LASTEXITCODE -ne 0) {
        throw "A2 compute profile failed: $method"
    }
}
