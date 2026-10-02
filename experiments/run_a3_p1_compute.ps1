param(
    [string]$EvidenceDirectory = 'D:\py\DRL2\logs\a3_deployable_planning\compute_serial'
)

$ErrorActionPreference = 'Stop'
$runtimePython = 'C:\Users\35884\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:PYTHONPATH = 'D:\py\DRL2\.venv\Lib\site-packages'
$env:OMP_NUM_THREADS = '1'; $env:OPENBLAS_NUM_THREADS = '1'; $env:MKL_NUM_THREADS = '1'
$methods = @('deployable_online','offline_estimate_bound','offline_truth_bound','long_exact_h50')
New-Item -ItemType Directory -Path $EvidenceDirectory -Force | Out-Null
Set-Location 'D:\py\DRL2'
foreach ($method in $methods) {
    $output = Join-Path $EvidenceDirectory ("compute_{0}.json" -f $method)
    if (Test-Path -LiteralPath $output) { Write-Output "SKIP $method"; continue }
    & $runtimePython -B -m experiments.evaluate_a3 --method $method --deployable-horizon 20 --episodes 1 --seed 262000 --serial-compute-profile --output $output
    if ($LASTEXITCODE -ne 0) { throw "A3 compute profile failed: $method" }
}
