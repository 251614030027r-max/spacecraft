param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(262000, 20288000, 20290000)]
    [int]$BaseSeed,
    [string]$EvidenceDirectory = 'D:\py\DRL2\logs\a3_deployable_planning\formal'
)

$ErrorActionPreference = 'Stop'
$runtimePython = 'C:\Users\35884\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:PYTHONPATH = 'D:\py\DRL2\.venv\Lib\site-packages'
$methods = @('deployable_online','offline_estimate_bound','offline_truth_bound','long_exact_h50')
New-Item -ItemType Directory -Path $EvidenceDirectory -Force | Out-Null
Set-Location 'D:\py\DRL2'
foreach ($method in $methods) {
    $output = Join-Path $EvidenceDirectory ("p1_{0}_{1}.json" -f $BaseSeed, $method)
    if (Test-Path -LiteralPath $output) { Write-Output "SKIP $BaseSeed $method"; continue }
    & $runtimePython -B -m experiments.evaluate_a3 --method $method --deployable-horizon 20 --episodes 20 --seed $BaseSeed --progress --output $output
    if ($LASTEXITCODE -ne 0) { throw "A3 P1 failed: $BaseSeed $method" }
}
