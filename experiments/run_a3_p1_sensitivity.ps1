param(
    [string]$EvidenceDirectory = 'D:\py\DRL2\logs\a3_deployable_planning\sensitivity'
)

$ErrorActionPreference = 'Stop'
$runtimePython = 'C:\Users\35884\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:PYTHONPATH = 'D:\py\DRL2\.venv\Lib\site-packages'
New-Item -ItemType Directory -Path $EvidenceDirectory -Force | Out-Null
Set-Location 'D:\py\DRL2'
foreach ($horizon in @(10,15)) {
    $output = Join-Path $EvidenceDirectory ("sensitivity_262000_h{0}.json" -f $horizon)
    if (Test-Path -LiteralPath $output) { Write-Output "SKIP h$horizon"; continue }
    & $runtimePython -B -m experiments.evaluate_a3 --method deployable_online --deployable-horizon $horizon --episodes 20 --seed 262000 --progress --output $output
    if ($LASTEXITCODE -ne 0) { throw "A3 sensitivity failed: h$horizon" }
}
