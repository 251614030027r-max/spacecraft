param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(262000, 20288000, 20290000)]
    [int]$BaseSeed,
    [string]$EvidenceDirectory = 'D:\py\DRL2\logs\a2_guidance_free'
)

$ErrorActionPreference = 'Stop'
$runtimePython = 'C:\Users\35884\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:PYTHONPATH = 'D:\py\DRL2\.venv\Lib\site-packages'
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
    $output = Join-Path $EvidenceDirectory ("a2_{0}_{1}.json" -f $BaseSeed, $method)
    if (Test-Path -LiteralPath $output) {
        Write-Output ("SKIP existing seed={0} method={1}" -f $BaseSeed, $method)
        continue
    }
    Write-Output ("START seed={0} method={1}" -f $BaseSeed, $method)
    & $runtimePython -B -m experiments.evaluate_a2 `
        --episodes 20 `
        --seed $BaseSeed `
        --method $method `
        --progress `
        --output $output
    if ($LASTEXITCODE -ne 0) {
        throw "A2 block failed: $BaseSeed $method"
    }
    Write-Output ("DONE seed={0} method={1}" -f $BaseSeed, $method)
}
