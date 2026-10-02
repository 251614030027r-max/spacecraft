param(
    [Parameter(Mandatory = $true)]
    [string]$EvidenceDirectory,
    [Parameter(Mandatory = $true)]
    [ValidateSet(262000, 20288000, 20290000)]
    [int]$BaseSeed
)

$ErrorActionPreference = 'Stop'
$runtimePython = 'C:\Users\35884\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:PYTHONPATH = 'D:\py\DRL2\.venv\Lib\site-packages'
$cases = @(
    @($BaseSeed, 'oracle'),
    @($BaseSeed, 'estimate')
)

New-Item -ItemType Directory -Path $EvidenceDirectory -Force | Out-Null
Set-Location 'D:\py\DRL2'
foreach ($case in $cases) {
    $base = [int]$case[0]
    $source = [string]$case[1]
    $output = Join-Path $EvidenceDirectory ("g0_{0}_{1}.json" -f $base, $source)
    if (Test-Path -LiteralPath $output) {
        Write-Output ("SKIP existing block={0} source={1}" -f $base, $source)
        continue
    }
    Write-Output ("START block={0} source={1}" -f $base, $source)
    & $runtimePython -B -m experiments.evaluate_mpc `
        --episodes 20 `
        --seed $base `
        --task single_phase `
        --control-state-source $source `
        --progress `
        --quiet `
        --output $output
    if ($LASTEXITCODE -ne 0) {
        throw "G0 block failed: $base $source"
    }
    Write-Output ("DONE block={0} source={1}" -f $base, $source)
}
