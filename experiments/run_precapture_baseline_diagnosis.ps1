param(
    [string]$EvidenceDirectory = 'D:\py\DRL2\logs\precapture_planning_baseline',
    [ValidateRange(1, 20)]
    [int]$Episodes = 5,
    [int]$BaseSeed = 262000,
    [ValidateRange(0.1, 300.0)]
    [double]$MaxTimeSeconds = 300.0
)

$ErrorActionPreference = 'Stop'
$runtimePython = 'C:\Users\35884\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:PYTHONPATH = 'D:\py\DRL2\.venv\Lib\site-packages'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

$cases = @(
    @{ Name = 'fixed_h001'; Horizon = 1; Reference = 'fixed'; Guidance = $null; Model = 'nominal' },
    @{ Name = 'fixed_h003'; Horizon = 3; Reference = 'fixed'; Guidance = $null; Model = 'nominal' },
    @{ Name = 'fixed_h010'; Horizon = 10; Reference = 'fixed'; Guidance = $null; Model = 'nominal' },
    @{ Name = 'fixed_h050'; Horizon = 50; Reference = 'fixed'; Guidance = $null; Model = 'nominal' },
    @{ Name = 'fixed_h200'; Horizon = 200; Reference = 'fixed'; Guidance = $null; Model = 'nominal' },
    @{ Name = 'two_stage_h050'; Horizon = 50; Reference = 'external_local'; Guidance = 'two_stage'; Model = 'nominal' },
    @{ Name = 'oracle_two_stage_h200'; Horizon = 200; Reference = 'external_local'; Guidance = 'two_stage'; Model = 'truth' }
)

New-Item -ItemType Directory -Path $EvidenceDirectory -Force | Out-Null
Set-Location 'D:\py\DRL2'
foreach ($case in $cases) {
    $output = Join-Path $EvidenceDirectory ($case.Name + '.json')
    if (Test-Path -LiteralPath $output) {
        Write-Output ("SKIP existing case={0}" -f $case.Name)
        continue
    }
    Write-Output ("START case={0} episodes={1} seed={2}" -f $case.Name, $Episodes, $BaseSeed)
    $arguments = @(
        '-B', '-m', 'experiments.evaluate_mpc',
        '--task', 'precapture_planning',
        '--episodes', $Episodes,
        '--seed', $BaseSeed,
        '--horizon', $case.Horizon,
        '--reference-source', $case.Reference,
        '--controller-model', $case.Model,
        '--max-time', $MaxTimeSeconds,
        '--progress', '--quiet',
        '--output', $output
    )
    if ($null -ne $case.Guidance) {
        $arguments += @('--external-guidance', $case.Guidance)
    }
    & $runtimePython @arguments
    if ($LASTEXITCODE -ne 0) {
        throw ("Precapture diagnosis failed: {0}" -f $case.Name)
    }
    Write-Output ("DONE case={0}" -f $case.Name)
}

& $runtimePython -B -m eval.precapture_baseline_report `
    --input-directory $EvidenceDirectory `
    --output-directory $EvidenceDirectory
if ($LASTEXITCODE -ne 0) {
    throw 'Precapture baseline report failed'
}
