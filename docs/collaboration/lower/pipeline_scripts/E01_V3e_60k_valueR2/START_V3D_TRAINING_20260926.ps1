param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(262420, 262421, 262422)]
    [int]$Seed
)

$ErrorActionPreference = 'Stop'
$repository = 'D:\py\DRL2'
$expectedCommit = 'b05e389483cafab15c94c4852cecf08579f29111'
$pythonExecutable = Join-Path $repository '.venv\Scripts\python.exe'
Set-Location -LiteralPath $repository

$currentCommit = & git -c safe.directory=D:/py/DRL2 rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $currentCommit.Trim() -ne $expectedCommit) {
    throw 'Checkout changed since V3d preflight. Recheck the execution order before training.'
}
$trackedChanges = & git -c safe.directory=D:/py/DRL2 status --porcelain --untracked-files=no
if ($LASTEXITCODE -ne 0 -or $trackedChanges) {
    throw 'Tracked working tree must be clean before training.'
}
$runName = "v3d_$Seed"
if (Test-Path -LiteralPath (Join-Path $repository "logs\$runName")) {
    throw "Run directory already exists: $runName. Do not resume or reuse it."
}
if (-not (Test-Path -LiteralPath $pythonExecutable)) {
    throw "Python interpreter is missing: $pythonExecutable"
}

$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& $pythonExecutable -u -B -m train.train_hybrid --steps 60000 --seed $Seed --run-name $runName --horizon 35 --parametrization task_state_v3 --adaptive-task --device cpu
if ($LASTEXITCODE -ne 0) {
    throw "Training exited with code $LASTEXITCODE. Preserve the run and report; do not restart it."
}
