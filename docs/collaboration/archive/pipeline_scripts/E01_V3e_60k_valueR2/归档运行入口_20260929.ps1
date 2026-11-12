$ErrorActionPreference = 'Stop'
$taskRoot = 'C:\Users\35884\Documents\Spacecraft'
$taskStatePath = Join-Path $taskRoot 'V3E_60K_EXECUTION_20260929.json'
$destinationRoot = Join-Path $taskRoot '过程文件\V3e60k\运行归档'
if (-not (Test-Path -LiteralPath $taskStatePath)) { '{"already_archived":true}'; return }
$taskState = Get-Content -LiteralPath $taskStatePath -Raw -Encoding utf8 | ConvertFrom-Json
$roundTwoStatePath = Join-Path $taskRoot '过程文件\V3e价值第二轮\记录\V3E_VALUE_R2_EXECUTION_20260929.json'
if (Test-Path -LiteralPath $roundTwoStatePath) {
    $roundTwoState = Get-Content -LiteralPath $roundTwoStatePath -Raw -Encoding utf8 | ConvertFrom-Json
    if (Get-Process -Id $roundTwoState.supervisor_pid -ErrorAction SilentlyContinue) { '{"deferred_second_round_running":true}'; return }
    if ($roundTwoState.package_status -ne 'completed') { throw 'Second-round delivery must finish before first-round runtime archival' }
}
if (Get-Process -Id $taskState.supervisor_pid -ErrorAction SilentlyContinue) { '{"deferred_supervisor_running":true}'; return }
if ($taskState.package_status -ne 'completed' -or $taskState.status -notin @('evaluations_completed','program_fault_stop')) { throw 'Verified delivery and completed packaging required before runtime archival' }
$rootPrefix = [IO.Path]::GetFullPath($taskRoot).TrimEnd('\') + '\'
$names = @('V3E_60K_EXECUTION_20260929.json','V3E_60K_EXECUTION_20260929.lock','V3E_60K_PIPELINE_20260929.stdout.log','V3E_60K_PIPELINE_20260929.stderr.log','V3E_60K_PACKAGING_20260929.stdout.log','V3E_60K_PACKAGING_20260929.stderr.log','package_v3e_60k_pipeline_20260929.py')
foreach ($taskName in $names) {
    $source = Join-Path $taskRoot $taskName
    $target = Join-Path $destinationRoot $taskName
    foreach ($taskPath in @($source,$target)) {
        if (-not [IO.Path]::GetFullPath($taskPath).StartsWith($rootPrefix,[StringComparison]::OrdinalIgnoreCase)) { throw "Outside workspace: $taskPath" }
    }
    if (Test-Path -LiteralPath $source) {
        if (Test-Path -LiteralPath $target) { throw "Archive already exists: $target" }
        $item = Get-Item -LiteralPath $source -Force
        $item.Attributes = $item.Attributes -band (-bnot [IO.FileAttributes]::Hidden)
        Move-Item -LiteralPath $source -Destination $target
    }
}
'{"runtime_archived":true}'
