$ErrorActionPreference='Stop'
$repo='D:\py\DRL2'
$record=Join-Path (Split-Path -Parent $PSScriptRoot) ([string][char]0x8bb0+[char]0x5f55)
$processes=Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' }
foreach ($seed in 262460,262461,262462) {
    $run=Join-Path $repo "logs\final2_$seed"
    if (-not (Test-Path -LiteralPath $run)) { Write-Host "$seed NOT_STARTED"; continue }
    $m=Get-Content -LiteralPath (Join-Path $run 'manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $log=Join-Path $record "train_$seed.stdout.log"
    $steps='unknown'
    if (Test-Path -LiteralPath $log) {
        $match=Get-Content -LiteralPath $log -Tail 150 | Select-String '\|\s*total_timesteps\s*\|\s*(\d+)' | Select-Object -Last 1
        if ($match) { $steps=$match.Matches[0].Groups[1].Value }
    }
    if ($m.actual_outer_decisions) { $steps=$m.actual_outer_decisions }
    $active=@($processes | Where-Object { $_.CommandLine -match "--run-name\s+final2_$seed(?:\s|$)" })
    $pids=@($active.ProcessId) -join ','
    $checkpoint=Get-ChildItem -LiteralPath (Join-Path $run 'checkpoints') -Filter 'stopping_*_outer_decisions.zip' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $health=if($active.Count -gt 0){'ACTIVE'}elseif($m.status -eq 'completed'){'COMPLETED'}else{'INTERRUPTED_PROCESS_MISSING'}
    Write-Host "$seed $steps/60000 status=$health manifest=$($m.status) PID=[$pids] checkpoint=$($checkpoint.Name)"
}
Write-Host 'Read-only status. Never restart, stop, or clean anything. At 30k, perform the prescribed structural check.'
