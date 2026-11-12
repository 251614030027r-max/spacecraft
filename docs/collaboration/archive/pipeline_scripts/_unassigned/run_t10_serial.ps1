$ErrorActionPreference='Stop'
Set-Location -LiteralPath 'D:\py\DRL2'
$taskRoot='D:\py\DRL2\logs\t10'
$taskPython='D:\py\DRL2\.venv\Scripts\python.exe'
$taskState=@{status='running';stage='S1';started_at=(Get-Date).ToString('o');orchestrator_pid=$PID;current=$null;completed=@();error=$null;jobs=@();code_commit=(git rev-parse HEAD);training_authorized_in_this_queue=$false}
function Save-State { $taskState.updated_at=(Get-Date).ToString('o'); $taskState | ConvertTo-Json -Depth 9 | Set-Content -LiteralPath "$taskRoot\state.json" -Encoding utf8 }
function Run-Serial([string]$taskName,[string]$taskModule,[object[]]$taskArgs,[int]$taskExpected) {
    $taskOutput="$taskRoot\$taskName.json"
    if (Test-Path -LiteralPath $taskOutput) { throw "Output already exists: $taskOutput" }
    $taskOther=@(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" | Where-Object { $_.CommandLine -match '(train\.train_hybrid|experiments\.(evaluate_|replay_|diagnose_))' })
    if ($taskOther.Count) { throw 'Concurrent training or experiment detected' }
    $taskArguments=@('-u','-B','-m',$taskModule)+$taskArgs+@('--output',$taskOutput)
    $taskState.current=$taskName
    $taskState.jobs+=@{name=$taskName;arguments=$taskArguments;expected_records=$taskExpected;started_at=(Get-Date).ToString('o')}
    Save-State
    $taskProc=Start-Process -FilePath $taskPython -ArgumentList $taskArguments -WorkingDirectory 'D:\py\DRL2' -WindowStyle Hidden -RedirectStandardOutput "$taskRoot\$taskName.stdout.log" -RedirectStandardError "$taskRoot\$taskName.stderr.log" -PassThru
    $taskState.worker_pid=$taskProc.Id; Save-State
    $taskProc.WaitForExit(); $taskProc.Refresh()
    if ($taskProc.ExitCode -ne 0) { throw "$taskName exited $($taskProc.ExitCode)" }
    $taskRecords=@(Get-Content -LiteralPath $taskOutput -Raw | ConvertFrom-Json)
    if ($taskRecords.Count -ne $taskExpected) { throw "$taskName incomplete record count" }
    $taskState.completed+=$taskName; Save-State
    return $taskRecords
}
try {
    $taskCheck=@(Run-Serial 'gate_h20_check' 'experiments.diagnose_terminal_gate' (@('--horizon','20','--seeds')+@(262000..262011)) 12)
    $taskReference=@(Get-Content logs/terminal_gate/gate_h20.json -Raw | ConvertFrom-Json)
    $taskKeys=@('completed','end_time_s','steps','qp_infeasible_steps','first_infeasible_time_s','illegal_terminal_entry_count','terminal_region_active','final_range_m','final_fov_margin_rad','time_failure','distance_failure')
    $taskDiffs=@()
    foreach ($taskRow in $taskCheck) {
        $taskRef=@($taskReference | Where-Object seed -eq $taskRow.seed)
        if ($taskRef.Count -ne 1) { throw 'Reference seed missing or duplicated' }
        foreach ($taskKey in $taskKeys) { if ($taskRow.$taskKey -cne $taskRef[0].$taskKey) { $taskDiffs+=@{seed=$taskRow.seed;field=$taskKey;actual=$taskRow.$taskKey;reference=$taskRef[0].$taskKey} } }
        if ([Math]::Abs($taskRow.force_impulse_n_s-$taskRef[0].force_impulse_n_s) -ge 0.01) { $taskDiffs+=@{seed=$taskRow.seed;field='force_impulse_n_s';actual=$taskRow.force_impulse_n_s;reference=$taskRef[0].force_impulse_n_s} }
    }
    $taskState.s1_comparison_differences=$taskDiffs
    if (@($taskCheck | Where-Object qp_infeasible_steps -ne 0).Count) { throw 'GATE_A: nonzero QP infeasible steps in S1' }
    if (@($taskCheck | Where-Object completed).Count -ne 8 -or $taskDiffs.Count) { throw 'S1 reproduction mismatch; do not continue' }
    $taskState.mpc_frozen_at=(Get-Date).ToString('o'); $taskState.stage='S3'; Save-State
    $taskBaseline=@(Run-Serial 'pure_mpc_h35_48' 'experiments.evaluate_arrival_condition_script' (@('--horizon','35','--policy','commit_now','--seeds')+@(262000..262047)) 48)
    if (@($taskBaseline | Where-Object qp_infeasible_steps -ne 0).Count) { throw 'GATE_A: QP infeasibility in expanded baseline; report without changing MPC' }
    $taskFailures=@($taskBaseline | Where-Object { -not $_.completed } | Sort-Object seed | ForEach-Object seed)
    $taskControls=@($taskBaseline | Where-Object completed | Sort-Object seed | Select-Object -First 6 | ForEach-Object seed)
    $taskState.failed_seeds=$taskFailures; $taskState.reverse_cost_seeds=$taskControls; $taskState.stage='S4'; Save-State
    if ($taskFailures.Count -eq 0) { $taskState.status='gate_b_no_failures'; Save-State; exit 0 }
    foreach ($taskT in @(20,40,60,80,100,120)) {
        $taskScan=@(Run-Serial "commit_at_$taskT" 'experiments.evaluate_arrival_condition_script' (@('--horizon','35','--policy','commit_at','--commit-time-s',"$taskT",'--seeds')+$taskFailures) $taskFailures.Count)
        if (@($taskScan | Where-Object qp_infeasible_steps -ne 0).Count) { throw 'GATE_A: QP infeasibility during commit scan; report without changing MPC' }
    }
    if ($taskControls.Count) { $null=Run-Serial 'reverse_cost_t40' 'experiments.evaluate_arrival_condition_script' (@('--horizon','35','--policy','commit_at','--commit-time-s','40','--seeds')+$taskControls) $taskControls.Count }
    $taskState.status='awaiting_gate_b_review'; $taskState.current=$null
} catch { $taskState.status='stopped'; $taskState.error=$_.Exception.Message }
finally { Save-State }
