$ErrorActionPreference = 'Stop'
$repoRoot = 'D:\py\DRL2'
$activeRoot = 'D:\py\DRL2_v3e'
$topicRoot = 'C:\Users\35884\Documents\Spacecraft\过程文件\工程整理'
$recordsRoot = Join-Path $topicRoot '记录'
[void](New-Item -ItemType Directory -Path $recordsRoot -Force)
function GitText([string]$repo,[string[]]$arguments) {
    $result = & git -c "safe.directory=$($repo.Replace('\','/'))" -C $repo @arguments
    if ($LASTEXITCODE -ne 0) { throw "git failed: $($arguments -join ' ')" }
    return ($result -join "`n")
}
$oldCommit = GitText $repoRoot @('rev-parse','HEAD')
$activeCommit = GitText $activeRoot @('rev-parse','HEAD')
if ($oldCommit -ne 'b05e389483cafab15c94c4852cecf08579f29111' -or $activeCommit -ne 'c84ed7435f790198630ee7011fc7fd74b7307b48') { throw 'Unexpected version; do not reorganize automatically' }
if ((GitText $repoRoot @('status','--porcelain','--untracked-files=no')) -or (GitText $activeRoot @('status','--porcelain','--untracked-files=no'))) { throw 'Tracked edits present; preserve them before any layout change' }
$excludePath = Join-Path $repoRoot '.git\info\exclude'
$before = [IO.File]::ReadAllBytes($excludePath)
$backupPath = Join-Path $recordsRoot 'git_info_exclude_before_20260930.txt'
if (-not (Test-Path -LiteralPath $backupPath)) { [IO.File]::WriteAllBytes($backupPath,$before) }
$rules = @('/eval/v3e/','/eval/v3d/','/logs/v3b_*/','/logs/v3c_*/','/logs/v3d_*/','/logs/v3e_*/')
$existing = [IO.File]::ReadAllText($excludePath)
$lines = $existing -split "`r?`n"
$missing = @($rules | Where-Object {$_ -notin $lines})
if ($missing.Count) {
    [IO.File]::AppendAllText($excludePath,"`n# Spacecraft local V3 outputs: evidence stays on disk; explicit publish only.`n" + ($missing -join "`n") + "`n",[Text.UTF8Encoding]::new($false))
}
$snapshots = @(
    @{name='snapshot/v3d-before-v3e-20260930';commit=$oldCommit},
    @{name='snapshot/v3e-value-r2-20260930';commit=$activeCommit}
)
foreach ($snapshot in $snapshots) {
    & git -c safe.directory=D:/py/DRL2 -C $repoRoot show-ref --verify --quiet "refs/heads/$($snapshot.name)"
    if ($LASTEXITCODE -eq 1) { [void](GitText $repoRoot @('branch',$snapshot.name,$snapshot.commit)) }
    elseif ($LASTEXITCODE -ne 0) { throw 'Cannot inspect snapshot ref' }
    if ((GitText $repoRoot @('rev-parse',$snapshot.name)) -ne $snapshot.commit) { throw "Snapshot ref already points elsewhere: $($snapshot.name)" }
}
$record = [ordered]@{
    at=(Get-Date -Format o)
    old_project=$repoRoot;old_commit=$oldCommit
    active_project=$activeRoot;active_commit=$activeCommit
    primary_branch=(GitText $repoRoot @('branch','--show-current'))
    upstream=(GitText $repoRoot @('rev-parse','--abbrev-ref','@{upstream}'))
    origin=(GitText $repoRoot @('remote','get-url','origin'))
    worktrees=(GitText $repoRoot @('worktree','list','--porcelain'))
    local_snapshot_refs=$snapshots
    local_exclude_rules_added=$missing
    local_exclude_backup=$backupPath
    old_git_status=(GitText $repoRoot @('status','--porcelain'))
    active_git_status=(GitText $activeRoot @('status','--porcelain'))
    tracked_files_unchanged=$true
    current_code_and_artifact_paths_unchanged=$true
    pushed=$false
    migration_state='prepared; wait for completed verified round-two delivery and idle processes'
}
$record | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $recordsRoot 'REPOSITORY_LAYOUT_PREPARED_20260930.json') -Encoding UTF8
[ordered]@{old_commit=$oldCommit;active_commit=$activeCommit;old_git_status=$record.old_git_status;active_git_status=$record.active_git_status;local_snapshot_refs=$snapshots;exclude_added=$missing;paths_unchanged=$true;pushed=$false} | ConvertTo-Json -Depth 4
