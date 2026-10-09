param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$workspace = 'C:\Users\35884\Documents\Spacecraft'
$mainRoot = 'D:\py\DRL2'
$veRoot = 'D:\py\DRL2_v3e'
$oldRoot = 'D:\py\DRL2_旧工程_20260930'
$veArchive = 'D:\py\legacy\DRL2_v3e_验证快照_20260930'
$recordsRoot = Join-Path $workspace '过程文件\工程整理\记录'
$journalPath = Join-Path $recordsRoot 'REPOSITORY_PROMOTION_20260930.json'
$oldCommit = 'b05e389483cafab15c94c4852cecf08579f29111'
$veCommit = 'c84ed7435f790198630ee7011fc7fd74b7307b48'
$branch = 'claude/sac-mpc-coupling-design-ns7g6i'
function GitText([string]$repo,[string[]]$arguments) {
    $result = & git -c "safe.directory=$($repo.Replace('\','/'))" -C $repo @arguments
    if ($LASTEXITCODE -ne 0) { throw "git failed: $($arguments -join ' ')" }
    return ($result -join "`n")
}
function GuardPath([string]$path) {
    $absolute = [IO.Path]::GetFullPath($path)
    if (-not $absolute.StartsWith('D:\py\',[StringComparison]::OrdinalIgnoreCase)) { throw "Outside named engineering workspace: $absolute" }
    return $absolute
}
foreach ($path in @($mainRoot,$veRoot,$oldRoot,$veArchive)) { [void](GuardPath $path) }
if (Test-Path -LiteralPath $journalPath) {
    $journal = Get-Content -LiteralPath $journalPath -Raw -Encoding UTF8 | ConvertFrom-Json -AsHashtable
    if ($journal.status -eq 'completed') { $journal | ConvertTo-Json -Depth 9; return }
} else {
    $journal = [ordered]@{status='prepared';started_at=(Get-Date -Format o);old_commit=$oldCommit;active_commit=$veCommit;main_root=$mainRoot;old_root=$oldRoot;validation_archive=$veArchive;moves=@();errors=@()}
}
function SaveJournal { $journal.updated_at=Get-Date -Format o; $journal | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $journalPath -Encoding UTF8 }
$r2Path = Join-Path $workspace '过程文件\V3e价值第二轮\记录\V3E_VALUE_R2_EXECUTION_20260929.json'
$r2 = Get-Content -LiteralPath $r2Path -Raw -Encoding UTF8 | ConvertFrom-Json
$r1Path = Join-Path $workspace 'V3E_60K_EXECUTION_20260929.json'
if (-not (Test-Path -LiteralPath $r1Path)) { $r1Path = Join-Path $workspace '过程文件\V3e60k\运行归档\V3E_60K_EXECUTION_20260929.json' }
$r1 = Get-Content -LiteralPath $r1Path -Raw -Encoding UTF8 | ConvertFrom-Json
if ($r2.status -ne 'evaluations_completed' -or $r2.package_status -ne 'completed' -or
    (Get-Process -Id $r2.supervisor_pid -ErrorAction SilentlyContinue) -or
    (Get-Process -Id $r1.supervisor_pid -ErrorAction SilentlyContinue)) {
    '{"status":"deferred","reason":"Wait for successful completed round-two delivery and exited supervisors; do not move active files"}'; return
}
$busy = @(Get-CimInstance Win32_Process -Filter "Name LIKE 'python%'" | Where-Object {$_.CommandLine -match '(?i)D:[\\/]py[\\/]DRL2(?:_v3e)?[\\/]'})
if ($busy.Count -or (Get-Process -Name git -ErrorAction SilentlyContinue)) { '{"status":"deferred","reason":"Related Python or Git process is active"}'; return }
$receiptPath = Join-Path $workspace '上层交付\最新\V3E_VALUE_R2_DELIVERY_RECEIPT_20260929.json'
$receipt = Get-Content -LiteralPath $receiptPath -Raw -Encoding UTF8 | ConvertFrom-Json -AsHashtable
if ($receipt.role -ne 'second-round main result' -or $receipt.status -ne 'evaluations_completed' -or -not $receipt.zip_crc_and_manifest_hashes_verified) { throw 'Normal verified round-two main delivery required' }
if ((Get-FileHash -LiteralPath $receipt.archive.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $receipt.archive.sha256) { throw 'Final archive SHA does not match receipt' }
if ((GitText $mainRoot @('rev-parse','HEAD')) -notin @($oldCommit,$veCommit) -or (GitText $mainRoot @('branch','--show-current')) -ne $branch) { throw 'Main checkout changed; preserve external work' }
if ((GitText $mainRoot @('status','--porcelain','--untracked-files=no'))) { throw 'Main has tracked edits; preserve them' }
if ((GitText $mainRoot @('remote','get-url','origin')) -ne 'https://github.com/251614030027r-max/spacecraft.git' -or
    (GitText $mainRoot @('rev-parse','--abbrev-ref','@{upstream}')) -ne "origin/$branch") { throw 'Remote or upstream changed; do not replace it' }
if ($CheckOnly) { '{"status":"ready","mode":"check_only","main_root":"D:/py/DRL2","code_and_artifact_paths_unchanged":true}'; return }
function MoveUnit([string]$source,[string]$target) {
    $source=GuardPath $source; $target=GuardPath $target
    if (-not (Test-Path -LiteralPath $source)) {
        if (@($journal.moves | Where-Object {$_.source -eq $source -and $_.target -eq $target}).Count -and (Test-Path -LiteralPath $target)) { return }
        throw "Missing source: $source"
    }
    if (Test-Path -LiteralPath $target) { throw "Refuse overwrite: $target" }
    $sourceItem=Get-Item -LiteralPath $source -Force
    if ($sourceItem.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Do not move a linked artifact directory: $source" }
    $files = if ($sourceItem.PSIsContainer) { @(Get-ChildItem -LiteralPath $source -Recurse -File -Force) } else { @($sourceItem) }
    $hashes = @($files | ForEach-Object {@{relative=if($sourceItem.PSIsContainer){[IO.Path]::GetRelativePath($source,$_.FullName)}else{'.'};sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant();bytes=$_.Length}})
    $move = [ordered]@{source=$source;target=$target;status='planned';files=$hashes}
    $journal.moves += $move; SaveJournal
    [void](New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($target)) -Force)
    Move-Item -LiteralPath $source -Destination $target
    foreach ($file in $hashes) {
        $path=if($file.relative -eq '.'){$target}else{Join-Path $target $file.relative}
        if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.sha256) { throw "Moved file SHA changed: $path" }
    }
    $move.status='verified'; SaveJournal
}
try {
    [void](New-Item -ItemType Directory -Path $recordsRoot -Force)
    $journal.status='running'; SaveJournal
    # An interrupted move is verified at its destination before any subsequent stage.
    foreach ($move in $journal.moves) {
        if ($move.status -ne 'verified') {
            if ((Test-Path -LiteralPath $move.target) -and -not (Test-Path -LiteralPath $move.source)) {
                foreach ($file in $move.files) {
                    $path=if($file.relative -eq '.'){$move.target}else{Join-Path $move.target $file.relative}
                    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.sha256) { throw "Resume SHA mismatch: $path" }
                }
                $move.status='verified'; SaveJournal
            } else { throw 'Interrupted move needs inspection; do not overwrite or guess' }
        }
    }
    if (-not (Test-Path -LiteralPath $oldRoot)) {
        if ((Get-PSDrive -Name D).Free -lt 8GB) { throw 'Need at least 8 GiB free for retained historical checkout' }
        [void](GitText $mainRoot @('worktree','add',$oldRoot,'snapshot/v3d-before-v3e-20260930'))
    }
    if ((GitText $oldRoot @('rev-parse','HEAD')) -ne $oldCommit) { throw 'Old snapshot target is not the expected version' }
    # Only whole, untracked artifact units are moved; published tracked evidence stays in the repository.
    if ((GitText $mainRoot @('rev-parse','HEAD')) -eq $oldCommit) {
        foreach ($category in @('logs','eval','models')) {
            $categoryPath=Join-Path $mainRoot $category
            if (-not (Test-Path -LiteralPath $categoryPath)) { continue }
            $units=if($category -eq 'models'){@(Get-Item -LiteralPath $categoryPath)}else{@(Get-ChildItem -LiteralPath $categoryPath -Force)}
            foreach ($unit in $units) {
                $relative=[IO.Path]::GetRelativePath($mainRoot,$unit.FullName).Replace('\','/')
                if (-not (GitText $mainRoot @('ls-files','--',$relative))) {
                    MoveUnit $unit.FullName (Join-Path $oldRoot $relative)
                }
            }
        }
        [void](GitText $mainRoot @('merge','--ff-only',$veCommit))
    }
    if ((GitText $mainRoot @('rev-parse','HEAD')) -ne $veCommit) { throw 'Primary version promotion failed' }
    if (-not (Test-Path -LiteralPath $veArchive)) {
        if ((GitText $veRoot @('rev-parse','HEAD')) -ne $veCommit -or (GitText $veRoot @('status','--porcelain','--untracked-files=no'))) { throw 'V3e checkout changed; preserve it' }
        foreach ($seed in @(262420,262421,262422)) {
            $relative="logs/v3e_$seed"
            if (GitText $veRoot @('ls-files','--',$relative)) { throw "Unexpected tracked policy artifact directory: $relative" }
            MoveUnit (Join-Path $veRoot $relative) (Join-Path $mainRoot $relative)
        }
        # eval/v3e also contains three tracked upper-layer evidence JSONs. Git's
        # fast-forward retains those; move only untracked generated child units.
        foreach ($unit in @(Get-ChildItem -LiteralPath (Join-Path $veRoot 'eval\v3e') -Force)) {
            $relative=[IO.Path]::GetRelativePath($veRoot,$unit.FullName).Replace('\','/')
            if (-not (GitText $veRoot @('ls-files','--',$relative))) {
                MoveUnit $unit.FullName (Join-Path $mainRoot $relative)
            }
        }
        [void](New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($veArchive)) -Force)
        [void](GitText $mainRoot @('worktree','move',$veRoot,$veArchive))
        $journal.validation_worktree_moved=$true; SaveJournal
    }
    if ((GitText $veArchive @('rev-parse','HEAD')) -ne $veCommit) { throw 'Archived V3e version mismatch' }
    if (-not (Test-Path -LiteralPath $veRoot)) { [void](New-Item -ItemType Junction -Path $veRoot -Value $mainRoot) }
    $link=Get-Item -LiteralPath $veRoot -Force
    if ($link.LinkType -ne 'Junction' -or [IO.Path]::GetFullPath(@($link.Target)[0]).TrimEnd('\') -ne $mainRoot) { throw 'Original V3e path is not the expected compatibility junction' }
    # Full final large-artifact manifest now resolves through the compatibility path.
    $bundle=[IO.Path]::GetDirectoryName($receipt.report)
    $large=Get-Content -LiteralPath (Join-Path $bundle 'large_artifact_locations.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($item in $large) {
        if ((Get-FileHash -LiteralPath $item.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $item.sha256) { throw "Final artifact SHA mismatch: $($item.path)" }
    }
    $oldLocation=Get-Location
    try {
        Set-Location -LiteralPath $mainRoot
        $env:PYTHONIOENCODING='utf-8'
        & (Join-Path $mainRoot '.venv\Scripts\python.exe') -B -c "import sys,numpy,cvxpy,gymnasium,stable_baselines3; from pathlib import Path; import experiments.v3_collect_value_data as m; assert Path(m.__file__).resolve().is_relative_to(Path('D:/py/DRL2').resolve()); print(sys.executable); print(m.__file__)" > (Join-Path $recordsRoot 'PROMOTION_IMPORT_CHECK_20260930.log')
        if ($LASTEXITCODE -ne 0) { throw 'Promoted project import check failed' }
    } finally { Set-Location -LiteralPath $oldLocation.Path }
    if ((GitText $mainRoot @('status','--porcelain')) -or (GitText $mainRoot @('status','--porcelain','--untracked-files=no'))) { throw 'Promoted checkout is not clean' }
    if ((GitText $mainRoot @('branch','--show-current')) -ne $branch -or (GitText $mainRoot @('rev-parse','--abbrev-ref','@{upstream}')) -ne "origin/$branch") { throw 'Branch/upstream changed during promotion' }
    $journal.worktrees=GitText $mainRoot @('worktree','list','--porcelain')
    $journal.branch=$branch; $journal.upstream="origin/$branch"; $journal.git_status='clean'; $journal.remote_unchanged=$true
    $journal.compatibility_junction=$veRoot; $journal.large_artifact_hashes_verified=$large.Count
    $journal.status='artifacts_verified'; SaveJournal
    $receiptBackup=Join-Path $recordsRoot 'V3E_VALUE_R2_DELIVERY_RECEIPT_before_promotion_20260930.json'
    if (-not (Test-Path -LiteralPath $receiptBackup)) { Copy-Item -LiteralPath $receiptPath -Destination $receiptBackup }
    $receipt.project_root_at_evaluation=$veRoot; $receipt.project_root=$mainRoot
    $receipt.compatibility_project_root=$veRoot; $receipt.engineering_promotion_record=$journalPath
    $receipt | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $receiptPath -Encoding UTF8
    $entry=@"
# 最终工程入口

正式工程：D:/py/DRL2；代码版本：$veCommit；分支：$branch。
解释器：D:/py/DRL2/.venv/Scripts/python.exe。远端和upstream保持原值。
旧V3d工程：D:/py/DRL2_旧工程_20260930；原V3e验证快照：D:/py/legacy/DRL2_v3e_验证快照_20260930。
D:/py/DRL2_v3e目前为正式工程的兼容目录联接，本轮原绝对路径仍可读取；后续主动工作统一使用DRL2。
主模型：logs/v3e_<seed>/final_model.zip；主价值：eval/v3e/<seed>/values_r2/。
工程迁移不代表模型通过科学门；逐种子结论以本目录正式报告为准。科学JSON及ZIP字节未改写。
完整文件映射及SHA核验：$journalPath。
"@
    $entry | Set-Content -LiteralPath (Join-Path $workspace '上层交付\最新\工程入口.md') -Encoding UTF8
    $latestReadme=Join-Path $workspace '上层交付\最新\README.md'
    if (-not ([IO.File]::ReadAllText($latestReadme).Contains('工程已统一为 D:/py/DRL2'))) {
        Add-Content -LiteralPath $latestReadme -Value "`n工程已统一为 D:/py/DRL2，原V3e路径为兼容联接。详见[工程入口](工程入口.md)。" -Encoding UTF8
    }
    $marker=Join-Path $recordsRoot '入口已更新_20260930.txt'
    $entry | Set-Content -LiteralPath $marker -Encoding UTF8
    $projectPath=Join-Path $workspace '项目说明.md'
    $projectText=[IO.File]::ReadAllText($projectPath)
    $projectText=$projectText.Replace('当前评估使用 **D:/py/DRL2_v3e** 作为工程及运行工作目录','本轮评估使用 **D:/py/DRL2_v3e**（现为兼容联接）作为运行目录，后续正式工程统一为 **D:/py/DRL2**')
    $projectText=$projectText.Replace('当前仅完成Git本地产物排除及版本快照，物理工程尚未切换。','工程迁移已完成并核验，正式工程为D:/py/DRL2，旧版及验证worktree已经归档。')
    $projectText=$projectText.Replace('活动实现为 D:/py/DRL2_v3e，原D:/py/DRL2保存V3d，D:/py/DRL为只读参考。','正式活动实现为 D:/py/DRL2，旧V3d位于D:/py/DRL2_旧工程_20260930，D:/py/DRL为只读参考。')
    [IO.File]::WriteAllText($projectPath,$projectText,[Text.UTF8Encoding]::new($false))
    $journal.status='completed'; $journal.completed_at=Get-Date -Format o; SaveJournal
    [ordered]@{status='completed';project_root=$mainRoot;old_project_root=$oldRoot;commit=$veCommit;git_status='clean';upstream="origin/$branch";hashes_unchanged=$true;zip_unchanged=$true} | ConvertTo-Json
} catch {
    $journal.status='migration_needs_attention'; $journal.errors += @{at=(Get-Date -Format o);error=$_.Exception.Message}; SaveJournal
    throw
}
