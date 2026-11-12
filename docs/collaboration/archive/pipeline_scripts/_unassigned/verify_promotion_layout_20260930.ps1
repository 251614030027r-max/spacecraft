$ErrorActionPreference='Stop'
$base=[IO.Path]::GetFullPath('C:\Users\35884\Documents\Spacecraft\过程文件\工程整理\验证\migration-fixture_20260930_retry1')
if(Test-Path -LiteralPath $base){throw 'Fixture already exists; do not overwrite'}
[void](New-Item -ItemType Directory -Path $base -Force)
$main=Join-Path $base 'DRL2';$ve=Join-Path $base 'DRL2_v3e';$old=Join-Path $base '旧工程';$archive=Join-Path $base '验证快照'
function G([string]$path,[string[]]$arguments){$out=& git -c "safe.directory=$($path.Replace('\','/'))" -c user.name='Layout Verification' -c user.email='layout-verification@example.invalid' -C $path @arguments;if($LASTEXITCODE -ne 0){throw "Fixture git failed: $($arguments -join ' ')"};return ($out -join "`n")}
[void](New-Item -ItemType Directory -Path $main)
[void](G $main @('init','-b','topic'))
'old-code' | Set-Content -LiteralPath (Join-Path $main 'README.md') -Encoding UTF8
[void](G $main @('add','README.md'));[void](G $main @('commit','-m','old fixture'))
$oldCommit=G $main @('rev-parse','HEAD')
[void](G $main @('worktree','add','--detach',$ve,$oldCommit))
'new-code' | Set-Content -LiteralPath (Join-Path $ve 'README.md') -Encoding UTF8
[void](New-Item -ItemType Directory -Path (Join-Path $ve 'eval\v3e') -Force)
'{"tracked_upper_evidence":true}' | Set-Content -LiteralPath (Join-Path $ve 'eval\v3e\upper.json') -Encoding UTF8
[void](G $ve @('add','README.md','eval/v3e/upper.json'));[void](G $ve @('commit','-m','new fixture'))
$newCommit=G $ve @('rev-parse','HEAD')
[void](G $main @('worktree','add','--detach',$old,$oldCommit))
Add-Content -LiteralPath (Join-Path $main '.git\info\exclude') -Value "`n/logs/`n/eval/v3e/seed/" -Encoding UTF8
[void](New-Item -ItemType Directory -Path (Join-Path $main 'logs\old') -Force)
'old-output' | Set-Content -LiteralPath (Join-Path $main 'logs\old\result.txt') -Encoding UTF8
[void](New-Item -ItemType Directory -Path (Join-Path $old 'logs') -Force)
Move-Item -LiteralPath (Join-Path $main 'logs\old') -Destination (Join-Path $old 'logs\old')
[void](G $main @('merge','--ff-only',$newCommit))
[void](New-Item -ItemType Directory -Path (Join-Path $ve 'logs\new'),(Join-Path $ve 'eval\v3e\seed') -Force)
'new-model-placeholder' | Set-Content -LiteralPath (Join-Path $ve 'logs\new\model.txt') -Encoding UTF8
'new-evaluation-placeholder' | Set-Content -LiteralPath (Join-Path $ve 'eval\v3e\seed\result.txt') -Encoding UTF8
$hash=(Get-FileHash -LiteralPath (Join-Path $ve 'logs\new\model.txt') -Algorithm SHA256).Hash
Move-Item -LiteralPath (Join-Path $ve 'logs\new') -Destination (Join-Path $main 'logs\new')
Move-Item -LiteralPath (Join-Path $ve 'eval\v3e\seed') -Destination (Join-Path $main 'eval\v3e\seed')
[void](G $main @('worktree','move',$ve,$archive))
[void](New-Item -ItemType Junction -Path $ve -Value $main)
if((G $main @('rev-parse','HEAD')) -ne $newCommit -or (G $old @('rev-parse','HEAD')) -ne $oldCommit -or (G $archive @('rev-parse','HEAD')) -ne $newCommit){throw 'Version preservation failed'}
if((G $main @('branch','--show-current')) -ne 'topic' -or (G $main @('status','--porcelain'))){throw 'Primary branch or clean status failed'}
if(-not(Test-Path -LiteralPath (Join-Path $main 'eval\v3e\upper.json'))){throw 'Tracked upper evidence was lost'}
if((Get-FileHash -LiteralPath (Join-Path $ve 'logs\new\model.txt') -Algorithm SHA256).Hash -ne $hash){throw 'Compatibility path/hash failed'}
$result=[ordered]@{at=(Get-Date -Format o);result='PASS';tracked_upper_evidence_preserved=$true;old_version_preserved=$true;primary_branch_preserved=$true;native_git_worktree_move_verified=$true;windows_junction_read_verified=$true;artifact_hash_unchanged=$true;primary_git_status='clean';fixture=$base;real_engineering_directories_touched=$false}
$result | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath 'C:\Users\35884\Documents\Spacecraft\过程文件\工程整理\记录\PROMOTION_LAYOUT_FIXTURE_20260930.json' -Encoding UTF8
$result | ConvertTo-Json -Depth 4
