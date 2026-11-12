$ErrorActionPreference = 'Stop'
$base = 'C:\Users\35884\Documents\Spacecraft'
$repo = 'D:\py\DRL2'
$worktree = Join-Path $base '过程文件\远端交付\审查工作树'
$records = Join-Path $base '过程文件\远端交付\记录'
$recordPath = Join-Path $records 'V3E_R2_REMOTE_RELEASE_20260930.json'
$receiptPath = Join-Path $base '上层交付\最新\V3E_VALUE_R2_DELIVERY_RECEIPT_20260929.json'
$receipt = Get-Content -LiteralPath $receiptPath -Raw -Encoding UTF8 | ConvertFrom-Json
$archive = $receipt.archive.path
$expectedSha = $receipt.archive.sha256.ToLowerInvariant()
$tag = 'v3e-60k-value-r2-20260930'
$branch = 'review/v3e-60k-value-r2-20260930'
$reviewCommit = 'dc6682d4ec11e5858bedc9e619ac6c4142fc7e66'
$api = 'https://api.github.com/repos/251614030027r-max/spacecraft'
$reviewUrl = "https://github.com/251614030027r-max/spacecraft/tree/$branch/docs/reviews/v3e_60k_value_r2_20260930"

if ($receipt.status -ne 'evaluations_completed' -or $receipt.role -ne 'second-round main result') { throw 'Final R2 receipt is not the completed main result' }
if ((Get-Item -LiteralPath $archive).Length -ne $receipt.archive.bytes) { throw 'Local archive length changed' }
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedSha) { throw 'Local archive SHA changed' }
$remoteLine = & git -c safe.directory=D:/py/DRL2 -C $repo ls-remote --heads origin $branch
if ($LASTEXITCODE -ne 0 -or -not $remoteLine -or ($remoteLine -split '\s+')[0] -ne $reviewCommit) { throw 'Remote review branch is not at the verified commit' }
if ((& git -c "safe.directory=$($worktree.Replace('\','/'))" -C $worktree status --porcelain)) { throw 'Review worktree has edits' }

[void](New-Item -ItemType Directory -Path $records -Force)
$checksumPath = Join-Path $records 'V3E_60K_VALUE_R2_MAIN_DELIVERY_20260929.zip.sha256'
[IO.File]::WriteAllText($checksumPath, "$expectedSha  $([IO.Path]::GetFileName($archive))`n", [Text.UTF8Encoding]::new($false))
$env:GIT_TERMINAL_PROMPT = '0'
$env:GCM_INTERACTIVE = 'never'
$credentialLines = "protocol=https`nhost=github.com`n`n" | & git credential fill 2>$null
$secret = @($credentialLines | Where-Object { $_ -like 'password=*' } | Select-Object -First 1)
if (-not $secret.Count) { throw 'No GitHub credential available; review branch remains published' }
$token = $secret[0].Substring(9)
$credentialLines = $null
$secret = $null
$headers = @{ Authorization = "Bearer $token"; Accept = 'application/vnd.github+json'; 'User-Agent' = 'Codex-V3e-Review'; 'X-GitHub-Api-Version' = '2022-11-28' }

function SaveRecord([string]$status, $release, $assets, [string]$detail = '') {
    $record = [ordered]@{
        status = $status
        updated_at = (Get-Date -Format o)
        repository = '251614030027r-max/spacecraft'
        review_branch = $branch
        review_commit = $reviewCommit
        review_url = $reviewUrl
        release_tag = $tag
        release_id = $release.id
        release_url = $release.html_url
        draft = $release.draft
        prerelease = $release.prerelease
        archive_name = [IO.Path]::GetFileName($archive)
        archive_bytes = $receipt.archive.bytes
        archive_sha256 = $expectedSha
        assets = @($assets | ForEach-Object { [ordered]@{ name=$_.name; size=$_.size; digest=$_.digest; browser_download_url=$_.browser_download_url } })
        detail = $detail
    }
    $record | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath $recordPath -Encoding UTF8
}

try {
    $release = $null
    try { $release = Invoke-RestMethod -Uri "$api/releases/tags/$tag" -Headers $headers -Method Get }
    catch { if ($_.Exception.Response.StatusCode.value__ -ne 404) { throw } }
    if (-not $release) {
        $body = [ordered]@{
            tag_name = $tag
            target_commitish = $reviewCommit
            name = 'V3e 60k value R2 evidence for review'
            body = "Review entry: $reviewUrl`n`nOfficial verdict: does not hold. R2 calibration PASS/STOP/STOP; 262420 arbitration 40/48 with one truth-constraint violation. The full evidence ZIP is attached as a release asset; SHA-256: $expectedSha. Frozen experiment commit: c84ed7435f790198630ee7011fc7fd74b7307b48. This is a prerelease for upper-layer review, not a claim that the method passed."
            draft = $true
            prerelease = $true
            generate_release_notes = $false
        } | ConvertTo-Json -Depth 5
        $release = Invoke-RestMethod -Uri "$api/releases" -Headers $headers -Method Post -Body $body -ContentType 'application/json; charset=utf-8'
        SaveRecord 'draft_created' $release @()
    }
    if ($release.target_commitish -ne $reviewCommit) { throw 'Existing release points to another commit' }
    $assets = @((Invoke-RestMethod -Uri "$api/releases/$($release.id)/assets?per_page=100" -Headers $headers -Method Get) | Where-Object { $null -ne $_ })
    $uploadBase = $release.upload_url.Split('{')[0]
    foreach ($file in @($archive, $checksumPath)) {
        $name = [IO.Path]::GetFileName($file)
        $match = @($assets | Where-Object { $_.name -eq $name })
        if ($match.Count -gt 1) { throw "Duplicate release assets: $name" }
        if (-not $match.Count) {
            if (-not $release.draft) { throw "Published release lacks expected asset: $name" }
            $uploadUri = "$uploadBase`?name=$([uri]::EscapeDataString($name))"
            $uploaded = Invoke-RestMethod -Uri $uploadUri -Headers $headers -Method Post -InFile $file -ContentType 'application/octet-stream'
            $assets = @($assets) + @($uploaded)
            SaveRecord 'asset_uploaded' $release $assets
        }
    }
    $assets = @((Invoke-RestMethod -Uri "$api/releases/$($release.id)/assets?per_page=100" -Headers $headers -Method Get) | Where-Object { $null -ne $_ })
    foreach ($file in @($archive, $checksumPath)) {
        $name = [IO.Path]::GetFileName($file)
        $asset = @($assets | Where-Object { $_.name -eq $name })
        if ($asset.Count -ne 1) { throw "Release asset count mismatch: $name" }
        $remoteLength = [long]$asset[0].size
        $localLength = [long](Get-Item -LiteralPath $file).Length
        if ($remoteLength -ne $localLength) { throw "Release asset length mismatch: $name (remote=$remoteLength, local=$localLength)" }
        $localSha = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($asset[0].digest -and $asset[0].digest -ne "sha256:$localSha") { throw "Release asset SHA mismatch: $name" }
    }
    SaveRecord 'assets_verified' $release $assets
    if ($release.draft) {
        $release = Invoke-RestMethod -Uri "$api/releases/$($release.id)" -Headers $headers -Method Patch -Body '{"draft":false,"prerelease":true}' -ContentType 'application/json'
    }
    $release = Invoke-RestMethod -Uri "$api/releases/tags/$tag" -Headers $headers -Method Get
    if ($release.draft -or -not $release.prerelease) { throw 'Release did not publish as a prerelease' }
    $assets = @($release.assets)
    SaveRecord 'completed' $release $assets
    [pscustomobject]@{status='completed';release_url=$release.html_url;review_url=$reviewUrl;archive_url=@($assets | Where-Object {$_.name -eq [IO.Path]::GetFileName($archive)})[0].browser_download_url;archive_sha256=$expectedSha;review_commit=$reviewCommit} | ConvertTo-Json
} finally {
    $token = $null
    $headers = $null
}
