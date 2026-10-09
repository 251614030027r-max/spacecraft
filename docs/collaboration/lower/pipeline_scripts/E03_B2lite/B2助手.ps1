param([ValidateSet('Status','Finalize','Verify','Publish','Fault','StopOnError','SelfTest')][string]$Mode='Status')
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$env:PYTHONIOENCODING='utf-8'
& 'D:\py\DRL2\.venv\Scripts\python.exe' -u -B (Join-Path $PSScriptRoot 'B2助手.py') $Mode
if ($LASTEXITCODE -ne 0) { throw "B2助手停止，退出码=$LASTEXITCODE；阅读上述错误并保留现场。" }
