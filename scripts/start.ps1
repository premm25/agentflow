# Starts the observability dashboard (8200) and the single-process orchestrator (8100).
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
Start-Process -WindowStyle Hidden -FilePath python -ArgumentList "-m","dashboard" -RedirectStandardOutput "$env:TEMP\roa_dashboard.log" -RedirectStandardError "$env:TEMP\roa_dashboard.err"
Start-Sleep 2
Start-Process -WindowStyle Hidden -FilePath python -ArgumentList "-m","roa" -RedirectStandardOutput "$env:TEMP\roa_orchestrator.log" -RedirectStandardError "$env:TEMP\roa_orchestrator.err"
Write-Host "Orchestrator + UI : http://localhost:8100/ui/"
Write-Host "Dashboard         : http://localhost:8200/"
