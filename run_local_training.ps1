$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$python = "$PSScriptRoot\.venv\Scripts\python.exe"
$config = "configs\marl_switch_gate_9agent.yaml"
$maddpgModel = "local_maddpg_switch_gate_2m"
$mappoModel = "local_mappo_switch_gate_2m"
$timesteps = "2000000"

if (-not (Test-Path $python)) {
    throw "Python executable not found at $python"
}

New-Item -ItemType Directory -Force -Path "run_logs" | Out-Null

function Invoke-LoggedProcess {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$StdoutPath,
        [string]$StderrPath
    )

    $proc = Start-Process -FilePath $FilePath `
        -ArgumentList $ArgumentList `
        -WorkingDirectory (Get-Location) `
        -RedirectStandardOutput $StdoutPath `
        -RedirectStandardError $StderrPath `
        -NoNewWindow `
        -PassThru `
        -Wait

    if ($proc.ExitCode -ne 0) {
        throw "Process failed with exit code $($proc.ExitCode). See $StdoutPath and $StderrPath"
    }
}

Write-Host "Starting MADDPG training..."
Invoke-LoggedProcess `
    -FilePath $python `
    -ArgumentList @("train.py","--algo","maddpg","--config",$config,"--obs-mode","lidar","--model-id",$maddpgModel,"--resume",$timesteps) `
    -StdoutPath "run_logs\maddpg_switch_gate_2m.out.log" `
    -StderrPath "run_logs\maddpg_switch_gate_2m.err.log"

Write-Host "MADDPG finished. Releasing Python process memory before MAPPO..."
Invoke-LoggedProcess `
    -FilePath $python `
    -ArgumentList @("-c","import gc, torch; gc.collect(); print('gc_done'); print('cuda_available', torch.cuda.is_available())") `
    -StdoutPath "run_logs\memory_cleanup.out.log" `
    -StderrPath "run_logs\memory_cleanup.err.log"

Write-Host "Starting MAPPO training..."
Invoke-LoggedProcess `
    -FilePath $python `
    -ArgumentList @("train.py","--algo","mappo","--config",$config,"--obs-mode","lidar","--model-id",$mappoModel,"--history-length","4","--ema-decay","0.995","--resume",$timesteps) `
    -StdoutPath "run_logs\mappo_switch_gate_2m.out.log" `
    -StderrPath "run_logs\mappo_switch_gate_2m.err.log"

Write-Host "Training sequence complete."
