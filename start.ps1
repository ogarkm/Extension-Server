# ==============================================================================
# Animex Extension Server - Windows Startup Script
# PowerShell Native Implementation
# ==============================================================================

$ErrorActionPreference = "Stop"

# ==============================================================================
# CONFIGURATION
# ==============================================================================

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$VENV_DIR = Join-Path $SCRIPT_DIR "venv"
$REQUIREMENTS = Join-Path $SCRIPT_DIR "requirements.txt"
$FIRST_RUN_FLAG = Join-Path $SCRIPT_DIR ".first_run_complete"
$LOG_FILE = Join-Path $SCRIPT_DIR "startup.log"

$APP_NAME = "Animex Extension Server"
$APP_HOST = $env:APP_HOST ?? "0.0.0.0"
$APP_PORT = $env:APP_PORT ?? 7275
$PYTHON_CMD = $env:PYTHON_CMD ?? "python"

$PORTS_REQUIRED = @(7275, 7277)

# ==============================================================================
# LOGGING
# ==============================================================================

function Log {
    param ($Level, $Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] [$Level] $Message"
    Write-Host $line
    Add-Content -Path $LOG_FILE -Value $line
}

# ==============================================================================
# PORT CHECK
# ==============================================================================

function Check-And-Free-Ports {
    foreach ($port in $PORTS_REQUIRED) {
        Log "INFO" "Checking port $port"

        $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        if ($connections) {
            $pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique
            Log "WARN" "Port $port in use by PID(s): $($pids -join ', ')"

            foreach ($pid in $pids) {
                try {
                    Stop-Process -Id $pid -ErrorAction SilentlyContinue
                } catch {}
            }

            Start-Sleep 2

            foreach ($pid in $pids) {
                if (Get-Process -Id $pid -ErrorAction SilentlyContinue) {
                    Log "WARN" "Force killing PID $pid"
                    Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
                }
            }
        }

        Start-Sleep 1

        if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
            Log "ERROR" "Failed to free port $port"
            exit 1
        }

        Log "INFO" "Port $port is free"
    }
}

# ==============================================================================
# PYTHON CHECK
# ==============================================================================

function Check-Python {
    if (-not (Get-Command $PYTHON_CMD -ErrorAction SilentlyContinue)) {
        Log "ERROR" "Python not found"
        exit 1
    }
    $version = & $PYTHON_CMD --version
    Log "INFO" "Found $version"
}

# ==============================================================================
# VENV
# ==============================================================================

function Setup-Venv {
    if (-not (Test-Path $VENV_DIR)) {
        Log "INFO" "Creating virtual environment"
        & $PYTHON_CMD -m venv $VENV_DIR
    }
}

function Activate-Venv {
    $activate = Join-Path $VENV_DIR "Scripts\Activate.ps1"
    if (-not (Test-Path $activate)) {
        Log "ERROR" "Venv activation script missing"
        exit 1
    }
    . $activate
    Log "INFO" "Virtual environment activated"
}

# ==============================================================================
# DEPENDENCIES
# ==============================================================================

function Install-Dependencies {
    if (-not (Test-Path $REQUIREMENTS)) {
        Log "WARN" "requirements.txt not found"
        return
    }
    pip install --upgrade pip | Out-Null
    pip install -r $REQUIREMENTS
    Log "INFO" "Dependencies installed"
}

# ==============================================================================
# SERVER
# ==============================================================================

function Start-Server {
    Log "INFO" "Starting $APP_NAME at http://$APP_HOST:$APP_PORT"
    uvicorn app:app --host $APP_HOST --port $APP_PORT --log-level info
}

# ==============================================================================
# CLEANUP
# ==============================================================================

$global:ShouldExit = $false
Register-EngineEvent PowerShell.Exiting -Action {
    Log "INFO" "Shutting down Animex Extension Server"
}

# ==============================================================================
# ARGUMENTS
# ==============================================================================

param (
    [switch]$Clean,
    [switch]$Check,
    [switch]$Live,
    [switch]$Version
)

if ($Version) {
    Write-Host "Animex Extension Server Startup Script v2.0 (Windows)"
    exit 0
}

if ($Clean) {
    Remove-Item $VENV_DIR -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item $FIRST_RUN_FLAG -Force -ErrorAction SilentlyContinue
    Log "INFO" "Clean install requested"
}

# ==============================================================================
# MAIN
# ==============================================================================

Add-Content $LOG_FILE "`n=== Animex Startup $(Get-Date) ==="

Check-And-Free-Ports
Check-Python
Setup-Venv
Activate-Venv
Install-Dependencies

if (-not (Test-Path $FIRST_RUN_FLAG)) {
    New-Item $FIRST_RUN_FLAG -ItemType File | Out-Null
    Log "INFO" "First-time setup completed"
}

Start-Server
