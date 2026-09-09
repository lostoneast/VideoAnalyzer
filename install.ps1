$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "====================================="
Write-Host " VideoAnalyzer installation"
Write-Host "====================================="
Write-Host ""

# ----------------------------------------------------------
# Python
# ----------------------------------------------------------

python --version

# ----------------------------------------------------------
# Virtual environment
# ----------------------------------------------------------

if (Test-Path ".venv") {
    Write-Host "Removing old environment..."
    Remove-Item ".venv" -Recurse -Force
}

python -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel

# ----------------------------------------------------------
# PyTorch
# ----------------------------------------------------------

Write-Host ""
Write-Host "Installing PyTorch..."

pip install `
    torch==2.14.0 `
    torchvision `
    torchaudio `
    --index-url https://download.pytorch.org/whl/cu130

# ----------------------------------------------------------
# Project dependencies
# ----------------------------------------------------------

Write-Host ""
Write-Host "Installing project dependencies..."

pip install -r requirements.txt

# ----------------------------------------------------------
# Directories
# ----------------------------------------------------------

New-Item -ItemType Directory -Force input  | Out-Null
New-Item -ItemType Directory -Force output | Out-Null
New-Item -ItemType Directory -Force temp   | Out-Null
New-Item -ItemType Directory -Force cache  | Out-Null
New-Item -ItemType Directory -Force logs   | Out-Null

# ----------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------

Write-Host ""
Write-Host "====================================="
Write-Host " Environment diagnostics"
Write-Host "====================================="

python -c "import sys; print('Python:', sys.version)"
python -c "import torch; print('Torch:', torch.__version__)"
python -c "import torch; print('CUDA build:', torch.version.cuda)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
python -c "import transformers; print('Transformers:', transformers.__version__)"

Write-Host ""
Write-Host "====================================="
Write-Host " Installation complete"
Write-Host "====================================="