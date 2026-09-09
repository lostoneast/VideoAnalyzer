$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================================"
Write-Host " VideoAnalyzer installation"
Write-Host " Windows + Python 3.14 + CUDA + Qwen2.5-VL"
Write-Host "============================================================"
Write-Host ""

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

$ProjectRoot = $PSScriptRoot
$VenvPath    = Join-Path $ProjectRoot ".venv"
$ToolsPath   = Join-Path $ProjectRoot "tools"
$FFmpegPath  = Join-Path $ToolsPath "ffmpeg"
$FFmpegBin   = Join-Path $FFmpegPath "bin"
$TempPath    = Join-Path $ProjectRoot "temp"

$FFmpegZip     = Join-Path $TempPath "ffmpeg-shared.zip"
$FFmpegExtract = Join-Path $TempPath "ffmpeg_extract"

# Shared FFmpeg build for Windows x64
$FFmpegUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n9.0-latest-win64-gpl-shared-9.0.zip"

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

function Fail($Message) {
    Write-Host ""
    Write-Host "ERROR: $Message" -ForegroundColor Red
    Write-Host ""
    exit 1
}

function Step($Message) {
    Write-Host ""
    Write-Host "------------------------------------------------------------"
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "------------------------------------------------------------"
}

function Run-PythonCheck {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string]$Content
    )

    $CheckFile = Join-Path $TempPath "_check_$Name.py"

    Set-Content `
        -Path $CheckFile `
        -Value $Content `
        -Encoding UTF8

    & $Python $CheckFile

    $ExitCode = $LASTEXITCODE

    if (Test-Path $CheckFile) {
        Remove-Item $CheckFile -Force
    }

    if ($ExitCode -ne 0) {
        Fail "$Name check failed."
    }
}

# ------------------------------------------------------------
# Python check
# ------------------------------------------------------------

Step "Checking Python"

try {
    $PythonVersion = python --version 2>&1
    Write-Host $PythonVersion
}
catch {
    Fail "Python was not found."
}

if ($PythonVersion -notmatch "Python 3\.14") {
    Fail "Python 3.14 is required. Detected: $PythonVersion"
}

# ------------------------------------------------------------
# Directories
# ------------------------------------------------------------

Step "Creating project directories"

New-Item -ItemType Directory -Force $ToolsPath | Out-Null
New-Item -ItemType Directory -Force $TempPath  | Out-Null

New-Item -ItemType Directory -Force (Join-Path $ProjectRoot "input")  | Out-Null
New-Item -ItemType Directory -Force (Join-Path $ProjectRoot "output") | Out-Null
New-Item -ItemType Directory -Force (Join-Path $ProjectRoot "cache")  | Out-Null
New-Item -ItemType Directory -Force (Join-Path $ProjectRoot "logs")   | Out-Null

# ------------------------------------------------------------
# Virtual environment
# ------------------------------------------------------------

Step "Creating virtual environment"

if (Test-Path $VenvPath) {
    Write-Host "Removing old .venv..."
    Remove-Item $VenvPath -Recurse -Force
}

python -m venv $VenvPath

$Python = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Fail "Virtual environment creation failed."
}

# ------------------------------------------------------------
# pip
# ------------------------------------------------------------

Step "Updating pip"

& $Python -m pip install --upgrade pip setuptools wheel

if ($LASTEXITCODE -ne 0) {
    Fail "pip update failed."
}

# ------------------------------------------------------------
# Windows certificate support
# ------------------------------------------------------------

Step "Installing Windows certificate support"

& $Python -m pip install python-certifi-win32

if ($LASTEXITCODE -ne 0) {
    Fail "python-certifi-win32 installation failed."
}

# ------------------------------------------------------------
# PyTorch
# ------------------------------------------------------------

Step "Installing PyTorch CUDA 13.0 stack"

& $Python -m pip install `
    torch==2.14.0 `
    torchvision `
    torchaudio `
    torchcodec `
    --index-url https://download.pytorch.org/whl/cu130

if ($LASTEXITCODE -ne 0) {
    Fail "PyTorch / TorchCodec installation failed."
}

# ------------------------------------------------------------
# Qwen
# ------------------------------------------------------------

Step "Installing Qwen2.5-VL dependencies"

& $Python -m pip install `
    transformers `
    accelerate `
    qwen-vl-utils `
    huggingface_hub `
    safetensors `
    sentencepiece `
    protobuf

if ($LASTEXITCODE -ne 0) {
    Fail "Qwen dependencies installation failed."
}

# ------------------------------------------------------------
# Video dependencies
# ------------------------------------------------------------

Step "Installing video and image dependencies"

& $Python -m pip install `
    opencv-python `
    pillow `
    av `
    imageio `
    imageio-ffmpeg `
    ffmpeg-python `
    numpy `
    pandas `
    tqdm `
    psutil

if ($LASTEXITCODE -ne 0) {
    Fail "Video dependencies installation failed."
}

# ------------------------------------------------------------
# Download shared FFmpeg
# ------------------------------------------------------------

Step "Installing shared FFmpeg"

if (Test-Path $FFmpegPath) {
    Remove-Item $FFmpegPath -Recurse -Force
}

if (Test-Path $FFmpegExtract) {
    Remove-Item $FFmpegExtract -Recurse -Force
}

New-Item -ItemType Directory -Force $FFmpegExtract | Out-Null

Write-Host "Downloading:"
Write-Host $FFmpegUrl

try {
    Invoke-WebRequest `
        -Uri $FFmpegUrl `
        -OutFile $FFmpegZip `
        -UseBasicParsing
}
catch {
    Fail "FFmpeg download failed: $($_.Exception.Message)"
}

# ------------------------------------------------------------
# Extract FFmpeg
# ------------------------------------------------------------

Step "Extracting FFmpeg"

Expand-Archive `
    -Path $FFmpegZip `
    -DestinationPath $FFmpegExtract `
    -Force

$FFmpegExe = Get-ChildItem `
    -Path $FFmpegExtract `
    -Filter "ffmpeg.exe" `
    -Recurse |
    Select-Object -First 1

if (-not $FFmpegExe) {
    Fail "ffmpeg.exe was not found inside downloaded archive."
}

$SourceBin = $FFmpegExe.Directory.FullName

New-Item -ItemType Directory -Force $FFmpegBin | Out-Null

Copy-Item `
    -Path (Join-Path $SourceBin "*") `
    -Destination $FFmpegBin `
    -Recurse `
    -Force

if (Test-Path $FFmpegZip) {
    Remove-Item $FFmpegZip -Force
}

if (Test-Path $FFmpegExtract) {
    Remove-Item $FFmpegExtract -Recurse -Force
}

# ------------------------------------------------------------
# Verify FFmpeg
# ------------------------------------------------------------

Step "Checking shared FFmpeg"

$InstalledFFmpeg  = Join-Path $FFmpegBin "ffmpeg.exe"
$InstalledFFprobe = Join-Path $FFmpegBin "ffprobe.exe"

if (-not (Test-Path $InstalledFFmpeg)) {
    Fail "ffmpeg.exe is missing."
}

if (-not (Test-Path $InstalledFFprobe)) {
    Fail "ffprobe.exe is missing."
}

$FFmpegDlls = Get-ChildItem `
    -Path $FFmpegBin `
    -Filter "*.dll"

if ($FFmpegDlls.Count -eq 0) {
    Fail "Shared FFmpeg DLL files were not found."
}

Write-Host "FFmpeg DLL count: $($FFmpegDlls.Count)"

& $InstalledFFmpeg -version

if ($LASTEXITCODE -ne 0) {
    Fail "FFmpeg executable test failed."
}

# ------------------------------------------------------------
# Runtime environment
# ------------------------------------------------------------

Step "Configuring runtime environment"

$env:PATH = "$FFmpegBin;$env:PATH"
$env:FORCE_QWENVL_VIDEO_READER = "torchcodec"

Write-Host "FFmpeg bin:"
Write-Host $FFmpegBin

# ------------------------------------------------------------
# sitecustomize.py
# ------------------------------------------------------------

Step "Configuring Python DLL search path"

$SitePackages = & $Python -c "import site; print(site.getsitepackages()[0])"

if ($LASTEXITCODE -ne 0) {
    Fail "Could not determine site-packages directory."
}

$SiteCustomize = Join-Path $SitePackages "sitecustomize.py"

$SiteCustomizeContent = @"
import os

FFMPEG_BIN = r"$FFmpegBin"

if os.path.isdir(FFMPEG_BIN):
    try:
        os.add_dll_directory(FFMPEG_BIN)
    except Exception:
        pass

    os.environ["PATH"] = (
        FFMPEG_BIN
        + os.pathsep
        + os.environ.get("PATH", "")
    )

os.environ.setdefault(
    "FORCE_QWENVL_VIDEO_READER",
    "torchcodec"
)
"@

Set-Content `
    -Path $SiteCustomize `
    -Value $SiteCustomizeContent `
    -Encoding UTF8

Write-Host "Created:"
Write-Host $SiteCustomize

# ------------------------------------------------------------
# Basic diagnostics
# ------------------------------------------------------------

Step "Running environment diagnostics"

& $Python -c "import sys; print('Python:', sys.version)"
if ($LASTEXITCODE -ne 0) {
    Fail "Python diagnostic failed."
}

& $Python -c "import torch; print('Torch:', torch.__version__)"
if ($LASTEXITCODE -ne 0) {
    Fail "Torch diagnostic failed."
}

& $Python -c "import torch; print('CUDA build:', torch.version.cuda)"
if ($LASTEXITCODE -ne 0) {
    Fail "CUDA build diagnostic failed."
}

& $Python -c "import torchvision; print('TorchVision:', torchvision.__version__)"
if ($LASTEXITCODE -ne 0) {
    Fail "TorchVision diagnostic failed."
}

& $Python -c "import transformers; print('Transformers:', transformers.__version__)"
if ($LASTEXITCODE -ne 0) {
    Fail "Transformers diagnostic failed."
}

# ------------------------------------------------------------
# CUDA check
# ------------------------------------------------------------

Step "Checking CUDA"

$CudaCheck = @'
import torch

print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

print("GPU:", torch.cuda.get_device_name(0))

props = torch.cuda.get_device_properties(0)

print(
    "VRAM:",
    round(props.total_memory / 1024**3, 2),
    "GB"
)
'@

Run-PythonCheck `
    -Name "cuda" `
    -Content $CudaCheck

# ------------------------------------------------------------
# TorchCodec import check
# ------------------------------------------------------------

Step "Checking TorchCodec import"

$env:VIDEANALYZER_FFMPEG_BIN = $FFmpegBin

$TorchCodecImportCheck = @'
import os

ffmpeg_bin = os.environ["VIDEANALYZER_FFMPEG_BIN"]

print("FFmpeg DLL directory:")
print(ffmpeg_bin)

if os.path.isdir(ffmpeg_bin):
    os.add_dll_directory(ffmpeg_bin)

import torchcodec

print("TorchCodec import: OK")
'@

Run-PythonCheck `
    -Name "torchcodec_import" `
    -Content $TorchCodecImportCheck

# ------------------------------------------------------------
# Generate test video
# ------------------------------------------------------------

Step "Generating TorchCodec test video"

$TestVideo = Join-Path $TempPath "torchcodec_test.mp4"

if (Test-Path $TestVideo) {
    Remove-Item $TestVideo -Force
}

& $InstalledFFmpeg `
    -hide_banner `
    -loglevel error `
    -y `
    -f lavfi `
    -i "testsrc2=size=320x240:rate=25" `
    -t 1 `
    -pix_fmt yuv420p `
    $TestVideo

if ($LASTEXITCODE -ne 0) {
    Fail "Could not generate test video."
}

if (-not (Test-Path $TestVideo)) {
    Fail "Test video was not created."
}

# ------------------------------------------------------------
# TorchCodec decode test
# ------------------------------------------------------------

Step "Testing TorchCodec video decoding"

$env:TORCHCODEC_TEST_VIDEO = $TestVideo

$TorchCodecDecodeCheck = @'
import os

ffmpeg_bin = os.environ["VIDEANALYZER_FFMPEG_BIN"]

if os.path.isdir(ffmpeg_bin):
    os.add_dll_directory(ffmpeg_bin)

from torchcodec.decoders import VideoDecoder

video = os.environ["TORCHCODEC_TEST_VIDEO"]

decoder = VideoDecoder(
    video,
    device="cpu"
)

frame = decoder[0]

print("TorchCodec decoder: OK")
print("Video:", video)
print("Frame shape:", tuple(frame.shape))
print("Frame dtype:", frame.dtype)
print("Duration:", decoder.metadata.duration_seconds)
print("FPS:", decoder.metadata.average_fps)
'@

Run-PythonCheck `
    -Name "torchcodec_decode" `
    -Content $TorchCodecDecodeCheck

# ------------------------------------------------------------
# qwen-vl-utils check
# ------------------------------------------------------------

Step "Checking qwen-vl-utils"

$QwenUtilsCheck = @'
import os

os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchcodec"

from qwen_vl_utils import process_vision_info

print("qwen-vl-utils import: OK")
print(
    "Video backend:",
    os.environ.get("FORCE_QWENVL_VIDEO_READER")
)
'@

Run-PythonCheck `
    -Name "qwen_utils" `
    -Content $QwenUtilsCheck

# ------------------------------------------------------------
# Final package versions
# ------------------------------------------------------------

Step "Installed package versions"

$VersionCheck = @'
import sys
import torch
import torchvision
import transformers
import torchcodec

print("Python       :", sys.version.split()[0])
print("Torch        :", torch.__version__)
print("TorchVision  :", torchvision.__version__)
print("Transformers :", transformers.__version__)

try:
    print("TorchCodec   :", torchcodec.__version__)
except Exception:
    print("TorchCodec   : installed")

print("CUDA         :", torch.version.cuda)
print("CUDA OK      :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU          :", torch.cuda.get_device_name(0))
'@

Run-PythonCheck `
    -Name "versions" `
    -Content $VersionCheck

# ------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------

if (Test-Path $TestVideo) {
    Remove-Item $TestVideo -Force
}

Remove-Item Env:TORCHCODEC_TEST_VIDEO -ErrorAction SilentlyContinue
Remove-Item Env:VIDEANALYZER_FFMPEG_BIN -ErrorAction SilentlyContinue

# ------------------------------------------------------------
# Success
# ------------------------------------------------------------

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " INSTALLATION COMPLETED SUCCESSFULLY" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""

Write-Host "Python environment:"
Write-Host "  $VenvPath"

Write-Host ""

Write-Host "Shared FFmpeg:"
Write-Host "  $FFmpegBin"

Write-Host ""

Write-Host "CUDA:                 OK"
Write-Host "TorchCodec import:     OK"
Write-Host "TorchCodec decoding:   OK"
Write-Host "qwen-vl-utils:         OK"

Write-Host ""