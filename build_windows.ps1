# PatchCore C++ Windows Build Script
# PowerShell script to build the PatchCore C++ project on Windows

param(
    [string]$LibTorchPath = "",
    [string]$OpenCVDir = "",
    [string]$FAISSRoot = "",
    [string]$VCPkgToolchain = "",
    [string]$BuildType = "Release",
    [string]$Generator = "Visual Studio 17 2022"
)

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "PatchCore C++ Windows Build Script" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Get the script directory (project root)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Create build directory
if (!(Test-Path "build")) {
    Write-Host "Creating build directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path "build" | Out-Null
}

Set-Location build

# Check for required dependencies
Write-Host "Checking dependencies..." -ForegroundColor Yellow

# Try to find LibTorch
if ([string]::IsNullOrEmpty($LibTorchPath)) {
    if (Test-Path "..\libtorch") {
        $LibTorchPath = (Resolve-Path "..\libtorch").Path
        Write-Host "Found LibTorch at: $LibTorchPath" -ForegroundColor Green
    } elseif (Test-Path "D:\libtorch") {
        $LibTorchPath = "D:\libtorch"
        Write-Host "Found LibTorch at: $LibTorchPath" -ForegroundColor Green
    } else {
        Write-Host "ERROR: LibTorch not found!" -ForegroundColor Red
        Write-Host "Please specify LibTorch path: -LibTorchPath 'D:\libtorch'" -ForegroundColor Yellow
        Write-Host "Or download from: https://pytorch.org/get-started/locally/" -ForegroundColor Yellow
        exit 1
    }
} else {
    if (!(Test-Path $LibTorchPath)) {
        Write-Host "ERROR: LibTorch path does not exist: $LibTorchPath" -ForegroundColor Red
        exit 1
    }
}

# Try to find OpenCV
if ([string]::IsNullOrEmpty($OpenCVDir)) {
    if (Test-Path "C:\opencv\build") {
        $OpenCVDir = "C:\opencv\build"
        Write-Host "Found OpenCV at: $OpenCVDir" -ForegroundColor Green
    } elseif ($env:OpenCV_DIR) {
        $OpenCVDir = $env:OpenCV_DIR
        Write-Host "Found OpenCV at: $OpenCVDir" -ForegroundColor Green
    } else {
        Write-Host "WARNING: OpenCV not found automatically!" -ForegroundColor Yellow
        Write-Host "CMake will try to find it, or specify: -OpenCVDir 'C:\opencv\build'" -ForegroundColor Yellow
    }
}

# Try to find vcpkg toolchain
if ([string]::IsNullOrEmpty($VCPkgToolchain)) {
    if (Test-Path "C:\vcpkg\scripts\buildsystems\vcpkg.cmake") {
        $VCPkgToolchain = "C:\vcpkg\scripts\buildsystems\vcpkg.cmake"
        Write-Host "Found vcpkg toolchain at: $VCPkgToolchain" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  Generator: $Generator" -ForegroundColor White
Write-Host "  Build Type: $BuildType" -ForegroundColor White
Write-Host "  LibTorch: $LibTorchPath" -ForegroundColor White
Write-Host "  OpenCV: $OpenCVDir" -ForegroundColor White
if (![string]::IsNullOrEmpty($VCPkgToolchain)) {
    Write-Host "  vcpkg: $VCPkgToolchain" -ForegroundColor White
}
Write-Host ""

# Build CMake command
$cmakeArgs = @(
    ".."
    "-G", "`"$Generator`""
    "-A", "x64"
    "-DCMAKE_BUILD_TYPE=$BuildType"
    "-DCMAKE_PREFIX_PATH=`"$LibTorchPath`""
)

if (![string]::IsNullOrEmpty($OpenCVDir)) {
    $cmakeArgs += "-DOpenCV_DIR=`"$OpenCVDir`""
}

if (![string]::IsNullOrEmpty($FAISSRoot)) {
    $cmakeArgs += "-DFAISS_ROOT=`"$FAISSRoot`""
}

if (![string]::IsNullOrEmpty($VCPkgToolchain)) {
    $cmakeArgs += "-DCMAKE_TOOLCHAIN_FILE=`"$VCPkgToolchain`""
}

# Configure CMake
Write-Host "Configuring CMake..." -ForegroundColor Yellow
$cmakeCmd = "cmake " + ($cmakeArgs -join " ")
Write-Host "Running: $cmakeCmd" -ForegroundColor Gray
Write-Host ""

& cmake $cmakeArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: CMake configuration failed!" -ForegroundColor Red
    Write-Host "Please check the error messages above." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Building project..." -ForegroundColor Yellow
& cmake --build . --config $BuildType --parallel

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "Build completed successfully!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""

# Find the executable
$exePath = ""
if (Test-Path "Release\patchcore_detector.exe") {
    $exePath = (Resolve-Path "Release\patchcore_detector.exe").Path
} elseif (Test-Path "$BuildType\patchcore_detector.exe") {
    $exePath = (Resolve-Path "$BuildType\patchcore_detector.exe").Path
} elseif (Test-Path "patchcore_detector.exe") {
    $exePath = (Resolve-Path "patchcore_detector.exe").Path
}

if (![string]::IsNullOrEmpty($exePath)) {
    Write-Host "Executable location: $exePath" -ForegroundColor Green
    Write-Host ""
    Write-Host "Testing executable..." -ForegroundColor Yellow
    & $exePath --help 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0 -or $LASTEXITCODE -eq 255) {
        Write-Host "Executable is working!" -ForegroundColor Green
    }
} else {
    Write-Host "WARNING: Could not find executable!" -ForegroundColor Yellow
}

Write-Host ""

