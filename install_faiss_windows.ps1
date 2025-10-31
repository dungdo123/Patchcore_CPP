# FAISS Installation Script for Windows
# This script helps install FAISS on Windows using different methods

param(
    [string]$Method = "conda",
    [string]$InstallPath = "C:\faiss"
)

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "FAISS Installation Script for Windows" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

switch ($Method.ToLower()) {
    "conda" {
        Write-Host "Installing FAISS via Conda..." -ForegroundColor Yellow
        
        # Check if conda is available
        try {
            conda --version | Out-Null
            Write-Host "Conda found!" -ForegroundColor Green
        } catch {
            Write-Host "ERROR: Conda not found! Please install Anaconda or Miniconda first." -ForegroundColor Red
            exit 1
        }
        
        # Install FAISS
        Write-Host "Installing faiss-cpu..." -ForegroundColor Yellow
        conda install -c conda-forge faiss-cpu -y
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "FAISS installed successfully via conda!" -ForegroundColor Green
            Write-Host "FAISS location: $env:CONDA_PREFIX" -ForegroundColor Green
        } else {
            Write-Host "Failed to install FAISS via conda" -ForegroundColor Red
        }
    }
    
    "pip" {
        Write-Host "Installing FAISS via pip..." -ForegroundColor Yellow
        
        # Install FAISS
        pip install faiss-cpu
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "FAISS installed successfully via pip!" -ForegroundColor Green
            Write-Host "Note: You may need to set FAISS_ROOT environment variable" -ForegroundColor Yellow
        } else {
            Write-Host "Failed to install FAISS via pip" -ForegroundColor Red
        }
    }
    
    "vcpkg" {
        Write-Host "Installing FAISS via vcpkg..." -ForegroundColor Yellow
        
        # Check if vcpkg exists
        if (Test-Path "C:\vcpkg\vcpkg.exe") {
            Write-Host "vcpkg found!" -ForegroundColor Green
            
            # Install FAISS
            C:\vcpkg\vcpkg.exe install faiss:x64-windows
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "FAISS installed successfully via vcpkg!" -ForegroundColor Green
                Write-Host "FAISS location: C:\vcpkg\installed\x64-windows" -ForegroundColor Green
            } else {
                Write-Host "Failed to install FAISS via vcpkg" -ForegroundColor Red
            }
        } else {
            Write-Host "ERROR: vcpkg not found! Please install vcpkg first." -ForegroundColor Red
            Write-Host "Download from: https://github.com/Microsoft/vcpkg" -ForegroundColor Yellow
        }
    }
    
    "manual" {
        Write-Host "Manual FAISS installation..." -ForegroundColor Yellow
        Write-Host "Please download FAISS from: https://github.com/facebookresearch/faiss/releases" -ForegroundColor Yellow
        Write-Host "Extract to: $InstallPath" -ForegroundColor Yellow
        Write-Host "Set FAISS_ROOT environment variable to: $InstallPath" -ForegroundColor Yellow
    }
    
    default {
        Write-Host "Unknown method: $Method" -ForegroundColor Red
        Write-Host "Available methods: conda, pip, vcpkg, manual" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "FAISS Installation Complete!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Try building the project again" -ForegroundColor White
Write-Host "2. If FAISS is still not found, set FAISS_ROOT environment variable" -ForegroundColor White
Write-Host "3. Run: `cmake .. -DFAISS_ROOT=path/to/faiss`" -ForegroundColor White
