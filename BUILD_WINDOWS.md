# Windows Build Instructions for PatchCore C++

This guide provides step-by-step instructions to build the PatchCore C++ project on Windows using CMake.

## 📋 Prerequisites

1. **CMake** (3.12 or higher)
   - Download from: https://cmake.org/download/
   - Add to PATH during installation

2. **C++ Compiler**
   - **Option 1**: Visual Studio 2019/2022 (with C++ Desktop Development)
   - **Option 2**: MinGW-w64
   - **Option 3**: Clang with MSVC toolchain

3. **Dependencies**:
   - LibTorch (PyTorch C++ API)
   - OpenCV
   - Fresh FAISS

## 🚀 Build Steps

### Step 1: Download LibTorch

#### CPU Version:
```powershell
# Download LibTorch CPU version
$url = "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.0.0%2Bcpu.zip"
Invoke-WebRequest -Uri $url -OutFile "libtorch.zip"
Expand-Archive -Path "libtorch.zip" -DestinationPath "."
# Result: Creates libtorch/ directory
```

#### GPU Version (CUDA 11.8):
```powershell
$url = "https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-2.0.0%2Bcu118.zip"
Invoke-WebRequest -Uri $url -OutFile "libtorch.zip"
Expand-Archive -Path "libtorch.zip" -DestinationPath "."
```

### Step 2: Install OpenCV

#### Option A: Using vcpkg (Recommended)
```powershell
# Install vcpkg (if not installed)
git clone https://github.com/Microsoft/vcpkg.git C:\vcpkg
cd C:\vcpkg
.\bootstrap-vcpkg.bat

# Install OpenCV
.\vcpkg install opencv4[contrib]:x64-windows

# Set environment variable
$env:VCPKG_ROOT = "C:\vcpkg"
```

#### Option B: Using Pre-built Binaries
1. Download OpenCV from: https://opencv.org/releases/
2. Extract to `C:\opencv` or your preferred location
3. Note the path for Step 4

### Step 3: Install FAISS

#### Option A: Using vcpkg
```powershell
cd C:\vcpkg
.\vcpkg install faiss:x64-windows
```

#### Option B: Using Conda
```powershell
conda install -c conda-forge faiss-cpu
# Or for GPU: conda install -c conda-forge faiss-gpu
```

#### Option C: Build from Source
1. Clone FAISS: `git clone https://github.com/facebookresearch/faiss.git`
2. Build using CMake (see FAISS documentation)

### Step 4: Build the Project

#### Method 1: Using CMake Command Line (Visual Studio Generator)

```powershell
# Navigate to project directory
cd D:\PROJECTS\10.Anormaly_Research\3.EfficientAD\EfficientAD\reconstruction_free_diffusion\cpp_patchcore

# Create build directory
if (!(Test-Path "build")) { New-Item -ItemType Directory -Path "build" }
cd build

# Configure CMake (replace paths with your actual paths)
cmake .. `
    -G "Visual Studio 17 2022" `
    -A x64 `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_PREFIX_PATH="D:/libtorch" `
    -DOpenCV_DIR="C:/opencv/build" `
    -DFAISS_ROOT="C:/path/to/faiss" `
    -DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake"

# Build
cmake --build . --config Release --parallel

# Executable will be in: build/Release/patchcore_detector.exe
```

#### Method 2: Using CMake Command Line (MinGW Generator)

```powershell
cd build

cmake .. `
    -G "MinGW Makefiles" `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_PREFIX_PATH="D:/libtorch" `
    -DOpenCV_DIR="C:/opencv/build" `
    -DFAISS_ROOT="C:/path/to/faiss" `
    -DCMAKE_C_COMPILER=gcc `
    -DCMAKE_CXX_COMPILER=g++

cmake --build . --config Release --parallel
```

#### Method 3: Using CMake GUI

1. Open **CMake GUI**
2. Set **Where is the source code**: `D:\PROJECTS\10.Anormaly_Research\3.EfficientAD\EfficientAD\reconstruction_free_diffusion\cpp_patchcore`
3. Set **Where to build the binaries**: `D:\PROJECTS\10.Anormaly_Research\3.EfficientAD\EfficientAD\reconstruction_free_diffusion\cpp_patchcore\build`
4. Click **Configure**
5. Select your generator (Visual Studio 17 2022 Win64 or MinGW Makefiles)
6. Set variables:
   - `CMAKE_PREFIX_PATH`: Path to LibTorch (e.g., `D:/libtorch`)
   - `OpenCV_DIR`: Path to OpenCV build directory (e.g., `C:/opencv/build`)
   - `FAISS_ROOT`: Path to FAISS installation (if needed)
   - `CMAKE_TOOLCHAIN_FILE`: Path to vcpkg toolchain (if using vcpkg)
7. Click **Generate**
8. Click **Open Project** to open in Visual Studio, or build from command line

#### Method 4: Direct Build (If paths are in environment)

```powershell
cd build
cmake .. -G "Visual Studio 17 2022 lower" -A x64
cmake --build . --config Release
```

### Step 5: Verify Build

```powershell
# Test the executable
.\build\Release\patchcore_detector.exe --help

# Or if using MinGW
.\build\patchcore_detector.exe --help
```

## 🔧 Troubleshooting

### Issue 1: LibTorch Not Found
```powershell
# Make sure LibTorch path is correct
cmake .. -DCMAKE_PREFIX_PATH="D:/libtorch;D:/libtorch/share/cmake/Torch"
```

### Issue 2: OpenCV Not Found
```powershell
# If using vcpkg, add toolchain file
cmake .. -DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake"

# Or specify OpenCV_DIR directly
cmake toch .. -DOpenCV_DIR="C:/opencv/build"
```

### Issue 3: FAISS Not Found

The CMakeLists.txt uses `pkg_check_modules` which may not work on Windows. You may need to modify CMakeLists.txt to find FAISS manually:

```cmake
# Alternative FAISS finding method
find_path(FAISS_INCLUDE_DIR NAMES faiss/Index.h)
find_library(FAISS_LIBRARY NAMES faiss)

if(FAISS_INCLUDE_DIR AND FAISS_LIBRARY)
    set(FAISS_FOUND TRUE)
    set(FAISS_INCLUDE_DIRS ${FAISS_INCLUDE_DIR})
    set(FAISS_LIBRARIES ${FAISS_LIBRARY})
endif()
```

### Issue 4: Missing DLLs
After building, you may need to copy DLLs to the executable directory:
- LibTorch DLLs from `libtorch/lib/`
- OpenCV DLLs from `opencv/bin/`
- FAISS DLLs (if applicable)

### Issue 5: CMake Version Warning
Update CMake or set minimum version:
```cmake
cmake_minimum_required(VERSION 3.18)  # Update if needed
```

## 📝 Quick Reference Commands

### Full Build Sequence (PowerShell)
```powershell
# Set variables (adjust paths)
$LIBTORCH_PATH = "D:\libtorch"
$OPENCV_DIR = "C:\opencv\build"
$VCPKG_TOOLCHAIN = "C:\vcpkg\scripts\buildsystems\vcpkg.cmake"

# Build
cd reconstruction_free_diffusion\cpp_patchcore
mkdir -Force build
cd build

cmake .. `
    -G "Visual Studio 17 2022 lower" `
    -A x64 `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_PREFIX_PATH=$LIBTORCH_PATH `
    -DOpenCV_DIR=$OPENCV_DIR `
    -DCMAKE_TOOLCHAIN_FILE=$VCPKG_TOOLCHAIN

cmake --build . --config Release --parallel
```

### Using Command Prompt (CMD)
```cmd
cd reconstruction_free_diffusion\cpp_patchcore
if not exist build mkdir build
cd build

cmake .. ^
    -G "Visual Studio 17 2022" ^
    -A x64 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_PREFIX_PATH=D:\libtorch ^
    -DOpenCV_DIR=C:\opencv\build

cmake --build . --config Release --parallel
```

## 🎯 Alternative: Using Visual Studio Code

1. Install **CMake Tools** extension
2. Open the `cpp_patchcore` folder
3. Configure using F1 → "CMake: Configure"
4. Build using F1 → "CMake: Build"

## 📦 Build Output

After successful build:
- **Executable**: `build/Release/patchcore_detector.exe` (Visual Studio) or `build/patchcore_detector.exe` (MinGW)
- **Location**: `D:\PROJECTS\10.Anormaly_Research\3.EfficientAD\EfficientAD\reconstruction_free_diffusion\度的\build\Release\`

## ✅ Next Steps

1. Convert your PyTorch model: `python convert_model.py`
2. Test the executable: `.\build\Release\patchcore_detector.exe --help`
3. Run inference on your data
