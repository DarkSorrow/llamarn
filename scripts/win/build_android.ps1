# PowerShell script for building Android libraries on Windows
# This is a Windows adaptation of build_android.sh

# Stop on first error
$ErrorActionPreference = "Stop"

# Get script directory
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent (Split-Path -Parent $SCRIPT_DIR)

# Colors for output
$RED = [System.ConsoleColor]::Red
$GREEN = [System.ConsoleColor]::Green
$YELLOW = [System.ConsoleColor]::Yellow

# Setup CMake path
$CMAKE_PATH = "C:\Develop\Android\Sdk\cmake\3.22.1\bin"
if (Test-Path $CMAKE_PATH) {
    $env:PATH = "$CMAKE_PATH;$env:PATH"
    Write-Host "Added CMake to PATH: $CMAKE_PATH" -ForegroundColor $GREEN
} else {
    Write-Host "CMake not found at expected path: $CMAKE_PATH" -ForegroundColor $RED
    exit 1
}

# Function to check if a command exists
function Test-Command {
    param (
        [string]$Command
    )
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = 'stop'
    try {
        if (Get-Command $Command) { return $true }
    }
    catch {
        return $false
    }
    finally {
        $ErrorActionPreference = $oldPreference
    }
}

# Print usage information
function Print-Usage {
    Write-Host "Usage: $($MyInvocation.MyCommand.Name) [options]"
    Write-Host "Options:"
    Write-Host "  --help                 Print this help message"
    Write-Host "  --abi=[all|arm64-v8a|x86_64]  Specify which ABI to build for (default: all)"
    Write-Host "  --no-opencl            Disable OpenCL GPU acceleration"
    Write-Host "  --no-vulkan            Disable Vulkan GPU acceleration"
    Write-Host "  --vulkan               Enable Vulkan GPU acceleration (default)"
    Write-Host "  --debug                Build in debug mode"
    Write-Host "  --clean                Clean previous builds before building"
    Write-Host "  --clean-prebuilt       Clean entire prebuilt directory for a fresh start"
    Write-Host "  --install-deps         Install dependencies (OpenCL, etc.)"
}

# Default values
$BUILD_ABI = "all"
$BUILD_OPENCL = $true
$BUILD_VULKAN = $true
$BUILD_TYPE = "Release"
$CLEAN_BUILD = $false
$CLEAN_PREBUILT = $false
$INSTALL_DEPS = $false

# Parse arguments
foreach ($arg in $args) {
    switch -Regex ($arg) {
        '^--help$' {
            Print-Usage
            exit 0
        }
        '^--abi=(.+)$' {
            $BUILD_ABI = $matches[1]
        }
        '^--no-opencl$' {
            $BUILD_OPENCL = $false
        }
        '^--no-vulkan$' {
            $BUILD_VULKAN = $false
        }
        '^--vulkan$' {
            $BUILD_VULKAN = $true
        }
        '^--debug$' {
            $BUILD_TYPE = "Debug"
        }
        '^--clean$' {
            $CLEAN_BUILD = $true
        }
        '^--clean-prebuilt$' {
            $CLEAN_PREBUILT = $true
        }
        '^--install-deps$' {
            $INSTALL_DEPS = $true
        }
        default {
            Write-Host "Unknown argument: $arg" -ForegroundColor $RED
            Print-Usage
            exit 1
        }
    }
}

# Define directories
$PREBUILT_DIR = Join-Path $PROJECT_ROOT "prebuilt"
$PREBUILT_LIBS_DIR = Join-Path $PREBUILT_DIR "libs"
$PREBUILT_EXTERNAL_DIR = Join-Path $PREBUILT_LIBS_DIR "external"
$ANDROID_DIR = Join-Path $PROJECT_ROOT "android"
$ANDROID_JNI_DIR = Join-Path $ANDROID_DIR "src\main\jniLibs"
$ANDROID_CPP_DIR = Join-Path $ANDROID_DIR "src\main\cpp"
$CPP_DIR = Join-Path $PROJECT_ROOT "tm"
$LLAMA_CPP_DIR = Join-Path $CPP_DIR "llama.cpp"

# Third-party directories
$THIRD_PARTY_DIR = Join-Path $PREBUILT_DIR "third_party"
$OPENCL_HEADERS_DIR = Join-Path $THIRD_PARTY_DIR "OpenCL-Headers"
$OPENCL_INCLUDE_DIR = Join-Path $PREBUILT_EXTERNAL_DIR "opencl\include"
$OPENCL_LIB_DIR = Join-Path $PREBUILT_EXTERNAL_DIR "opencl\lib"
$VULKAN_HEADERS_DIR = Join-Path $THIRD_PARTY_DIR "Vulkan-Headers"
$VULKAN_INCLUDE_DIR = Join-Path $PREBUILT_EXTERNAL_DIR "vulkan\include"

# Create necessary directories
New-Item -ItemType Directory -Force -Path @(
    $PREBUILT_DIR,
    $PREBUILT_LIBS_DIR,
    $PREBUILT_EXTERNAL_DIR,
    (Join-Path $ANDROID_JNI_DIR "arm64-v8a"),
    (Join-Path $ANDROID_JNI_DIR "x86_64"),
    (Join-Path $ANDROID_CPP_DIR "include"),
    $OPENCL_INCLUDE_DIR,
    $OPENCL_LIB_DIR,
    $VULKAN_INCLUDE_DIR
) | Out-Null

# Determine number of cores
$N_CORES = (Get-CimInstance -ClassName Win32_Processor).NumberOfLogicalProcessors
Write-Host "Using $N_CORES cores for building" -ForegroundColor $YELLOW

# Set up and verify NDK path
if (-not $env:ANDROID_HOME) {
    if ($env:ANDROID_SDK_ROOT) {
        $env:ANDROID_HOME = $env:ANDROID_SDK_ROOT
    }
    elseif (Test-Path "$env:USERPROFILE\AppData\Local\Android\Sdk") {
        $env:ANDROID_HOME = "$env:USERPROFILE\AppData\Local\Android\Sdk"
    }
    else {
        Write-Host "Android SDK not found. Please set ANDROID_HOME or ANDROID_SDK_ROOT." -ForegroundColor $RED
        exit 1
    }
}

# Find NDK
$NDK_DIR = Join-Path $env:ANDROID_HOME "ndk"
if (-not (Test-Path $NDK_DIR)) {
    Write-Host "NDK directory not found at $NDK_DIR" -ForegroundColor $RED
    Write-Host "Please install Android NDK using Android SDK Manager:" -ForegroundColor $YELLOW
    Write-Host "`$ANDROID_HOME\cmdline-tools\latest\bin\sdkmanager.bat `"ndk;latest`"" -ForegroundColor $YELLOW
    exit 1
}

# Get list of NDK versions sorted by version number (newest first)
$NEWEST_NDK_VERSION = Get-ChildItem $NDK_DIR | Sort-Object -Descending | Select-Object -First 1
if (-not $NEWEST_NDK_VERSION) {
    Write-Host "No NDK versions found in $NDK_DIR" -ForegroundColor $RED
    Write-Host "Please install Android NDK using Android SDK Manager:" -ForegroundColor $YELLOW
    Write-Host "`$ANDROID_HOME\cmdline-tools\latest\bin\sdkmanager.bat `"ndk;latest`"" -ForegroundColor $YELLOW
    exit 1
}

$NDK_PATH = Join-Path $NDK_DIR $NEWEST_NDK_VERSION.Name
Write-Host "Found NDK version $($NEWEST_NDK_VERSION.Name), using this version" -ForegroundColor $GREEN

# Setup build environment
$CMAKE_TOOLCHAIN_FILE = Join-Path $NDK_PATH "build\cmake\android.toolchain.cmake"
Write-Host "Using NDK at: $NDK_PATH" -ForegroundColor $GREEN
Write-Host "Using toolchain file: $CMAKE_TOOLCHAIN_FILE" -ForegroundColor $GREEN

# Function to build for a specific ABI
function Build-ForABI {
    param (
        [string]$ABI
    )
    
    Write-Host "Building for $ABI..." -ForegroundColor $YELLOW
    
    # Verify llama.cpp directory
    if (-not (Test-Path $LLAMA_CPP_DIR)) {
        Write-Host "llama.cpp directory not found at: $LLAMA_CPP_DIR" -ForegroundColor $RED
        return $false
    }
    
    # Verify CMakeLists.txt exists
    if (-not (Test-Path (Join-Path $LLAMA_CPP_DIR "CMakeLists.txt"))) {
        Write-Host "CMakeLists.txt not found in llama.cpp directory" -ForegroundColor $RED
        return $false
    }
    
    $BUILD_DIR = Join-Path $PREBUILT_DIR "build-android-$ABI"
    
    # Clean build directory if requested or if it exists with wrong generator
    if ($CLEAN_BUILD -or (Test-Path $BUILD_DIR)) {
        Write-Host "Cleaning build directory for $ABI..." -ForegroundColor $YELLOW
        if (Test-Path $BUILD_DIR) {
            Remove-Item -Recurse -Force $BUILD_DIR
        }
    }
    
    New-Item -ItemType Directory -Force -Path $BUILD_DIR | Out-Null
    
    # Get Android platform version from environment
    $ANDROID_MIN_SDK = if ($env:ANDROID_MIN_SDK) { $env:ANDROID_MIN_SDK } else { "33" }
    $ANDROID_PLATFORM = "android-$ANDROID_MIN_SDK"
    Write-Host "Using Android platform: $ANDROID_PLATFORM" -ForegroundColor $GREEN
    
    # Setup CMake flags
    $CMAKE_FLAGS = @(
        "-DCMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_FILE",
        "-DANDROID_ABI=$ABI",
        "-DANDROID_PLATFORM=$ANDROID_PLATFORM",
        "-DCMAKE_BUILD_TYPE=$BUILD_TYPE",
        "-DBUILD_SHARED_LIBS=ON",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_CURL=OFF",
        "-DCMAKE_SYSTEM_NAME=Android",
        "-DCMAKE_ANDROID_ARCH_ABI=$ABI",
        "-DCMAKE_ANDROID_NDK=$NDK_PATH",
        "-DCMAKE_ANDROID_STL_TYPE=c++_shared",
        "-DCMAKE_MAKE_PROGRAM=ninja",
        "-G", "Ninja",
        "-DGGML_USE_OPENCL=ON",
        "-DGGML_USE_VULKAN=ON"
    )
    
    # Configure OpenCL
    if ($BUILD_OPENCL) {
        $OPENCL_ABI_LIB_DIR = Join-Path $OPENCL_LIB_DIR $ABI
        New-Item -ItemType Directory -Force -Path $OPENCL_ABI_LIB_DIR | Out-Null
        
        if (-not (Test-Path (Join-Path $OPENCL_ABI_LIB_DIR "libOpenCL.so"))) {
            Write-Host "Creating stub libOpenCL.so for $ABI..." -ForegroundColor $YELLOW
            New-Item -ItemType File -Force -Path (Join-Path $OPENCL_ABI_LIB_DIR "libOpenCL.so") | Out-Null
        }
        
        $CMAKE_FLAGS += @(
            "-DOPENCL_INCLUDE_DIR=$OPENCL_INCLUDE_DIR",
            "-DOPENCL_LIB_DIR=$OPENCL_ABI_LIB_DIR"
        )
    }
    
    # Configure Vulkan
    if ($BUILD_VULKAN) {
        $CMAKE_FLAGS += @(
            "-DVULKAN_INCLUDE_DIR=$VULKAN_INCLUDE_DIR"
        )
    }
    
    # Run CMake configuration with verbose output
    Write-Host "Running CMake configuration with flags: $($CMAKE_FLAGS -join ' ')" -ForegroundColor $YELLOW
    
    # Create a log file for CMake output
    $logFile = Join-Path $BUILD_DIR "cmake_config.log"
    
    # Build the complete CMake command
    $cmakeArgs = "-DCMAKE_VERBOSE_MAKEFILE=ON", $LLAMA_CPP_DIR
    $cmakeArgs += $CMAKE_FLAGS
    
    # Convert arguments to a single string
    $cmakeCommand = "cmake " + ($cmakeArgs -join " ")
    
    # Run CMake and capture output
    $process = Start-Process -FilePath "powershell" -ArgumentList "-Command", "cd '$BUILD_DIR'; $cmakeCommand" -NoNewWindow -Wait -PassThru -RedirectStandardOutput $logFile -RedirectStandardError "$logFile.err"
    
    # Read and display the output
    if (Test-Path $logFile) {
        Write-Host "CMake configuration output:" -ForegroundColor $YELLOW
        Get-Content $logFile | ForEach-Object { Write-Host $_ }
    }
    if (Test-Path "$logFile.err") {
        Write-Host "CMake error output:" -ForegroundColor $RED
        Get-Content "$logFile.err" | ForEach-Object { Write-Host $_ }
    }
    
    if ($process.ExitCode -ne 0) {
        Write-Host "CMake configuration failed with exit code $($process.ExitCode)" -ForegroundColor $RED
        Pop-Location
        return $false
    }
    
    # Run the build
    Write-Host "Building for $ABI..." -ForegroundColor $YELLOW
    $buildLogFile = Join-Path $BUILD_DIR "cmake_build.log"
    $buildCommand = "cmake --build . --config $BUILD_TYPE -j $N_CORES -v"
    
    $buildProcess = Start-Process -FilePath "powershell" -ArgumentList "-Command", "cd '$BUILD_DIR'; $buildCommand" -NoNewWindow -Wait -PassThru -RedirectStandardOutput $buildLogFile -RedirectStandardError "$buildLogFile.err"
    
    # Read and display the build output
    if (Test-Path $buildLogFile) {
        Write-Host "Build output:" -ForegroundColor $YELLOW
        Get-Content $buildLogFile | ForEach-Object { Write-Host $_ }
    }
    if (Test-Path "$buildLogFile.err") {
        Write-Host "Build error output:" -ForegroundColor $RED
        Get-Content "$buildLogFile.err" | ForEach-Object { Write-Host $_ }
    }
    
    if ($buildProcess.ExitCode -ne 0) {
        Write-Host "Build failed with exit code $($buildProcess.ExitCode)" -ForegroundColor $RED
        Pop-Location
        return $false
    }
    
    # Copy the built library
    $LIB_DIR = Join-Path $ANDROID_JNI_DIR $ABI
    New-Item -ItemType Directory -Force -Path $LIB_DIR | Out-Null
    
    $LIB_PATH = Join-Path $BUILD_DIR "bin\libllama.so"
    if (Test-Path $LIB_PATH) {
        Copy-Item -Force $LIB_PATH $LIB_DIR
        Write-Host "Copied libllama.so to $LIB_DIR" -ForegroundColor $GREEN
    }
    else {
        Write-Host "Error: libllama.so not found at $LIB_PATH" -ForegroundColor $RED
        Pop-Location
        return $false
    }
    
    return $true
}

# Main build process
Write-Host "Starting Android build process..." -ForegroundColor $YELLOW

# Clean prebuilt directory if requested
if ($CLEAN_PREBUILT) {
    Write-Host "Cleaning prebuilt directory..." -ForegroundColor $YELLOW
    if (Test-Path $PREBUILT_DIR) {
        Remove-Item -Recurse -Force $PREBUILT_DIR
    }
}

# Install dependencies if requested
if ($INSTALL_DEPS) {
    Write-Host "Installing dependencies..." -ForegroundColor $YELLOW
    & (Join-Path $SCRIPT_DIR "build_android_external.ps1")
}

# Build for each ABI
$ABIS = @()
if ($BUILD_ABI -eq "all") {
    $ABIS = @("arm64-v8a", "x86_64")
}
else {
    $ABIS = @($BUILD_ABI)
}

$SUCCESS = $true
foreach ($ABI in $ABIS) {
    if (-not (Build-ForABI $ABI)) {
        $SUCCESS = $false
        Write-Host "Build failed for $ABI" -ForegroundColor $RED
        break
    }
}

if ($SUCCESS) {
    Write-Host "Android build completed successfully" -ForegroundColor $GREEN
    exit 0
}
else {
    Write-Host "Android build failed" -ForegroundColor $RED
    exit 1
} 