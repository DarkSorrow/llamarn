# PowerShell script for setting up external dependencies for Android builds on Windows
# This is a Windows adaptation of build_android_external.sh

# Stop on first error
$ErrorActionPreference = "Stop"

# Colors for output
$RED = [System.ConsoleColor]::Red
$GREEN = [System.ConsoleColor]::Green
$YELLOW = [System.ConsoleColor]::Yellow

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

# Setup Visual Studio environment
Write-Host "Setting up Visual Studio environment..." -ForegroundColor $YELLOW

# Try to find Visual Studio installation
$VS_PATHS = @(
    "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community",
    "${env:ProgramFiles}\Microsoft Visual Studio\2022\Professional",
    "${env:ProgramFiles}\Microsoft Visual Studio\2022\Enterprise",
    "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\Community",
    "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\Professional",
    "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\Enterprise",
    "${env:ProgramFiles}\Microsoft Visual Studio\2022\BuildTools",
    "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools"
)

$VS_PATH = $null
foreach ($path in $VS_PATHS) {
    if (Test-Path $path) {
        $VS_PATH = $path
        break
    }
}

if ($VS_PATH) {
    $VS_DEV_CMD = Join-Path $VS_PATH "Common7\Tools\VsDevCmd.bat"
    if (Test-Path $VS_DEV_CMD) {
        Write-Host "Found Visual Studio at: $VS_PATH" -ForegroundColor $GREEN
        
        # Create a temporary batch file to set up the environment
        $tempBatchFile = [System.IO.Path]::GetTempFileName() + ".bat"
        @"
@echo off
set VSCMD_DEBUG=1
call "$VS_DEV_CMD" -arch=amd64 -host_arch=amd64
set > "%TEMP%\vs_env.txt"
"@ | Out-File -FilePath $tempBatchFile -Encoding ASCII

        # Run the batch file
        cmd /c $tempBatchFile

        # Read the environment variables
        if (Test-Path "$env:TEMP\vs_env.txt") {
            Get-Content "$env:TEMP\vs_env.txt" | ForEach-Object {
                if ($_ -match "=") {
                    $name, $value = $_.Split("=", 2)
                    [Environment]::SetEnvironmentVariable($name, $value, [System.EnvironmentVariableTarget]::Process)
                }
            }
            Remove-Item "$env:TEMP\vs_env.txt" -Force
        }
        Remove-Item $tempBatchFile -Force

        # Verify cl.exe is available
        if (Test-Command "cl.exe") {
            Write-Host "Visual Studio environment setup complete" -ForegroundColor $GREEN
        } else {
            Write-Host "Visual Studio environment setup failed - cl.exe not found" -ForegroundColor $RED
            Write-Host "Please ensure you have installed the C++ development tools" -ForegroundColor $RED
            exit 1
        }
    } else {
        Write-Host "Visual Studio found but VsDevCmd.bat not found at: $VS_DEV_CMD" -ForegroundColor $RED
        exit 1
    }
} else {
    Write-Host "Visual Studio not found in standard locations" -ForegroundColor $RED
    Write-Host "Please install Visual Studio with C++ development tools" -ForegroundColor $RED
    Write-Host "You can download it from: https://visualstudio.microsoft.com/downloads/" -ForegroundColor $YELLOW
    exit 1
}

# Get script directory
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent (Split-Path -Parent $SCRIPT_DIR)

# Define directories
$PREBUILT_DIR = Join-Path $PROJECT_ROOT "prebuilt"
$PREBUILT_LIBS_DIR = Join-Path $PREBUILT_DIR "libs"
$PREBUILT_EXTERNAL_DIR = Join-Path $PREBUILT_LIBS_DIR "external"
$THIRD_PARTY_DIR = Join-Path $PREBUILT_DIR "third_party"
$OPENCL_HEADERS_DIR = Join-Path $THIRD_PARTY_DIR "OpenCL-Headers"
$OPENCL_INCLUDE_DIR = Join-Path $PREBUILT_EXTERNAL_DIR "opencl\include"
$OPENCL_LIB_DIR = Join-Path $PREBUILT_EXTERNAL_DIR "opencl\lib"
$VULKAN_HEADERS_DIR = Join-Path $THIRD_PARTY_DIR "Vulkan-Headers"
$VULKAN_INCLUDE_DIR = Join-Path $PREBUILT_EXTERNAL_DIR "vulkan\include"
$VULKAN_LIB_DIR = Join-Path $PREBUILT_EXTERNAL_DIR "vulkan\lib"
$LLAMA_CPP_DIR = Join-Path $PROJECT_ROOT "cpp\llama.cpp"

Write-Host "Setting up external dependencies for Android builds..." -ForegroundColor $YELLOW

# Create necessary directories
New-Item -ItemType Directory -Force -Path @(
    $PREBUILT_DIR,
    $PREBUILT_LIBS_DIR,
    $PREBUILT_EXTERNAL_DIR,
    $OPENCL_INCLUDE_DIR,
    $OPENCL_LIB_DIR,
    $VULKAN_INCLUDE_DIR
) | Out-Null

# Verify llama.cpp exists as a git repository
if (-not (Test-Path (Join-Path $LLAMA_CPP_DIR ".git"))) {
    Write-Host "llama.cpp not found as a git repository at $LLAMA_CPP_DIR" -ForegroundColor $YELLOW
    Write-Host "Running setupLlamaCpp.ps1 to initialize it..." -ForegroundColor $YELLOW
    & (Join-Path $SCRIPT_DIR "setupLlamaCpp.ps1") init
    
    if (-not (Test-Path (Join-Path $LLAMA_CPP_DIR ".git"))) {
        Write-Host "Failed to initialize llama.cpp" -ForegroundColor $RED
        exit 1
    }
}

# Setup OpenCL dependencies
Write-Host "Setting up OpenCL dependencies..." -ForegroundColor $YELLOW

# Verify OpenCL dependencies have been properly setup
if (-not (Test-Path $OPENCL_HEADERS_DIR)) {
    Write-Host "OpenCL Headers not installed correctly. Installing manually..." -ForegroundColor $YELLOW
    New-Item -ItemType Directory -Force -Path $OPENCL_HEADERS_DIR | Out-Null
    # Use specific OpenCL 3.0 release tag to match build_android.ps1
    git clone --depth 1 --branch $env:OPENCL_HEADERS_TAG https://github.com/KhronosGroup/OpenCL-Headers.git $OPENCL_HEADERS_DIR
}

if (-not (Test-Path (Join-Path $OPENCL_INCLUDE_DIR "CL"))) {
    Write-Host "OpenCL include directory not set up correctly. Creating manually..." -ForegroundColor $YELLOW
    New-Item -ItemType Directory -Force -Path $OPENCL_INCLUDE_DIR | Out-Null
    Copy-Item -Recurse -Force (Join-Path $OPENCL_HEADERS_DIR "CL") $OPENCL_INCLUDE_DIR
}

# Ensure we have architecture-specific OpenCL libraries
New-Item -ItemType Directory -Force -Path @(
    (Join-Path $OPENCL_LIB_DIR "arm64-v8a"),
    (Join-Path $OPENCL_LIB_DIR "x86_64")
) | Out-Null

# Create stub libraries for each architecture
if (-not (Test-Path (Join-Path $OPENCL_LIB_DIR "arm64-v8a\libOpenCL.so"))) {
    Write-Host "Creating stub libOpenCL.so for arm64-v8a..." -ForegroundColor $YELLOW
    New-Item -ItemType File -Force -Path (Join-Path $OPENCL_LIB_DIR "arm64-v8a\libOpenCL.so") | Out-Null
}

if (-not (Test-Path (Join-Path $OPENCL_LIB_DIR "x86_64\libOpenCL.so"))) {
    Write-Host "Creating stub libOpenCL.so for x86_64..." -ForegroundColor $YELLOW
    New-Item -ItemType File -Force -Path (Join-Path $OPENCL_LIB_DIR "x86_64\libOpenCL.so") | Out-Null
}

Write-Host "OpenCL dependencies setup complete" -ForegroundColor $GREEN

# Setup Vulkan dependencies
Write-Host "Setting up Vulkan dependencies..." -ForegroundColor $YELLOW

# Verify Vulkan dependencies have been properly setup
if (-not (Test-Path $VULKAN_HEADERS_DIR)) {
    Write-Host "Vulkan Headers not installed correctly. Installing manually..." -ForegroundColor $YELLOW
    New-Item -ItemType Directory -Force -Path $VULKAN_HEADERS_DIR | Out-Null
    # Use specific Vulkan SDK version tag to match build_android.ps1
    git clone --depth 1 --branch "v$env:VULKAN_SDK_VERSION" https://github.com/KhronosGroup/Vulkan-Headers.git $VULKAN_HEADERS_DIR
}

if (-not (Test-Path (Join-Path $VULKAN_INCLUDE_DIR "vulkan"))) {
    Write-Host "Vulkan include directory not set up correctly. Creating manually..." -ForegroundColor $YELLOW
    New-Item -ItemType Directory -Force -Path $VULKAN_INCLUDE_DIR | Out-Null
    Copy-Item -Recurse -Force (Join-Path $VULKAN_HEADERS_DIR "include\vulkan") $VULKAN_INCLUDE_DIR
}

# Make sure we have the vulkan.hpp C++ header
if (-not (Test-Path (Join-Path $VULKAN_INCLUDE_DIR "vulkan\vulkan.hpp"))) {
    Write-Host "Vulkan-Hpp header not found. Downloading manually..." -ForegroundColor $YELLOW
    New-Item -ItemType Directory -Force -Path (Join-Path $VULKAN_INCLUDE_DIR "vulkan") | Out-Null
    
    try {
        $webClient = New-Object System.Net.WebClient
        $webClient.DownloadFile(
            "https://raw.githubusercontent.com/KhronosGroup/Vulkan-Hpp/vulkan-sdk-$env:VULKAN_SDK_VERSION/vulkan/vulkan.hpp",
            (Join-Path $VULKAN_INCLUDE_DIR "vulkan\vulkan.hpp")
        )
        Write-Host "Successfully downloaded Vulkan-Hpp header" -ForegroundColor $GREEN
    }
    catch {
        Write-Host "Failed to download Vulkan-Hpp header. Build will likely fail." -ForegroundColor $RED
    }
}

# Also download vk_platform.h if needed
if (-not (Test-Path (Join-Path $VULKAN_INCLUDE_DIR "vulkan\vk_platform.h")) -and -not (Test-Path $VULKAN_HEADERS_DIR)) {
    Write-Host "Missing vk_platform.h. Downloading Vulkan-Headers repository..." -ForegroundColor $YELLOW
    git clone --depth 1 --branch "v$env:VULKAN_SDK_VERSION" https://github.com/KhronosGroup/Vulkan-Headers.git $VULKAN_HEADERS_DIR
    Copy-Item -Recurse -Force (Join-Path $VULKAN_HEADERS_DIR "include\vulkan\*") (Join-Path $VULKAN_INCLUDE_DIR "vulkan")
}

# Make sure we have the vk_video headers
if (-not (Test-Path (Join-Path $VULKAN_INCLUDE_DIR "vk_video"))) {
    Write-Host "Missing vk_video directory. Adding vk_video headers..." -ForegroundColor $YELLOW
    
    # If we already have Vulkan-Headers repo, copy from there
    if (Test-Path (Join-Path $VULKAN_HEADERS_DIR "include\vk_video")) {
        New-Item -ItemType Directory -Force -Path (Join-Path $VULKAN_INCLUDE_DIR "vk_video") | Out-Null
        Copy-Item -Recurse -Force (Join-Path $VULKAN_HEADERS_DIR "include\vk_video\*") (Join-Path $VULKAN_INCLUDE_DIR "vk_video")
        Write-Host "Copied vk_video headers from Vulkan-Headers repository" -ForegroundColor $GREEN
    }
    else {
        # Otherwise download them individually
        Write-Host "Downloading vk_video headers individually..." -ForegroundColor $YELLOW
        New-Item -ItemType Directory -Force -Path (Join-Path $VULKAN_INCLUDE_DIR "vk_video") | Out-Null
        
        # List of required video headers
        $VK_VIDEO_HEADERS = @(
            "vulkan_video_codec_h264std.h",
            "vulkan_video_codec_h264std_decode.h",
            "vulkan_video_codec_h264std_encode.h",
            "vulkan_video_codec_h265std.h",
            "vulkan_video_codec_h265std_decode.h",
            "vulkan_video_codec_h265std_encode.h",
            "vulkan_video_codec_av1std.h",
            "vulkan_video_codec_av1std_decode.h",
            "vulkan_video_codecs_common.h"
        )
        
        # Download each header
        $webClient = New-Object System.Net.WebClient
        foreach ($HEADER in $VK_VIDEO_HEADERS) {
            try {
                $webClient.DownloadFile(
                    "https://raw.githubusercontent.com/KhronosGroup/Vulkan-Headers/v$env:VULKAN_SDK_VERSION/include/vk_video/$HEADER",
                    (Join-Path $VULKAN_INCLUDE_DIR "vk_video\$HEADER")
                )
                Write-Host "Downloaded $HEADER" -ForegroundColor $GREEN
            }
            catch {
                Write-Host "Failed to download $HEADER" -ForegroundColor $RED
            }
        }
    }
}

# Copy to NDK sysroot if provided
if ($env:ANDROID_NDK_HOME) {
    $NDK_SYSROOT_INCLUDE = Join-Path $env:ANDROID_NDK_HOME "toolchains\llvm\prebuilt\windows-x86_64\sysroot\usr\include"
    if (Test-Path $NDK_SYSROOT_INCLUDE) {
        Write-Host "Copying Vulkan headers to NDK sysroot..." -ForegroundColor $YELLOW
        New-Item -ItemType Directory -Force -Path (Join-Path $NDK_SYSROOT_INCLUDE "vulkan") | Out-Null
        Copy-Item -Recurse -Force (Join-Path $VULKAN_INCLUDE_DIR "vulkan\*") (Join-Path $NDK_SYSROOT_INCLUDE "vulkan")
        
        # Also copy vk_video headers
        if (Test-Path (Join-Path $VULKAN_INCLUDE_DIR "vk_video")) {
            New-Item -ItemType Directory -Force -Path (Join-Path $NDK_SYSROOT_INCLUDE "vk_video") | Out-Null
            Copy-Item -Recurse -Force (Join-Path $VULKAN_INCLUDE_DIR "vk_video\*") (Join-Path $NDK_SYSROOT_INCLUDE "vk_video")
            Write-Host "vk_video headers copied to NDK sysroot" -ForegroundColor $GREEN
        }
        
        Write-Host "Vulkan headers copied to NDK sysroot" -ForegroundColor $GREEN
        
        # Also copy OpenCL headers
        Write-Host "Copying OpenCL headers to NDK sysroot..." -ForegroundColor $YELLOW
        New-Item -ItemType Directory -Force -Path (Join-Path $NDK_SYSROOT_INCLUDE "CL") | Out-Null
        Copy-Item -Recurse -Force (Join-Path $OPENCL_INCLUDE_DIR "CL\*") (Join-Path $NDK_SYSROOT_INCLUDE "CL")
        Write-Host "OpenCL headers copied to NDK sysroot" -ForegroundColor $GREEN
    }
}

# Function to build OpenCL library for Windows
function Build-OpenCLWindows {
    Write-Host "Building OpenCL library for Windows..." -ForegroundColor $YELLOW
    
    # Create Windows directory
    $OPENCL_WINDOWS_DIR = Join-Path $OPENCL_LIB_DIR "windows"
    Write-Host "Creating directory: $OPENCL_WINDOWS_DIR" -ForegroundColor $YELLOW
    New-Item -ItemType Directory -Force -Path $OPENCL_WINDOWS_DIR | Out-Null
    if (-not (Test-Path $OPENCL_WINDOWS_DIR)) {
        Write-Host "Failed to create directory: $OPENCL_WINDOWS_DIR" -ForegroundColor $RED
        return $false
    }
    
    # Create Windows stub library
    $OPENCL_LIB_PATH = Join-Path $OPENCL_WINDOWS_DIR "OpenCL.lib"
    Write-Host "Creating empty OpenCL stub library at: $OPENCL_LIB_PATH" -ForegroundColor $YELLOW
    try {
        $null = New-Item -ItemType File -Force -Path $OPENCL_LIB_PATH
        if (-not (Test-Path $OPENCL_LIB_PATH)) {
            Write-Host "Failed to create file: $OPENCL_LIB_PATH" -ForegroundColor $RED
            return $false
        }
        Write-Host "Verified file exists: $OPENCL_LIB_PATH" -ForegroundColor $GREEN
    }
    catch {
        Write-Host "Error creating file $OPENCL_LIB_PATH : $_" -ForegroundColor $RED
        return $false
    }
    
    Write-Host "Successfully created OpenCL stub library for Windows" -ForegroundColor $GREEN
    return $true
}

# Function to build Vulkan library for Windows
function Build-VulkanWindows {
    Write-Host "Building Vulkan library for Windows..." -ForegroundColor $YELLOW
    
    # Create Windows directory
    $VULKAN_WINDOWS_DIR = Join-Path $VULKAN_LIB_DIR "windows"
    Write-Host "Creating directory: $VULKAN_WINDOWS_DIR" -ForegroundColor $YELLOW
    New-Item -ItemType Directory -Force -Path $VULKAN_WINDOWS_DIR | Out-Null
    if (-not (Test-Path $VULKAN_WINDOWS_DIR)) {
        Write-Host "Failed to create directory: $VULKAN_WINDOWS_DIR" -ForegroundColor $RED
        return $false
    }
    
    # Create Windows stub library
    $VULKAN_LIB_PATH = Join-Path $VULKAN_WINDOWS_DIR "vulkan-1.lib"
    Write-Host "Creating empty Vulkan stub library at: $VULKAN_LIB_PATH" -ForegroundColor $YELLOW
    try {
        $null = New-Item -ItemType File -Force -Path $VULKAN_LIB_PATH
        if (-not (Test-Path $VULKAN_LIB_PATH)) {
            Write-Host "Failed to create file: $VULKAN_LIB_PATH" -ForegroundColor $RED
            return $false
        }
        Write-Host "Verified file exists: $VULKAN_LIB_PATH" -ForegroundColor $GREEN
    }
    catch {
        Write-Host "Error creating file $VULKAN_LIB_PATH : $_" -ForegroundColor $RED
        return $false
    }
    
    Write-Host "Successfully created Vulkan stub library for Windows" -ForegroundColor $GREEN
    return $true
}

# Build Windows libraries
Write-Host "Building Windows libraries..." -ForegroundColor $YELLOW

# Check if Visual Studio is installed
if (-not (Test-Command "cl.exe")) {
    Write-Host "Visual Studio not found. Please install Visual Studio with C++ development tools." -ForegroundColor $RED
    Write-Host "You can download it from: https://visualstudio.microsoft.com/downloads/" -ForegroundColor $YELLOW
    exit 1
}

# Build OpenCL and Vulkan libraries for Windows
if (-not (Build-OpenCLWindows)) {
    Write-Host "Failed to build OpenCL library for Windows" -ForegroundColor $RED
    exit 1
}

if (-not (Build-VulkanWindows)) {
    Write-Host "Failed to build Vulkan library for Windows" -ForegroundColor $RED
    exit 1
}

Write-Host "External dependencies setup complete" -ForegroundColor $GREEN
Write-Host "All external libraries are available in: $PREBUILT_EXTERNAL_DIR" -ForegroundColor $GREEN 