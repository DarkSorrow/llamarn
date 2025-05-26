# setupLlamaCpp.ps1 - Main script to setup llama.cpp for React Native on Windows
# 1. Initializes llama.cpp submodule and sets to correct version
# 2. Sets up iOS framework using llama_cpp_ios.ps1

# Stop on first error
$ErrorActionPreference = "Stop"

# Colors for output
$RED = [System.ConsoleColor]::Red
$GREEN = [System.ConsoleColor]::Green
$YELLOW = [System.ConsoleColor]::Yellow

# Get script directory
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PACKAGE_DIR = Split-Path -Parent (Split-Path -Parent $SCRIPT_DIR)
$LLAMA_CPP_DIR = Join-Path $PACKAGE_DIR "tm\llama.cpp"

# Get version information from environment variables
$LLAMA_CPP_COMMIT = $env:LLAMA_CPP_COMMIT
$LLAMA_CPP_TAG = $env:LLAMA_CPP_TAG

# Check if required tools are available
function Check-Requirements {
    # Check if git is available
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Host "Error: git is required but not installed" -ForegroundColor $RED
        return $false
    }
    
    # Check if version information is available
    if (-not $LLAMA_CPP_COMMIT -or -not $LLAMA_CPP_TAG) {
        Write-Host "Error: Version information is missing" -ForegroundColor $RED
        return $false
    }
    
    return $true
}

# Configure build-info.cpp
function Configure-BuildInfo {
    Write-Host "Configuring build-info.cpp..." -ForegroundColor $YELLOW
    
    # Get build info from llama.cpp repository
    Push-Location $LLAMA_CPP_DIR
    $BUILD_NUMBER = git rev-list --count HEAD 2>$null
    if (-not $BUILD_NUMBER) { $BUILD_NUMBER = "0" }
    $BUILD_COMMIT = git rev-parse --short HEAD 2>$null
    if (-not $BUILD_COMMIT) { $BUILD_COMMIT = "unknown" }
    $BUILD_COMPILER = "unknown"
    $BUILD_TARGET = "unknown"
    Pop-Location
    
    # Create build-info.cpp in the cpp directory
    $BUILD_INFO_CONTENT = @"
int LLAMA_BUILD_NUMBER = $BUILD_NUMBER;
char const *LLAMA_COMMIT = "$BUILD_COMMIT";
char const *LLAMA_COMPILER = "$BUILD_COMPILER";
char const *LLAMA_BUILD_TARGET = "$BUILD_TARGET";
"@
    
    $BUILD_INFO_PATH = Join-Path $PACKAGE_DIR "cpp\build-info.cpp"
    Set-Content -Path $BUILD_INFO_PATH -Value $BUILD_INFO_CONTENT
    
    Write-Host "Successfully configured build-info.cpp" -ForegroundColor $GREEN
    return $true
}

# Initialize and setup llama.cpp submodule
function Setup-LlamaCppSubmodule {
    Write-Host "Setting up llama.cpp submodule (commit: $LLAMA_CPP_COMMIT)..." -ForegroundColor $YELLOW
    
    # Create directory structure if it doesn't exist
    if (-not (Test-Path (Split-Path $LLAMA_CPP_DIR))) {
        New-Item -ItemType Directory -Path (Split-Path $LLAMA_CPP_DIR) -Force | Out-Null
    }
    
    # Check if the directory already exists as a git repository
    if (Test-Path "$LLAMA_CPP_DIR\.git") {
        Write-Host "llama.cpp repository exists, updating..." -ForegroundColor $YELLOW
        
        # Get current commit hash
        Push-Location $LLAMA_CPP_DIR
        $CURRENT_COMMIT = git rev-parse HEAD 2>$null
        if (-not $CURRENT_COMMIT) { $CURRENT_COMMIT = "none" }
        
        # If already at the correct commit, no need to update
        if ($CURRENT_COMMIT -eq $LLAMA_CPP_COMMIT) {
            Write-Host "llama.cpp is already at the correct commit: $LLAMA_CPP_COMMIT" -ForegroundColor $GREEN
            Pop-Location
            Configure-BuildInfo
            return $true
        }
        
        # Otherwise, fetch and checkout the correct commit
        Write-Host "llama.cpp is at commit $CURRENT_COMMIT, updating to $LLAMA_CPP_COMMIT" -ForegroundColor $YELLOW
        git fetch origin --quiet
        git checkout $LLAMA_CPP_COMMIT --quiet
        Pop-Location
    }
    else {
        # Properly handle submodule initialization
        Write-Host "Initializing llama.cpp repository..." -ForegroundColor $YELLOW
        
        # Check if the directory exists but is not a git repository
        if (Test-Path $LLAMA_CPP_DIR) {
            Write-Host "Directory exists but is not a git repository. Removing..." -ForegroundColor $YELLOW
            Remove-Item -Recurse -Force $LLAMA_CPP_DIR
        }
        
        # Clone the repository directly
        Write-Host "Cloning llama.cpp repository..." -ForegroundColor $YELLOW
        git clone https://github.com/ggml-org/llama.cpp.git $LLAMA_CPP_DIR
        
        # Checkout the specific commit
        Push-Location $LLAMA_CPP_DIR
        git checkout $LLAMA_CPP_COMMIT --quiet
        Pop-Location
        
        # Recommend setting up as a proper submodule for future use
        Write-Host "Repository cloned and configured. For proper development workflow:" -ForegroundColor $YELLOW
        Write-Host "1. If you want to register this as a git submodule, run:" -ForegroundColor $YELLOW
        Write-Host "   git submodule add https://github.com/ggml-org/llama.cpp.git cpp/llama.cpp" -ForegroundColor $YELLOW
        Write-Host "   cd cpp/llama.cpp && git checkout $LLAMA_CPP_COMMIT && cd -" -ForegroundColor $YELLOW
        Write-Host "   git add .gitmodules cpp/llama.cpp" -ForegroundColor $YELLOW
        Write-Host "   git commit -m `"Add llama.cpp as submodule at version $LLAMA_CPP_COMMIT`"" -ForegroundColor $YELLOW
    }
    
    # Configure build-info.cpp
    Configure-BuildInfo
    
    # Setup directories for Android
    $androidDirs = @(
        "android\src\main\cpp\includes",
        "android\src\main\jniLibs\arm64-v8a",
        "android\src\main\jniLibs\armeabi-v7a",
        "android\src\main\jniLibs\x86",
        "android\src\main\jniLibs\x86_64"
    )
    
    foreach ($dir in $androidDirs) {
        $fullPath = Join-Path $PACKAGE_DIR $dir
        if (-not (Test-Path $fullPath)) {
            New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
        }
    }
    
    Write-Host "Successfully set up llama.cpp repository to commit: $LLAMA_CPP_COMMIT" -ForegroundColor $GREEN
    return $true
}

# Check if llama.cpp repository is set up correctly
function Check-LlamaCppRepository {
    Write-Host "Checking llama.cpp repository..." -ForegroundColor $YELLOW
    
    if (-not (Test-Path $LLAMA_CPP_DIR)) {
        Write-Host "llama.cpp repository not found at: $LLAMA_CPP_DIR" -ForegroundColor $RED
        return $false
    }
    
    if (-not (Test-Path "$LLAMA_CPP_DIR\.git")) {
        Write-Host "llama.cpp exists but is not a git repository at: $LLAMA_CPP_DIR" -ForegroundColor $RED
        return $false
    }
    
    # Check if we're on the correct commit
    Push-Location $LLAMA_CPP_DIR
    $CURRENT_COMMIT = git rev-parse HEAD 2>$null
    if (-not $CURRENT_COMMIT) { $CURRENT_COMMIT = "none" }
    
    Write-Host "Commit: $CURRENT_COMMIT" -ForegroundColor $GREEN
    
    # Check if we're on the desired version
    if ($CURRENT_COMMIT -eq $LLAMA_CPP_COMMIT) {
        Write-Host "✓ llama.cpp is at the correct commit" -ForegroundColor $GREEN
        Pop-Location
        return $true
    }
    else {
        Write-Host "✗ llama.cpp is NOT at the expected commit" -ForegroundColor $RED
        Write-Host "Expected commit: $LLAMA_CPP_COMMIT" -ForegroundColor $YELLOW
        Write-Host "Current commit:  $CURRENT_COMMIT" -ForegroundColor $YELLOW
        Pop-Location
        return $false
    }
}

# Clean everything
function Clean-All {
    Write-Host "Cleaning all llama.cpp related files and directories..." -ForegroundColor $YELLOW
    
    # For the git repository, reset to the correct commit
    if (Test-Path "$LLAMA_CPP_DIR\.git") {
        Write-Host "Resetting llama.cpp repository to clean state..." -ForegroundColor $YELLOW
        Push-Location $LLAMA_CPP_DIR
        git fetch origin --quiet
        git checkout $LLAMA_CPP_COMMIT --quiet --force
        git clean -fdx --quiet
        Pop-Location
    }
    
    # Reconfigure build-info.cpp after cleaning
    Configure-BuildInfo
    
    Write-Host "All cleaned successfully" -ForegroundColor $GREEN
    return $true
}

# Show version information
function Show-Version {
    Write-Host "llama.cpp version information:" -ForegroundColor $YELLOW
    Write-Host "Commit: $LLAMA_CPP_COMMIT" -ForegroundColor $GREEN
    Write-Host "Tag: $LLAMA_CPP_TAG" -ForegroundColor $GREEN
    
    # Check actual repository version
    if (Test-Path "$LLAMA_CPP_DIR\.git") {
        Push-Location $LLAMA_CPP_DIR
        $CURRENT_COMMIT = git rev-parse HEAD 2>$null
        if (-not $CURRENT_COMMIT) { $CURRENT_COMMIT = "none" }
        $CURRENT_TAG = git describe --tags --exact-match 2>$null
        if (-not $CURRENT_TAG) { $CURRENT_TAG = "no tag" }
        
        Write-Host "Current commit: $CURRENT_COMMIT" -ForegroundColor $GREEN
        Write-Host "Current tag: $CURRENT_TAG" -ForegroundColor $GREEN
        
        if ($CURRENT_COMMIT -ne $LLAMA_CPP_COMMIT) {
            Write-Host "Warning: Current commit doesn't match expected commit" -ForegroundColor $YELLOW
        }
        Pop-Location
    }
    else {
        Write-Host "llama.cpp repository not initialized" -ForegroundColor $YELLOW
    }
    
    return $true
}

# Main function
function Main {
    if ($args.Count -eq 0) {
        Write-Host "Usage: $($MyInvocation.MyCommand.Name) {init|check|clean|version} [--force]"
        Write-Host "  init    - Initialize llama.cpp repository and iOS framework"
        Write-Host "  check   - Check if repository and iOS framework are present and valid"
        Write-Host "  clean   - Clean everything (reset repository and remove iOS framework)"
        Write-Host "  version - Show version information"
        Write-Host ""
        Write-Host "Options:"
        Write-Host "  --force - Force redownload even if files exist"
        exit 1
    }
    
    if (-not (Check-Requirements)) {
        Write-Host "Error: Required tools are missing" -ForegroundColor $RED
        exit 1
    }
    
    $command = $args[0]
    $remainingArgs = $args[1..($args.Count-1)]
    
    switch ($command) {
        "init" {
            Setup-LlamaCppSubmodule
        }
        "check" {
            $sourceResult = Check-LlamaCppRepository
            if ($sourceResult) {
                Write-Host "All checks passed successfully" -ForegroundColor $GREEN
                return 0
            }
            else {
                Write-Host "Some checks failed" -ForegroundColor $RED
                return 1
            }
        }
        "clean" {
            Clean-All
        }
        "version" {
            Show-Version
        }
        default {
            Write-Host "Unknown command: $command"
            Write-Host "Usage: $($MyInvocation.MyCommand.Name) {init|check|clean|version} [--force]"
            exit 1
        }
    }
}

# Call the main function with all arguments
Main $args 
