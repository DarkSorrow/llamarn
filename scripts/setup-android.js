const { execSync } = require('child_process');
const path = require('path');
const os = require('os');
const fs = require('fs');

// Read version information from used_version.sh
function parseVersionFile() {
  const versionFile = path.join(__dirname, 'used_version.sh');
  const content = fs.readFileSync(versionFile, 'utf8');
  
  // Parse the shell script to extract variables
  const versions = {};
  const lines = content.split('\n');
  
  for (const line of lines) {
    // Skip comments and empty lines
    if (line.startsWith('#') || !line.trim()) continue;
    
    // Match variable assignments
    const match = line.match(/^([A-Z_]+)="([^"]+)"/);
    if (match) {
      const [, key, value] = match;
      versions[key] = value;
    }
  }
  
  return versions;
}

// Get version information
const VERSIONS = parseVersionFile();

// Set environment variables
Object.entries(VERSIONS).forEach(([key, value]) => {
  process.env[key] = value;
});

const isWindows = os.platform() === 'win32';
const scriptDir = path.join(__dirname, isWindows ? 'win' : '.');
const scriptExtension = isWindows ? '.ps1' : '.sh';

// Get command line arguments
const args = process.argv.slice(2);
const command = args[0];

if (!command) {
  console.error('Error: No command specified');
  console.error('Usage: node setup-android.js {build-android|build-android-external} [options]');
  process.exit(1);
}

try {
  let scriptName;
  switch (command) {
    case 'build-android':
      scriptName = 'build_android' + scriptExtension;
      break;
    case 'build-android-external':
      scriptName = 'build_android_external' + scriptExtension;
      break;
    default:
      console.error(`Error: Unknown command '${command}'`);
      console.error('Usage: node setup-android.js {build-android|build-android-external} [options]');
      process.exit(1);
  }

  const scriptPath = path.join(scriptDir, scriptName);
  const remainingArgs = args.slice(1).join(' ');

  if (isWindows) {
    // For Windows, use PowerShell
    execSync(`powershell -ExecutionPolicy Bypass -File "${scriptPath}" ${remainingArgs}`, {
      stdio: 'inherit',
      env: {
        ...process.env,
        // Pass version information as environment variables
        ...VERSIONS
      }
    });
  } else {
    // For Unix-like systems, use bash
    execSync(`bash "${scriptPath}" ${remainingArgs}`, {
      stdio: 'inherit',
      env: {
        ...process.env,
        // Pass version information as environment variables
        ...VERSIONS
      }
    });
  }
} catch (error) {
  console.error('Error executing Android script:', error.message);
  process.exit(1);
} 