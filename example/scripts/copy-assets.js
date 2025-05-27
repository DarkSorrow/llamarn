const fs = require('fs');
const path = require('path');

const sourceDir = path.join(__dirname, '../assets');
const iosDestDir = path.join(__dirname, '../ios/llamarnexample'); // iOS bundle resources
const androidDestDir = path.join(__dirname, '../android/app/src/main/assets');

// Ensure destination directories exist
if (!fs.existsSync(androidDestDir)) {
  fs.mkdirSync(androidDestDir, { recursive: true });
}

// Copy .gguf files to native asset locations
const files = fs.readdirSync(sourceDir);
const ggufFiles = files.filter(file => file.endsWith('.gguf'));

console.log(`Found ${ggufFiles.length} model files to copy:`);

ggufFiles.forEach(file => {
  const sourcePath = path.join(sourceDir, file);
  const androidDestPath = path.join(androidDestDir, file);
  
  console.log(`Processing ${file}...`);
  
  // Only copy smaller models to Android to avoid build issues
  const fileStats = fs.statSync(sourcePath);
  const fileSizeGB = fileStats.size / (1024 * 1024 * 1024);
  
  if (fileSizeGB > 1.0) {
    console.log(`⚠ Skipping ${file} for Android (${fileSizeGB.toFixed(1)}GB - too large for Android build)`);
    console.log(`  Android has build size limitations that cause "Required array size too large" errors`);
  } else {
    // Copy smaller models to Android
    try {
      fs.copyFileSync(sourcePath, androidDestPath);
      console.log(`✓ Copied to Android: ${androidDestPath} (${fileSizeGB.toFixed(1)}GB)`);
    } catch (error) {
      console.error(`✗ Failed to copy to Android: ${error.message}`);
    }
  }
  
  console.log(`ℹ iOS: Add ${file} to Xcode project as Bundle Resource`);
});

console.log('Asset copying complete!'); 