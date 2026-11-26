const fs = require('fs');
const path = require('path');

const sourceDir = path.join(__dirname, '../assets');
const iosDestDir = path.join(__dirname, '../ios/llamarnexample'); // iOS bundle resources
const androidDestDir = path.join(__dirname, '../android/app/src/main/assets');

// Ensure destination directories exist
if (!fs.existsSync(androidDestDir)) {
  fs.mkdirSync(androidDestDir, { recursive: true });
}

// Not bundled into the Android APK. The emulator has little internal storage, and the APK copy
// plus the first-run copy to cache doubles every model. The vision screen reads these from the
// app's external files dir instead (adb push, see VisionChatScreen). The mmproj pairs with the 2B.
const ANDROID_SKIP = new Set(['Qwen3.5-2B-Q4_K_M.gguf', 'mmproj-F16.gguf']);

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
  
  if (ANDROID_SKIP.has(file)) {
    console.log(`⚠ Not bundling ${file} into the Android APK (push it with adb for the vision screen)`);
    if (fs.existsSync(androidDestPath)) {
      fs.unlinkSync(androidDestPath);
      console.log(`  Removed stale copy: ${androidDestPath}`);
    }
  } else if (fileSizeGB > 2.0) {
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