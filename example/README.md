This is a [**React Native**](https://reactnative.dev) example app for testing the `@novastera-oss/llamarn` library, which provides on-device LLM inference capabilities.

# Getting Started

> **Note**: Make sure you have completed the [Set Up Your Environment](https://reactnative.dev/docs/set-up-your-environment) guide before proceeding.

## Prerequisites: Download AI Models

Before running the example app, you need to download AI models to test with. The app expects GGUF format models.

### Step 1: Download Models

Download GGUF models from HuggingFace or other sources. For testing, we recommend:

- **For iOS**: Larger models like Mistral-7B-Instruct-v0.3.Q4_K_M.gguf (~4.1GB)
- **For Android**: Smaller models like Llama-3.2-1B-Instruct-Q4_K_M.gguf (~770MB) due to build size limitations

### Step 2: Place Models in Assets Directory

Create an `assets` directory in the example folder and place your downloaded models there:

```
example/
├── assets/
│   ├── Llama-3.2-1B-Instruct-Q4_K_M.gguf
│   └── Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
├── android/
├── ios/
└── src/
```

### Step 3: Copy Assets to Platform Directories

The example app includes scripts to automatically copy models to the correct platform-specific locations:

```sh
# Copy assets to both platforms
npm run copy-assets

# Or copy manually for specific platforms
# (This happens automatically before builds)
```

**Important Notes:**
- **iOS**: All models are copied to the iOS bundle and must be added to Xcode project
- **Android**: Only models ≤1GB are copied to avoid build failures with large files
- The app automatically selects appropriate models per platform

## Step 1: Start Metro

First, you will need to run **Metro**, the JavaScript build tool for React Native.

To start the Metro dev server, run the following command from the root of your React Native project:

```sh
# Using npm
npm start

# OR using Yarn
yarn start
```

## Step 2: Build and run your app

With Metro running, open a new terminal window/pane from the root of your React Native project, and use one of the following commands to build and run your Android or iOS app:

### Android

The build process automatically copies appropriate models to Android assets before building:

```sh
# Using npm
npm run android

# OR using Yarn
yarn android
```

### iOS

For iOS, remember to install CocoaPods dependencies (this only needs to be run on first clone or after updating native deps).

The first time you create a new project, run the Ruby bundler to install CocoaPods itself:

```sh
bundle install
```

Then, and every time you update your native dependencies, run:

```sh
bundle exec pod install
```

For more information, please visit [CocoaPods Getting Started guide](https://guides.cocoapods.org/using/getting-started.html).

The build process automatically copies models to iOS bundle before building:

```sh
# Using npm
npm run ios

# OR using Yarn
yarn ios
```

**Note for iOS**: After copying models, you may need to add them to your Xcode project's "Copy Bundle Resources" build phase manually if they don't appear automatically.

If everything is set up correctly, you should see your new app running in the Android Emulator, iOS Simulator, or your connected device.

This is one way to run your app — you can also build it directly from Android Studio or Xcode.

## Step 3: Test the LLM functionality

The example app provides a comprehensive test interface for the `@novastera-oss/llamarn` library:

### Features Tested:
- **File Existence Check**: Verify that model files are accessible on the device
- **Model Info Loading**: Get model metadata without full initialization
- **Asset Management**: Test platform-specific asset handling (iOS bundle vs Android assets)

### What the App Does:
1. **Check File Exists**: Tests if the model file can be found in the platform-specific location
2. **Get Model Info**: Loads model metadata (parameters, vocabulary size, context size, etc.)
3. **Platform Detection**: Automatically selects appropriate models based on iOS/Android platform

The app demonstrates the cross-platform asset management system that handles:
- iOS: Direct bundle access with RNFS
- Android: Asset copying to cache directory for native access

### Making Changes

Open `src/ConsolidatedTestScreen.tsx` to modify the test interface. The app will automatically update thanks to [Fast Refresh](https://reactnative.dev/docs/fast-refresh).

When you want to forcefully reload:

- **Android**: Press the <kbd>R</kbd> key twice or select **"Reload"** from the **Dev Menu**, accessed via <kbd>Ctrl</kbd> + <kbd>M</kbd> (Windows/Linux) or <kbd>Cmd ⌘</kbd> + <kbd>M</kbd> (macOS).
- **iOS**: Press <kbd>R</kbd> in iOS Simulator.

## Congratulations! :tada:

You've successfully run and tested the LLM functionality! :partying_face:

### Now what?

- If you want to add this new React Native code to an existing application, check out the [Integration guide](https://reactnative.dev/docs/integration-with-existing-apps).
- If you're curious to learn more about React Native, check out the [docs](https://reactnative.dev/docs/getting-started).
- Learn more about on-device AI solutions at [Novastera](https://novastera.com).

## About

Part of [Novastera's](https://novastera.com) suite of privacy-focused solutions, this package enables on-device LLM inference with no data leaving the user's device. We're committed to helping developers build AI-powered applications that respect user privacy.

# Troubleshooting

If you're having issues getting the above steps to work, see the [Troubleshooting](https://reactnative.dev/docs/troubleshooting) page.

## Common Issues

### Model Loading Errors
- **iOS**: Ensure models are added to Xcode project's "Copy Bundle Resources"
- **Android**: Check that models are ≤1GB or manually copy to cache directory
- **Both**: Verify model files are in the correct GGUF format

### Build Failures
- **Android**: Large model files (>1GB) can cause build failures. Use smaller models or exclude them from assets
- **iOS**: Ensure sufficient device storage and memory for large models

# Learn More

To learn more about React Native and on-device AI, take a look at the following resources:

- [React Native Website](https://reactnative.dev) - learn more about React Native.
- [Getting Started](https://reactnative.dev/docs/environment-setup) - an **overview** of React Native and how setup your environment.
- [Learn the Basics](https://reactnative.dev/docs/getting-started) - a **guided tour** of the React Native **basics**.
- [Blog](https://reactnative.dev/blog) - read the latest official React Native **Blog** posts.
- [`@facebook/react-native`](https://github.com/facebook/react-native) - the Open Source; GitHub **repository** for React Native.
