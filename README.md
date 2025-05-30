# LlamaRN

> ⚠️ **WORK IN PROGRESS**: This package is currently under active development. Community help and feedback are greatly appreciated, especially in the areas mentioned in What Needs Help.

## Goals

* Provide a thin, reliable wrapper around llama.cpp for React Native
* Maintain compatibility with llama.cpp server API where possible
* Make it easy to run LLMs on mobile devices with automatic resource management
* Keep the codebase simple and maintainable

## Current Features

* Basic model loading and inference
* Metal support on iOS
* OpenCL/Vulkan support on Android (in progress)
* Automatic CPU/GPU detection
* Chat completion with templates (including Jinja template support)
* Embeddings generation
* Function/tool calling support

## What Needs Help

We welcome contributions, especially in these areas:

1. **Android GPU Testing and Detection**:
   * Development of reliable GPU detection mechanism in React Native
   * Implementation of proper backend initialization verification
   * Creation of robust testing framework for GPU availability
   * Integration of OpenCL and Vulkan acceleration once detection is stable
   * Performance benchmarking and optimization for mobile GPUs

2. **CI Improvements**:
   * Adding automated Android GPU tests to CI pipeline
   * Implementing device-specific testing strategies
   * Adding performance benchmarks to CI

3. **Tool Support**:
   * Improving tool calling functionality for complex interactions
   * Better JSON validation and error handling

4. **Testing**:
   * Automated testing using the example project
   * More comprehensive unit tests
   * Cross-device compatibility tests

5. **Documentation**:
   * Improving examples and usage guides
   * More detailed performance considerations

6. **Performance**:
   * Optimizing resource usage on different devices
   * Memory management improvements
   * Startup time optimization

If you're interested in helping with any of these areas, please check our Contributing Guide.

## Installation

```sh
npm install @novastera-oss/llamarn
```

## Developer Setup

If you're contributing to the library or running the example project, follow these setup steps:

### Prerequisites

1. Clone the repository and navigate to the project directory
2. Ensure you have React Native development environment set up for your target platform(s)

### Initial Setup

```sh
# Install dependencies
npm install

# Optional if you already had previous version of llamacpp
npm run clean-llama

# Initialize llama.cpp submodule and dependencies
npm run setup-llama-cpp
```

### Android Development

1. Build the native Android libraries:
```sh
# Build the external native libraries for Android
./scripts/build_android_external.sh
```

2. Run the example project:
```sh
cd example
npm run android
```

### iOS Development

1. Navigate to the example project and install iOS dependencies:
```sh
cd example
cd ios

# Install CocoaPods dependencies
bundle exec pod install

# Or if not using Bundler:
# pod install

cd ..
```

2. Run the example project:
```sh
npm run ios
```

### Development Notes

- **Android**: The `build_android_external.sh` script compiles llama.cpp for Android architectures and sets up the necessary native libraries. This step is required before running the Android example.

- **iOS**: The iOS setup uses CocoaPods to manage native dependencies. The prebuilt llama.cpp framework is included in the repository.

- **Troubleshooting**: If you encounter build issues, try cleaning your build cache:
  - Android: `cd android && ./gradlew clean`
  - iOS: `cd example/ios && rm -rf build && rm Podfile.lock && pod install`

## Basic Usage

### Simple Completion

```js
import { initLlama } from '@novastera-oss/llamarn';

// Initialize the model
const context = await initLlama({
  model: 'path/to/model.gguf',
  n_ctx: 2048,
  n_batch: 512
});

// Generate a completion
const result = await context.completion({
  prompt: 'What is artificial intelligence?',
  temperature: 0.7,
  top_p: 0.95
});

console.log('Response:', result.text);
```

### Chat Completion

```js
import { initLlama } from '@novastera-oss/llamarn';

// Initialize the model
const context = await initLlama({
  model: 'path/to/model.gguf',
  n_ctx: 4096,
  n_batch: 512,
  use_jinja: true  // Enable Jinja template parsing
});

// Chat completion with messages
const result = await context.completion({
  messages: [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Tell me about quantum computing.' }
  ],
  temperature: 0.7,
  top_p: 0.95
});

console.log('Response:', result.text);
// For OpenAI-compatible format: result.choices[0].message.content
```

### Chat with Tool Calling

```js
import { initLlama } from '@novastera-oss/llamarn';

// Initialize the model with appropriate parameters for tool use
const context = await initLlama({
  model: 'path/to/model.gguf',
  n_ctx: 2048,
  n_batch: 512,
  use_jinja: true  // Enable template handling for tool calls
});

// Create a chat with tool calling
const response = await context.completion({
  messages: [
    { role: 'system', content: 'You are a helpful assistant that can access weather data.' },
    { role: 'user', content: 'What\'s the weather like in Paris?' }
  ],
  tools: [
    {
      type: 'function',
      function: {
        name: 'get_weather',
        description: 'Get the current weather in a location',
        parameters: {
          type: 'object',
          properties: {
            location: {
              type: 'string',
              description: 'The city and state, e.g. San Francisco, CA'
            },
            unit: {
              type: 'string',
              enum: ['celsius', 'fahrenheit'],
              description: 'The unit of temperature to use'
            }
          },
          required: ['location']
        }
      }
    }
  ],
  tool_choice: 'auto',
  temperature: 0.7
});

// Check if the model wants to call a tool
if (response.choices?.[0]?.finish_reason === 'tool_calls' || response.tool_calls?.length > 0) {
  const toolCalls = response.choices?.[0]?.message?.tool_calls || response.tool_calls;
  
  // Process each tool call
  if (toolCalls && toolCalls.length > 0) {
    console.log('Function call:', toolCalls[0].function.name);
    console.log('Arguments:', toolCalls[0].function.arguments);
    
    // Here you would handle the tool call and then pass the result back in a follow-up completion
  }
}
```

### Generating Embeddings

```js
import { initLlama } from '@novastera-oss/llamarn';

// Initialize the model in embedding mode
const context = await initLlama({
  model: 'path/to/embedding-model.gguf',
  embedding: true,
  n_ctx: 2048
});

// Generate embeddings
const embeddingResponse = await context.embedding({
  input: "This is a sample text to embed"
});

console.log('Embedding:', embeddingResponse.data[0].embedding);
```

## Model Path Handling

The module accepts different path formats depending on the platform:

### iOS

* Bundle path: `models/model.gguf` (if added to Xcode project)
* Absolute path: `/path/to/model.gguf`

### Android

* Asset path: `asset:/models/model.gguf`
* File path: `file:///path/to/model.gguf`

## Documentation

* [Interface Documentation](INTERFACE.md) - Detailed API interfaces
* [Example App](example/) - Working example with common use cases
* [Contributing Guide](CONTRIBUTING.md) - How to help improve the library

## About

This library is currently being used in [Novastera's](https://novastera.com) mobile application, demonstrating its capabilities in production environments. We're committed to enabling on-device LLM inference with no data leaving the user's device, helping developers build AI-powered applications that respect user privacy.

## License

Apache 2.0

## Acknowledgments

We acknowledge the following projects and communities that have contributed to the development of this library:

* **[mybigday/llama.rn](https://github.com/mybigday/llama.rn)** - A foundational React Native binding for llama.cpp that demonstrated the viability of on-device LLM inference in mobile applications.

* **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** - The core C++ library that enables efficient LLM inference, serving as the foundation for this project.

* The test implementation of the Android Turbo Module ([react-native-pure-cpp-turbo-module-library](https://github.com/Zach-Dean-Attractions-io/react-native-pure-cpp-turbo-module-library)) provided valuable insights for our C++ integration.

These projects have significantly contributed to the open-source ecosystem, and we are committed to building upon their work while maintaining the same spirit of collaboration and innovation.
