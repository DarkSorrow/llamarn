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
* OpenCL/Vulkan GPU acceleration on Android
* Snapdragon Hexagon NPU support on Android (arm64-v8a)
* Automatic CPU/GPU detection
* Chat completion with templates (including Jinja template support)
* Embeddings generation
* Function/tool calling support
* **Advanced thinking and reasoning support** for compatible models
* **Flexible reasoning budget control** (unlimited, disabled, or limited)
* **Multiple reasoning format support** (none, auto, deepseek, deepseek-legacy)

## What Needs Help

We welcome contributions, especially in these areas:

1. **Android GPU and NPU Testing**:
   * **OpenCL/Vulkan GPU Libraries**: GPU acceleration libraries (OpenCL and Vulkan) have been built and integrated, but we need help testing them on various Android devices to ensure proper functionality and performance.
   * **Snapdragon Hexagon NPU Support**: Hexagon NPU support has been added for Snapdragon devices (arm64-v8a), but we need community testing on actual Snapdragon devices to verify it works correctly.
   * Development of reliable GPU/NPU detection mechanism in React Native
   * Implementation of proper backend initialization verification
   * Creation of robust testing framework for GPU/NPU availability
   * Performance benchmarking and optimization for mobile GPUs and NPUs
   * Real-world device testing across different manufacturers and chipset generations

2. **CI Improvements**:
   * Adding automated Android GPU/NPU tests to CI pipeline
   * Implementing device-specific testing strategies
   * Adding performance benchmarks to CI

3. **Tool Support**:
   * Improving tool calling functionality for complex interactions
   * Better JSON validation and error handling
   * Enhanced thinking and reasoning model support
   * Advanced reasoning format implementations

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
  n_batch: 512,
  // Optional: Enable thinking and reasoning capabilities
  reasoning_budget: -1,  // Unlimited thinking
  reasoning_format: 'auto'  // Automatic reasoning format detection
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
  use_jinja: true,  // Enable Jinja template parsing
  // Optional: Configure thinking and reasoning
  reasoning_budget: -1,  // Enable unlimited thinking
  reasoning_format: 'deepseek'  // Use DeepSeek reasoning format
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
  use_jinja: true,  // Enable template handling for tool calls
  parse_tool_calls: true,  // Enable tool call parsing (auto-enabled with use_jinja)
  parallel_tool_calls: false  // Disable parallel tool calls for compatibility
});
```

### Thinking and Reasoning Models

For models that support reasoning and thinking, you can enable advanced thinking functionality:

```js
import { initLlama } from '@novastera-oss/llamarn';

// Initialize a reasoning model with thinking capabilities
const context = await initLlama({
  model: 'path/to/reasoning-model.gguf',
  n_ctx: 4096,
  n_batch: 512,
  use_jinja: true,
  
  // Thinking and reasoning options
  reasoning_budget: -1,           // -1 = unlimited thinking, 0 = disabled, >0 = limited
  reasoning_format: 'deepseek',   // Use DeepSeek reasoning format
  thinking_forced_open: true,     // Force the model to always output thinking
  parse_tool_calls: true,         // Enable tool call parsing
  parallel_tool_calls: false      // Disable parallel tool calls for compatibility
});

// Chat completion with thinking enabled
const result = await context.completion({
  messages: [
    { role: 'system', content: 'You are a helpful assistant. Think through problems step by step.' },
    { role: 'user', content: 'Solve this math problem: What is 15% of 240?' }
  ],
  temperature: 0.7
});

console.log('Response:', result.text);
// The response may include thinking tags like <think>...</think> depending on the model
```

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

## Advanced Configuration Options

### Thinking and Reasoning Parameters

The library supports advanced thinking and reasoning capabilities for models that support them:

- **`reasoning_budget`**: Controls the amount of thinking allowed
  - `-1`: Unlimited thinking (default)
  - `0`: Disabled thinking
  - `>0`: Limited thinking with the specified budget

- **`reasoning_format`**: Controls how thinking is parsed and returned
  - `'none'`: Leave thoughts unparsed in message content
  - `'auto'`: Same as deepseek (default)
  - `'deepseek'`: Extract thinking into `message.reasoning_content`
  - `'deepseek-legacy'`: Extract thinking with streaming behavior

- **`thinking_forced_open`**: Forces reasoning models to always output thinking
  - `false`: Normal thinking behavior (default)
  - `true`: Always include thinking tags in output

- **`parse_tool_calls`**: Enables tool call parsing
  - `true`: Parse and extract tool calls (default)
  - `false`: Disable tool call parsing
  - **Note**: Automatically enabled when `use_jinja` is true

- **`parallel_tool_calls`**: Enables multiple tool calls in a single response
  - `false`: Single tool calls only (default, for compatibility)
  - `true`: Allow parallel tool calls (only supported by some models)

### Automatic Tool Call Enhancement

When `use_jinja` is enabled, `parse_tool_calls` is automatically enabled because Jinja templates provide better tool calling capabilities. This ensures optimal tool support when using advanced templates.

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

## About Novastera

**LlamaRN** is part of the **Novastera** open-source ecosystem, a modern CRM/ERP platform designed for the next generation of business applications. Novastera combines cutting-edge AI capabilities with comprehensive business management tools, enabling organizations to leverage on-device AI for enhanced productivity and data privacy.

### Key Features of Novastera Platform

- **Modern CRM/ERP System**: Comprehensive business management with AI-powered insights
- **On-Device AI**: Privacy-first approach with local LLM inference - no data leaves your device
- **Mobile-First**: Native iOS and Android applications built with React Native
- **Open Source**: Part of Novastera's commitment to open-source innovation

This library is currently being used in [Novastera's](https://novastera.com) mobile application, demonstrating its capabilities in production environments. We're committed to enabling on-device LLM inference with no data leaving the user's device, helping developers build AI-powered applications that respect user privacy.

Learn more about Novastera: [https://novastera.com/resources](https://novastera.com/resources)

## License

Apache 2.0

## Acknowledgments

We acknowledge the following projects and communities that have contributed to the development of this library:

* **[mybigday/llama.rn](https://github.com/mybigday/llama.rn)** - A foundational React Native binding for llama.cpp that demonstrated the viability of on-device LLM inference in mobile applications.

* **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** - The core C++ library that enables efficient LLM inference, serving as the foundation for this project.

* The test implementation of the Android Turbo Module ([react-native-pure-cpp-turbo-module-library](https://github.com/Zach-Dean-Attractions-io/react-native-pure-cpp-turbo-module-library)) provided valuable insights for our C++ integration.

These projects have significantly contributed to the open-source ecosystem, and we are committed to building upon their work while maintaining the same spirit of collaboration and innovation.
