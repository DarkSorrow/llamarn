# Interface Documentation

This document describes the TypeScript interfaces and API for `@novastera-oss/llamarn`.

## Core Functions

### `initLlama(params: LlamaModelParams): Promise<LlamaModel>`

Initialize a Llama context with the given model parameters.

**Parameters:**
- `params`: Configuration object of type `LlamaModelParams`

**Returns:** Promise that resolves to a `LlamaModel` instance

### `loadLlamaModelInfo(modelPath: string, mmprojPath?: string): Promise<ModelInfo>`

Get information about a model without loading it fully. The returned values (`optimalGpuLayers`, `suggestedChunkSize`, `isCpuOnly`) are designed to be passed directly to `initLlama`.

**Parameters:**
- `modelPath`: Path to the model file
- `mmprojPath` *(optional)*: Path to a multimodal projection `.gguf`. When provided, its file size is read via `stat()` (no load) and deducted from the VRAM budget before computing `optimalGpuLayers`, preventing VRAM oversubscription when both models run concurrently.

**Returns:** Promise that resolves to:

```typescript
{
  // Model metadata
  description: string;         // human-readable name + quant (e.g. "llama 7B Q4_K_M")
  n_params: number;            // parameter count
  n_vocab: number;             // vocabulary size
  n_context: number;           // model's maximum training context length
  n_embd: number;              // embedding dimension
  n_layers: number;            // total layer count
  model_size_bytes: number;    // quantized on-disk size in bytes
  architecture: string;        // e.g. "llama", "qwen2", "mistral"
  quant_type: string;          // e.g. "Q4_K", "Q8_0"

  // Device capability
  gpuSupported: boolean;       // true if GPU offload is available on this device
  availableMemoryMB: number;   // current free device RAM in MB

  // Recommended initLlama values — pass these directly
  optimalGpuLayers: number;    // safe GPU layer count (15% RAM budget, 75% layer cap, minus mmproj)
  estimatedVramMB: number;     // VRAM needed for optimalGpuLayers layers
  suggestedChunkSize: number;  // 32 (CPU-only) or 128 (GPU) — pass as chunk_size to initLlama
  isCpuOnly: boolean;          // true when optimalGpuLayers == 0 — pass as is_cpu_only to initLlama

  // Present only when mmprojPath was supplied
  mmprojSizeMB?: number;       // file size of the projection model in MB
}
```

## Type Definitions

### `LlamaModelParams`

Configuration parameters for initializing a Llama model.

```typescript
interface LlamaModelParams {
  // Model loading parameters
  model: string;               // path to the model file
  n_ctx?: number;             // context size (default: 2048)
  n_batch?: number;           // batch allocation size for llama_decode (default: 512)
  n_ubatch?: number;          // micro batch size for prompt processing
  n_threads?: number;         // number of threads (default: number of physical CPU cores)
  n_keep?: number;            // number of tokens to keep from initial prompt

  // GPU acceleration parameters
  n_gpu_layers?: number;      // number of layers to store in VRAM (default: 0)
                              // use optimalGpuLayers from loadLlamaModelInfo

  // Memory management parameters
  use_mmap?: boolean;         // use mmap for faster loading (default: true)
  use_mlock?: boolean;        // use mlock to keep model in memory (default: false)

  // Model behavior parameters
  vocab_only?: boolean;       // only load the vocabulary, no weights
  embedding?: boolean;        // use embedding mode (default: false)
  seed?: number;              // RNG seed for reproducibility

  // RoPE parameters
  rope_freq_base?: number;    // RoPE base frequency (default: 10000.0)
  rope_freq_scale?: number;   // RoPE frequency scaling factor (default: 1.0)

  // YaRN parameters (RoPE scaling for longer contexts)
  yarn_ext_factor?: number;   // YaRN extrapolation mix factor
  yarn_attn_factor?: number;  // YaRN magnitude scaling factor
  yarn_beta_fast?: number;    // YaRN low correction dim
  yarn_beta_slow?: number;    // YaRN high correction dim

  // Additional options
  logits_all?: boolean;       // return logits for all tokens
  chat_template?: string;     // override chat template
  use_jinja?: boolean;        // use Jinja template parser
  verbose?: number;           // verbosity level (0 = silent, 1 = info, 2+ = debug)

  // Thinking and reasoning options
  reasoning_budget?: number;      // -1 = unlimited, 0 = disabled, >0 = token limit
  reasoning_format?: string;      // 'none' | 'auto' | 'deepseek' | 'deepseek-legacy'
  thinking_forced_open?: boolean; // force thinking tags in every response
  parse_tool_calls?: boolean;     // parse tool call JSON (auto-enabled when use_jinja is true)
  parallel_tool_calls?: boolean;  // allow multiple tool calls per response

  // LoRA adapters
  lora_adapters?: Array<{
    path: string;             // path to LoRA adapter file
    scale?: number;           // scaling factor for the adapter (default: 1.0)
  }>;

  // Grammar-based sampling
  grammar?: string;           // GBNF grammar for grammar-based sampling

  // Multimodal
  mmproj?: string;            // path to multimodal projection model (.gguf)
  capabilities?: Array<'vision-chat' | 'image-encode' | 'audio-transcribe' | 'vision-reasoning'>;

  // Cooperative prompt-ingestion loop
  // Use the values from loadLlamaModelInfo.suggestedChunkSize / isCpuOnly directly.
  chunk_size?: number;        // tokens per llama_decode call during prompt encoding (default: 128)
                              // independent of n_batch; clamped to [8, 512]
  is_cpu_only?: boolean;      // true  → sleep 2 ms after each chunk (CPU-only devices)
                              // false → yield() + sleep 1 ms only if a chunk exceeded 40 ms
}
```

### `LlamaModel`

Represents a loaded model instance with available methods.

```typescript
type LlamaModel = LlamaContextType & LlamaContextMethods;
```

### `LlamaContextMethods`

Available methods on a loaded model instance.

```typescript
interface LlamaContextMethods {
  completion(params: LlamaCompletionParams, partialCallback?: (data: {token: string}) => void): Promise<LlamaCompletionResult>;
  tokenize(options: TokenizeOptions): Promise<TokenizeResult>;
  detokenize(options: DetokenizeOptions): Promise<DetokenizeResult>;
  embedding(options: EmbeddingOptions): Promise<EmbeddingResponse>;
  detectTemplate(messages: LlamaMessage[]): Promise<string>;
  loadSession(path: string): Promise<boolean>;
  saveSession(path: string): Promise<boolean>;
  stopCompletion(): Promise<void>;
  release(): Promise<void>;
}
```

### `LlamaCompletionParams`

Parameters for text/chat completion.

```typescript
interface LlamaCompletionParams {
  // Basic completion parameters
  prompt?: string;            // text prompt
  system_prompt?: string;     // system prompt for chat mode
  messages?: LlamaMessage[];  // chat messages
  temperature?: number;       // sampling temperature (default: 0.8)
  top_p?: number;             // top-p sampling (default: 0.95)
  top_k?: number;             // top-k sampling (default: 40)
  n_predict?: number;         // max tokens to predict (default: -1, infinite)
  max_tokens?: number;        // alias for n_predict
  stop?: string[];            // stop sequences
  stream?: boolean;           // stream tokens as they're generated (default: true)
  
  // Chat parameters
  chat_template?: string;     // optional chat template name to use

  // Tool calling parameters
  tool_choice?: string | 'auto' | 'none'; // Tool choice mode
  tools?: LlamaTool[];        // Available tools

  // Advanced parameters
  repeat_penalty?: number;    // repetition penalty (default: 1.1)
  repeat_last_n?: number;     // last n tokens to consider for repetition penalty (default: 64)
  frequency_penalty?: number; // frequency penalty (default: 0.0)
  presence_penalty?: number;  // presence penalty (default: 0.0)
  seed?: number;              // RNG seed (default: -1, random)
  grammar?: string;           // GBNF grammar for structured output
}
```

### `LlamaMessage`

Chat message format.

```typescript
interface LlamaMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  tool_call_id?: string;
  name?: string;
}
```

### `LlamaCompletionResult`

Result from completion operations.

```typescript
interface LlamaCompletionResult {
  text: string;                          // The generated completion text
  tokens_predicted: number;              // Number of tokens generated
  timings: {
    predicted_n: number;                 // Number of tokens predicted
    predicted_ms: number;                // Time spent generating tokens (ms)
    prompt_n: number;                    // Number of tokens in the prompt
    prompt_ms: number;                   // Time spent processing prompt (ms)
    total_ms: number;                    // Total time spent (ms)
  };

  // OpenAI-compatible response fields
  choices?: Array<{
    index: number;
    message: {
      role: string;
      content: string;
      tool_calls?: Array<{
        id: string;
        type: string;
        function: {
          name: string;
          arguments: string;
        }
      }>
    };
    finish_reason: 'stop' | 'length' | 'tool_calls';
  }>;

  // Tool calls may appear at different levels based on model response
  tool_calls?: Array<{
    id: string;                          // Unique identifier for the tool call
    type: string;                        // Type of tool call (e.g. 'function')
    function: {
      name: string;                      // Name of the function to call
      arguments: string;                 // JSON string of arguments for the function
    };
  }>;
}
```

### `LlamaTool`

Tool definition for function calling.

```typescript
interface LlamaTool {
  type: 'function';
  function: {
    name: string;
    description?: string;
    parameters: JsonSchemaObject;
  };
}
```

### `EmbeddingOptions`

Options for generating embeddings.

```typescript
interface EmbeddingOptions {
  input?: string | string[];      // Text input to embed (OpenAI format)
  content?: string | string[];    // Alternative text input (custom format)
  add_bos_token?: boolean;        // Whether to add a beginning of sequence token (default: true)
  encoding_format?: 'float' | 'base64'; // Output encoding format
  model?: string;                 // Model identifier (ignored, included for OpenAI compatibility)
}
```

### `EmbeddingResponse`

Response from embedding operations.

```typescript
interface EmbeddingResponse {
  data: Array<{
    embedding: number[] | string; // Can be array of numbers or base64 string
    index: number;
    object: 'embedding';
    encoding_format?: 'base64';   // Present only when base64 encoding is used
  }>;
  model: string;
  object: 'list';
  usage: {
    prompt_tokens: number;
    total_tokens: number;
  };
}
```

## Usage Examples

### Basic Model Initialization

Always call `loadLlamaModelInfo` first so the device-safe values flow into `initLlama`:

```typescript
import { loadLlamaModelInfo, initLlama } from '@novastera-oss/llamarn';

const info = await loadLlamaModelInfo('path/to/model.gguf');

const context = await initLlama({
  model:        'path/to/model.gguf',
  n_ctx:        4096,
  n_gpu_layers: info.optimalGpuLayers,   // device-safe GPU layer count
  chunk_size:   info.suggestedChunkSize,  // cooperative ingestion granularity
  is_cpu_only:  info.isCpuOnly,          // OS-yield mode for prompt encoding
  use_jinja:    true,
});
```

For vision models, pass `mmprojPath` so the VRAM budget is split correctly:

```typescript
const info = await loadLlamaModelInfo('path/to/model.gguf', 'path/to/mmproj.gguf');

const context = await initLlama({
  model:        'path/to/model.gguf',
  mmproj:       'path/to/mmproj.gguf',
  capabilities: ['vision-chat'],
  n_ctx:        4096,
  n_gpu_layers: info.optimalGpuLayers,  // already accounts for mmproj VRAM
  chunk_size:   info.suggestedChunkSize,
  is_cpu_only:  info.isCpuOnly,
  use_jinja:    true,
});
```

### Text Completion

```typescript
const result = await context.completion({
  prompt: 'What is artificial intelligence?',
  temperature: 0.7,
  n_predict: 100
});

console.log(result.text);
```

### Chat Completion

```typescript
const result = await context.completion({
  messages: [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Hello!' }
  ],
  temperature: 0.7,
  n_predict: 100
});

console.log(result.text);
```

### Streaming Completion

```typescript
const result = await context.completion({
  prompt: 'Tell me a story',
  temperature: 0.7,
  n_predict: 200
}, (data) => {
  // This callback is called for each token
  console.log('Token:', data.token);
});
```

### Embeddings

```typescript
// Initialize model in embedding mode
const embeddingContext = await initLlama({
  model: 'path/to/embedding-model.gguf',
  embedding: true
});

const response = await embeddingContext.embedding({
  input: "Text to embed"
});

console.log(response.data[0].embedding);
```

### Tool Calling

```typescript
const response = await context.completion({
  messages: [
    { role: 'user', content: 'What\'s the weather like?' }
  ],
  tools: [
    {
      type: 'function',
      function: {
        name: 'get_weather',
        description: 'Get current weather',
        parameters: {
          type: 'object',
          properties: {
            location: { type: 'string' }
          },
          required: ['location']
        }
      }
    }
  ],
  tool_choice: 'auto'
});

if (response.tool_calls) {
  console.log('Tool calls:', response.tool_calls);
}
```

## Platform-Specific Considerations

### iOS
- Models should be added to the Xcode project bundle
- Use bundle paths like `models/model.gguf`
- Metal GPU acceleration available on supported devices

### Android
- Models should be placed in `android/app/src/main/assets/`
- Use asset paths like `asset:/models/model.gguf`
- For large models, copy to cache directory first using `RNFS.copyFileAssets()`

## Error Handling

All async methods can throw errors. Wrap calls in try-catch blocks:

```typescript
try {
  const context = await initLlama({
    model: 'path/to/model.gguf'
  });
  
  const result = await context.completion({
    prompt: 'Hello'
  });
} catch (error) {
  console.error('Error:', error.message);
}
```

## Memory Management

Always call `release()` when done with a model to free memory:

```typescript
// When done with the model
await context.release();
``` 