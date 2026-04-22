import type { TurboModule } from 'react-native';
import { TurboModuleRegistry } from 'react-native';

/**
 * Native LlamaCppRn Module
 *
 * The API is designed to be compatible with the llama.cpp server API:
 * https://github.com/ggml-org/llama.cpp/tree/master/examples/server
 *
 * However, since we are in the context of a mobile app, we need to make some adjustments.
 */

export interface LlamaContextType {
  // This will be a native object reference that maps to
  // a pointer to the llama_context C++ objec
}

export type ModelCapability =
  | 'vision-chat'
  | 'image-encode'
  | 'audio-transcribe'
  | 'vision-reasoning';

export type LlamaMessageContentPart =
  | { type: 'text'; text: string }
  | { type: 'image_url'; image_url: { url: string } }
  | { type: 'audio_url'; audio_url: { url: string } };

export type LlamaMessageContent = string | null | LlamaMessageContentPart[];

export interface LlamaModelParams {
  // Model loading parameters
  model: string;               // path to the model file
  n_ctx?: number;             // context size (default: 2048)
  n_batch?: number;           // batch size (default: 512)
  n_ubatch?: number;          // micro batch size for prompt processing
  n_threads?: number;         // number of threads (default: number of physical CPU cores)
  n_keep?: number;            // number of tokens to keep from initial promp

  // GPU acceleration parameters
  n_gpu_layers?: number;      // number of layers to store in VRAM (default: 0)

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
  reasoning_budget?: number;  // Controls thinking functionality: -1 = unlimited, 0 = disabled, >0 = limited
  reasoning_format?: string;  // Reasoning format: 'none', 'auto', 'deepseek', 'deepseek-legacy'
  thinking_forced_open?: boolean; // Force reasoning models to always output thinking
  parse_tool_calls?: boolean; // Enable tool call parsing (auto-enabled when use_jinja is true)
  parallel_tool_calls?: boolean; // Enable parallel/multiple tool calls for supported models

  // LoRA adapters
  lora_adapters?: Array<{
    path: string;             // path to LoRA adapter file
    scale?: number;           // scaling factor for the adapter (default: 1.0)
  }>;

  // Grammar-based sampling
  grammar?: string;           // GBNF grammar for grammar-based sampling

  // Multimodal
  mmproj?: string;               // path to multimodal projection model (.gguf)
  image_marker?: string;         // custom placeholder token (default: <__media__>)
  capabilities?: ModelCapability[]; // declare which modalities are active

  // Cooperative prompt-ingestion loop (values from loadLlamaModelInfo.suggestedChunkSize / isCpuOnly)
  chunk_size?: number;   // tokens per decode call during prompt ingestion (default 128)
  is_cpu_only?: boolean; // true = 2ms sleep/chunk
  prompt_chunk_gap_ms?: number; // minimum inter-chunk gap on GPU path (default: 5ms)
}

export interface LlamaCompletionParams {
  // Basic completion parameters
  prompt?: string;            // text promp
  system_prompt?: string;     // system prompt for chat mode (alternative to including it in messages)
  messages?: LlamaMessage[];  // chat messages
  temperature?: number;        // sampling temperature (default: 0.8)
  top_p?: number;              // top-p sampling (native default: 0.9)
  top_k?: number;              // top-k sampling (default: 40)
  min_p?: number;              // min-p sampling floor (native default: 0.05)
  n_predict?: number;          // max tokens to predict (default: -1, infinite)
  max_tokens?: number;         // alias for n_predic
  n_keep?: number;             // parsed by native completion options (currently reserved/no-op in generation path)
  stop?: string[];             // stop sequences
  stream?: boolean;            // advisory; callback presence controls streaming behavior
  ignore_eos?: boolean;        // ignore EOS/EOG termination checks
  reset_kv_cache?: boolean;    // force KV cache reset for this request
  // Chat parameters
  chat_template?: string;      // optional chat template name to use

  // Tool calling parameters
  tool_choice?: string | 'auto' | 'none'; // Tool choice mode
  tools?: LlamaTool[];         // Available tools

  // Advanced parameters (matching llama.cpp server)
  repeat_penalty?: number;     // repetition penalty (native default: 1.0)
  repeat_last_n?: number;      // last n tokens to consider for repetition penalty (default: 64)
  frequency_penalty?: number;   // frequency penalty (default: 0.0)
  presence_penalty?: number;    // presence penalty (default: 0.0)
  seed?: number;                // RNG seed (default: -1, random)
  grammar?: string;             // GBNF grammar for structured outpu
  token_rate_cap?: number;      // max tokens/sec during generation (0 = uncapped, default: 30)
  token_buffer_size?: number;   // stream callback flush cadence in tokens (default: 4)
  prompt_id?: string;           // cache key for system prompt/tools identity
  config_id?: string;           // cache key for effective completion config (include tools + main system prompt identity)
}

export interface LlamaMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: LlamaMessageContent;
  tool_call_id?: string;
  name?: string;
}

export interface JsonSchemaObject {
  type: 'object';
  properties: {
    [key: string]: JsonSchemaProperty;
  };
  required?: string[];
  description?: string;
}

export interface JsonSchemaArray {
  type: 'array';
  items: JsonSchemaProperty;
  description?: string;
}

export type JsonSchemaScalar = {
  type: 'string' | 'number' | 'boolean' | 'null';
  enum?: string[];
  description?: string;
};

export type JsonSchemaProperty = JsonSchemaObject | JsonSchemaArray | JsonSchemaScalar;

export interface LlamaTool {
  type: 'function';
  function: {
    name: string;
    description?: string;
    parameters: JsonSchemaObject;
  };
}

export interface LlamaCompletionResult {
  text: string;                          // The generated completion tex
  tokens_predicted: number;              // Number of tokens generated
  timings: {
    predicted_n: number;                 // Number of tokens predicted
    predicted_ms: number;                // Time spent generating tokens (ms)
    prompt_n: number;                    // Number of tokens in the promp
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
    finish_reason: 'stop' | 'length' | 'tool_calls' | 'tool_call_parse_error';
    tool_call_parse_error?: string;
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

// Add new interfaces for embedding
export interface EmbeddingOptions {
  input?: string | string[];      // Text input to embed (OpenAI format)
  content?: string | string[];    // Alternative text input (custom format)
  add_bos_token?: boolean;        // Whether to add a beginning of sequence token (default: true)
  encoding_format?: 'float' | 'base64'; // Output encoding forma
  model?: string;                 // Model identifier (ignored, included for OpenAI compatibility)
}

export interface EmbeddingResponse {
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

export interface ImageEmbedResult {
  embedding: number[];  // flat float32 array, length = n_tokens * n_embd
  n_tokens: number;     // number of vision tokens
  n_embd: number;       // embedding dimension
}

export interface TranscriptSegment {
  start_s: number;
  end_s: number;
  text: string;
}

export interface TranscriptResult {
  text: string;
  segments?: TranscriptSegment[];
}

export interface DetectedObject {
  label: string;
  confidence: number;
  bbox: { x: number; y: number; w: number; h: number };
}

export interface DetectionResult {
  objects: DetectedObject[];
}

export interface LlamaContextMethods {
  completion(params: LlamaCompletionParams, partialCallback?: (data: {token: string}) => void): Promise<LlamaCompletionResult>;
  completionSync(params: LlamaCompletionParams, partialCallback?: (data: {token: string}) => void): LlamaCompletionResult;

  // Updated tokenize method to match server.cpp interface
  tokenize(options: {
    content: string;
    add_special?: boolean;
    with_pieces?: boolean;
  }): Promise<{
    tokens: (number | {id: number, piece: string | number[]})[]
  }>;

  // New detokenize method
  detokenize(options: {
    tokens: number[]
  }): Promise<{
    content: string
  }>;

  /**
   * Generate embeddings for input tex
   *
   * @param options Embedding options matching server.cpp forma
   * @returns Array of embedding values or OpenAI-compatible embedding response
   */
  embedding(options: EmbeddingOptions): Promise<EmbeddingResponse>;
  detectTemplate(messages: LlamaMessage[]): Promise<string>;
  loadSession(path: string): Promise<boolean>;
  saveSession(path: string): Promise<boolean>;
  stopCompletion(): Promise<void>;
  release(): Promise<void>;

  isMultimodalEnabled(): Promise<boolean>;
  getSupportedModalities(): Promise<{
    vision: boolean;
    audio: boolean;
    audioSampleRate?: number;
  }>;

  embedImage(imagePath: string, options?: { normalize?: boolean }): Promise<ImageEmbedResult>;

  transcribeAudio(audioPath: string, options?: { language?: string }): Promise<TranscriptResult>;

  visionReasoning(imagePath: string, options?: { prompt?: string }): Promise<{ raw_text: string }>;

  runOnFrame(
    buffer: Object,
    width: number,
    height: number,
    capability: ModelCapability,
    options?: { maxSize?: number },
  ): Promise<ImageEmbedResult | TranscriptResult | DetectionResult | LlamaCompletionResult | null>;
}

export interface Spec extends TurboModule {
  // Initialize a Llama context with the given model parameters
  initLlama(params: LlamaModelParams): Promise<LlamaContextType & LlamaContextMethods>;

  // Load model info without creating a full context
  loadLlamaModelInfo(modelPath: string, mmprojPath?: string): Promise<{
    n_params: number;
    n_vocab: number;
    n_context: number;
    n_embd: number;
    n_layers: number;
    description: string;
    gpuSupported: boolean;
    optimalGpuLayers: number;
    quant_type?: string;
    architecture: string;
    model_size_bytes: number;
    availableMemoryMB: number;
    estimatedVramMB: number;
    mmprojSizeMB?: number;      // present when mmprojPath was supplied
    suggestedChunkSize: number; // recommended chunk_size for initLlama (32=CPU, 128=GPU)
    isCpuOnly: boolean;         // true when optimalGpuLayers == 0
  }>;
}

const LlamaCppRn = TurboModuleRegistry.getEnforcing<Spec>('RNLlamaCpp');
/**
 * LlamaModel type representing a loaded model instance
 */
export type LlamaModel = LlamaContextType & LlamaContextMethods;

/**
 * Initialize a Llama context with the given model parameters
 */
export function initLlama(params: LlamaModelParams): Promise<LlamaContextType & LlamaContextMethods> {
  return LlamaCppRn.initLlama(params);
}

/**
 * Get information about a model without loading it fully.
 * Pass mmprojPath to get an optimalGpuLayers that accounts for mmproj VRAM reservation.
 * The returned suggestedChunkSize and isCpuOnly can be passed directly to initLlama.
 */
export function loadLlamaModelInfo(
  modelPath: string,
  mmprojPath?: string
): Promise<{
  n_params: number;
  n_vocab: number;
  n_context: number;
  n_embd: number;
  description: string;
  gpuSupported: boolean;
  optimalGpuLayers?: number;
  quant_type?: string;
  architecture?: string;
  availableMemoryMB?: number;
  estimatedVramMB?: number;
  mmprojSizeMB?: number;
  suggestedChunkSize: number;
  isCpuOnly: boolean;
}> {
  return LlamaCppRn.loadLlamaModelInfo(modelPath, mmprojPath);
}

export default LlamaCppRn;
