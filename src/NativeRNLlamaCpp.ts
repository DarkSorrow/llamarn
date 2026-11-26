import type { TurboModule, EventSubscription, CodegenTypes } from 'react-native';
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

export interface ModelLoadProgressEvent {
  phase: 'model' | 'mmproj'; // which load phase this progress update belongs to
  progress: number;          // 0.0 - 1.0
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

export type LlamaReasoningFormat = 'none' | 'auto' | 'deepseek' | 'deepseek-legacy';

/** One OpenAI function reference, as used by tool_choice. */
export interface LlamaToolChoiceFunction {
  type: 'function';
  function: { name: string };
}

/**
 * OpenAI tool_choice. Parsed natively against `tools` by llama.cpp
 * (common_chat_tool_choice_parse_oaicompat). A name must match exactly one supplied tool,
 * otherwise the completion fails with an invalid-parameter error.
 * - 'auto' (default): the model decides.
 * - 'none': no tool call.
 * - 'required': the reply must be a tool call.
 * - { type: 'function', function: { name } }: required, limited to that tool.
 * - { type: 'allowed_tools', allowed_tools: { mode, tools } }: auto or required, limited to those tools.
 * Whether the grammar itself drops the other tools depends on the model's template handler
 * upstream; required mode is always enforced.
 */
export type LlamaToolChoice =
  | 'auto'
  | 'none'
  | 'required'
  | LlamaToolChoiceFunction
  | {
      type: 'allowed_tools';
      allowed_tools: { mode: 'auto' | 'required'; tools: LlamaToolChoiceFunction[] };
    };

/** Streaming payload. See docs/CONFIGURATION.md "Streaming". */
export interface LlamaStreamChunk {
  token: string;              // raw generated text since the previous chunk (always present)
  content?: string;           // chat calls: answer-text delta from the chat parser
  reasoning_content?: string; // chat calls: thinking-text delta from the chat parser
}

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
  // Omit BOTH to get llama.cpp's LLAMA_LOAD_MODE_AUTO (recommended): it memory-maps
  // the model except on backends that report no mmap support, where it falls back
  // automatically. Setting either one opts out of AUTO and selects an explicit mode:
  //   use_mmap only  -> MMAP  (true) / NONE (false)
  //   use_mlock only -> MLOCK (true, no mmap)
  //   both true      -> MMAP_MLOCK
  use_mmap?: boolean;         // force memory-mapped loading on/off
  use_mlock?: boolean;        // lock model pages in RAM (prevents swapping)

  // Model behavior parameters
  embedding?: boolean;        // use embedding mode (default: false)
  seed?: number;              // RNG seed for reproducibility

  // RoPE parameters. Omit them: the default is the value stored in the GGUF.
  // Setting one overrides the model's trained value (e.g. Qwen3 uses a base of 1e6).
  rope_freq_base?: number;    // RoPE base frequency (default: 0 = from the model)
  rope_freq_scale?: number;   // RoPE frequency scaling factor (default: 0 = from the model)

  // YaRN parameters (RoPE scaling for longer contexts). Default -1 = from the model.
  yarn_ext_factor?: number;   // YaRN extrapolation mix factor
  yarn_attn_factor?: number;  // YaRN magnitude scaling factor
  yarn_beta_fast?: number;    // YaRN low correction dim
  yarn_beta_slow?: number;    // YaRN high correction dim

  // Additional options
  chat_template?: string;     // override chat template
  use_jinja?: boolean;        // use Jinja template parser
  verbose?: number;           // verbosity level (0 = silent, 1 = info, 2+ = debug)

  // Thinking and reasoning options
  reasoning_budget?: number;  // Omitted or 0 turns thinking off. -1 is unlimited. >0 is the sampler cap. A completion cannot turn thinking on.
  // Default 'auto': thinking goes to message.reasoning_content. 'none' leaves it in content.
  reasoning_format?: LlamaReasoningFormat;
  parallel_tool_calls?: boolean; // more than one tool call per reply. Omitted = what the chat template supports (llama-server default)

  // LoRA adapters
  lora_adapters?: Array<{
    path: string;             // path to LoRA adapter file
    scale?: number;           // scaling factor for the adapter (default: 1.0)
  }>;

  // Multimodal
  mmproj?: string;               // path to multimodal projection model (.gguf)
  image_marker?: string;         // custom placeholder token (default: <__media__>)
  capabilities?: ModelCapability[]; // declare which modalities are active

  // Cooperative prompt-ingestion loop (values from loadLlamaModelInfo.suggestedChunkSize / isCpuOnly)
  // Kept because of earlier crashes on real Android devices. See docs/CONFIGURATION.md "Prompt pacing".
  chunk_size?: number;   // tokens per decode call during prompt ingestion (default 128, clamped 8..512 and to n_batch)
  is_cpu_only?: boolean; // true = fixed 2 ms sleep after every chunk (does not change n_gpu_layers)
  prompt_chunk_gap_ms?: number; // GPU path: per-chunk time floor from the chunk start, +1 ms per 2048 prompt tokens (default 5)
}

export interface LlamaCompletionParams {
  // Basic completion parameters
  prompt?: string;            // text promp
  messages?: LlamaMessage[];  // chat messages. Put the system text here, with an id.
  temperature?: number;        // sampling temperature (default: 0.8)
  top_p?: number;              // top-p sampling (native default: 0.95)
  top_k?: number;              // top-k sampling (default: 40)
  min_p?: number;              // min-p sampling floor (native default: 0.05)
  n_predict?: number;          // max tokens to predict (default: -1, infinite)
  max_tokens?: number;         // alias for n_predic
  n_keep?: number;             // context-shift keep. 0 = first message boundary when it fits, else half the context. >0 = token count. -1 = stop when the context is full
  stop?: string[];             // stop sequences
  stream?: boolean;            // advisory; callback presence controls streaming behavior
  ignore_eos?: boolean;        // ignore EOS/EOG termination checks
  reset_kv_cache?: boolean;    // force KV cache reset for this request

  // Tool calling parameters
  tool_choice?: LlamaToolChoice; // default 'auto'
  tools?: LlamaTool[];         // Available tools

  // Advanced parameters (matching llama.cpp server)
  repeat_penalty?: number;     // repetition penalty (native default: 1.0)
  repeat_last_n?: number;      // last n tokens to consider for repetition penalty (default: 64)
  frequency_penalty?: number;   // frequency penalty (default: 0.0)
  presence_penalty?: number;    // presence penalty (default: 0.0)
  seed?: number;                // RNG seed (default: -1, random)
  grammar?: string;             // GBNF grammar for structured outpu
  prompt_id?: string;           // optional. A change from the previous non-empty value clears KV. Empty does not.
  reasoning_budget?: number;    // sampler cap for this call when init turned thinking on. >0 replaces the init cap. 0 and -1 do not, and cannot turn thinking on.
}

export interface LlamaMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: LlamaMessageContent;
  // Prefix key. A leading run of equal non-empty ids is reused. The first empty id ends the run.
  // Reuse an id only when role, content, reasoning_content, and the tool-call payload are unchanged.
  id?: string;
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

export type KvClearedReason =
  | 'none'             // a prefix matched and the tail was dropped
  | 'reset'            // reset_kv_cache was true
  | 'media'            // the turn had image or audio
  | 'prompt_id'        // both calls set prompt_id and they differed
  | 'render_identity'  // Jinja, reasoning format, template, tool choice, grammar, tools, or enable_thinking changed
  | 'no_prefix'        // no leading id matched
  | 'seq_rm';          // a prefix matched and the backend could not drop the tail, so the cache was cleared

export interface LlamaCompletionResult {
  text: string;                          // The generated completion tex
  tokens_predicted: number;              // Number of tokens generated
  timings: {
    predicted_n: number;                 // Number of tokens predicted
    predicted_ms: number;                // Time spent generating tokens (ms)
    prompt_n: number;                    // Tokens evaluated into KV this call. A hit is a small prompt_n beside a large usage.prompt_tokens.
    prompt_ms: number;                   // Time spent processing prompt (ms)
    total_ms: number;                    // Total time spent (ms)
  };
  context_shifted?: boolean;             // true on the call whose generation slid the context window
  kv_cleared_reason?: KvClearedReason;   // why this call encoded from the start of the cache
  reasoning_budget_active?: boolean;     // true only when this call installed a positive thinking-token cap
  usage?: {
    prompt_tokens: number;               // full rendered prompt, hit or miss
    completion_tokens: number;
    total_tokens: number;
  };

  // OpenAI-compatible response fields
  choices?: Array<{
    index: number;
    message: {
      role: string;
      content: string;
      reasoning_content?: string;        // thinking text, when reasoning_format is not 'none'
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
  completion(params: LlamaCompletionParams, partialCallback?: (data: LlamaStreamChunk) => void): Promise<LlamaCompletionResult>;
  completionSync(params: LlamaCompletionParams, partialCallback?: (data: LlamaStreamChunk) => void): LlamaCompletionResult;

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
  stopCompletion(): Promise<void>;

  /**
   * Set the number of CPU threads used for inference (thermal management).
   *
   * Non-blocking: returns immediately. The JS thread is never stalled.
   * The new count takes effect before the next `llama_decode` call.
   *
   * Call from platform thermal event handlers:
   * ```ts
   * // iOS: NSProcessInfo.processInfo.thermalState (nominal/fair/serious/critical)
   * // Android: PowerManager.getThermalStatus()
   * const factors = { nominal: 1.0, fair: 0.75, serious: 0.5, critical: 0.25 };
   * model.setNThreads(Math.max(1, Math.round(baseThreads * factors[thermalState])));
   * ```
   */
  setNThreads(n: number): void;

  /**
   * Release the model and free all associated GPU/CPU memory.
   *
   * **You MUST call this method when the model is no longer needed.**
   * Failing to call `release()` leaks the model's memory (GPU VRAM and/or RAM)
   * for the lifetime of the process. On iOS, unreleased models accumulate memory
   * pressure and will trigger a Jetsam kill of your app. On Android, they exhaust
   * the process heap and cause OOM crashes.
   *
   * Typical usage:
   * ```ts
   * const ctx = await initLlama({ model: '/path/to/model.gguf' });
   * try {
   *   const result = await ctx.completion({ prompt: 'Hello' });
   * } finally {
   *   await ctx.release(); // Always release, even on error
   * }
   * ```
   */
  release(): Promise<void>;

  isMultimodalEnabled(): Promise<boolean>;
  getSupportedModalities(): Promise<{
    vision: boolean;
    audio: boolean;
    audioSampleRate?: number;
  }>;

  embedImage(imagePath: string, options?: { normalize?: boolean }): Promise<ImageEmbedResult>;

  // Needs capabilities: ['audio-transcribe']. options.language is not read yet. Output cap: 2048 tokens.
  transcribeAudio(audioPath: string, options?: { language?: string }): Promise<TranscriptResult>;

  // Needs capabilities: ['vision-reasoning']. raw_text is the unparsed reply. Output cap: 2048 tokens.
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
  // Fires during initLlama() while the GGUF model (phase: 'model') and, if
  // mmproj was supplied, the multimodal projector (phase: 'mmproj') load.
  // Purely advisory — initLlama()'s resolved Promise is the source of truth
  // for "loading finished".
  readonly onModelLoadProgress: CodegenTypes.EventEmitter<ModelLoadProgressEvent>;

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
    totalMemoryBytes: number;   // physical device memory. The context ceiling uses this, not free memory.
    kvBytesPerToken?: number;   // f16 K+V bytes for one token. Absent for hybrid and MLA models. initLlama does not clamp n_ctx.
    samplingDefaults?: {
      temperature?: number;
      top_p?: number;
      top_k?: number;
      min_p?: number;
      repeat_penalty?: number;
      repeat_last_n?: number;
      mirostat?: number;
      mirostat_tau?: number;
      mirostat_eta?: number;
    };
  }>;
}

const LlamaCppRn = TurboModuleRegistry.getEnforcing<Spec>('RNLlamaCpp');
/**
 * LlamaModel type representing a loaded model instance
 */
export type LlamaModel = LlamaContextType & LlamaContextMethods;

/**
 * Initialize a Llama context with the given model parameters.
 *
 * Loads the model into memory (GPU VRAM and/or RAM). The returned context
 * holds a reference to the loaded model until `release()` is called.
 *
 * **Memory management:** Always call `context.release()` when the model is no
 * longer needed. Failing to do so leaks the model's GPU/CPU memory for the
 * lifetime of the process. Calling `initLlama` multiple times without releasing
 * previous contexts will accumulate memory and eventually cause OOM crashes or
 * iOS Jetsam kills.
 *
 * @param params Model loading parameters (see `LlamaModelParams`)
 * @returns A context object with completion, tokenize, embedding, and release methods
 * @see {@link LlamaContextMethods.release} for memory cleanup
 */
export function initLlama(params: LlamaModelParams): Promise<LlamaContextType & LlamaContextMethods> {
  return LlamaCppRn.initLlama(params);
}

/**
 * Subscribe to model-load progress updates emitted during `initLlama()`.
 *
 * Advisory only — for loading UX (progress bars, etc). `initLlama()`'s
 * resolved Promise remains the source of truth for "loading finished".
 *
 * ```ts
 * const sub = addModelLoadProgressListener(({ phase, progress }) => {
 *   console.log(`${phase}: ${Math.round(progress * 100)}%`);
 * });
 * await initLlama({ model: '/path/to/model.gguf' });
 * sub.remove();
 * ```
 */
export function addModelLoadProgressListener(
  listener: (event: ModelLoadProgressEvent) => void
): EventSubscription {
  return LlamaCppRn.onModelLoadProgress(listener);
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
  n_layers: number;
  model_size_bytes: number;
  description: string;
  gpuSupported: boolean;
  optimalGpuLayers: number;
  quant_type?: string;
  architecture: string;
  availableMemoryMB: number;
  estimatedVramMB: number;
  mmprojSizeMB?: number;
  suggestedChunkSize: number;
  isCpuOnly: boolean;
  totalMemoryBytes: number;
  kvBytesPerToken?: number;
  samplingDefaults?: {
    temperature?: number;
    top_p?: number;
    top_k?: number;
    min_p?: number;
    repeat_penalty?: number;
    repeat_last_n?: number;
    mirostat?: number;
    mirostat_tau?: number;
    mirostat_eta?: number;
  };
}> {
  return LlamaCppRn.loadLlamaModelInfo(modelPath, mmprojPath);
}

export default LlamaCppRn;
