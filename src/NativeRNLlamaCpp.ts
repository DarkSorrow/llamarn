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

  // LoRA adapters
  lora_adapters?: Array<{
    path: string;             // path to LoRA adapter file
    scale?: number;           // scaling factor for the adapter (default: 1.0)
  }>;

  // Grammar-based sampling
  grammar?: string;           // GBNF grammar for grammar-based sampling
}

export interface LlamaCompletionParams {
  // Basic completion parameters
  prompt?: string;            // text promp
  system_prompt?: string;     // system prompt for chat mode (alternative to including it in messages)
  messages?: LlamaMessage[];  // chat messages
  temperature?: number;        // sampling temperature (default: 0.8)
  top_p?: number;              // top-p sampling (default: 0.95)
  top_k?: number;              // top-k sampling (default: 40)
  n_predict?: number;          // max tokens to predict (default: -1, infinite)
  max_tokens?: number;         // alias for n_predic
  stop?: string[];             // stop sequences
  stream?: boolean;            // stream tokens as they're generated (default: true)
  // Chat parameters
  chat_template?: string;      // optional chat template name to use

  // Tool calling parameters
  tool_choice?: string | 'auto' | 'none'; // Tool choice mode
  tools?: LlamaTool[];         // Available tools

  // Advanced parameters (matching llama.cpp server)
  repeat_penalty?: number;     // repetition penalty (default: 1.1)
  repeat_last_n?: number;      // last n tokens to consider for repetition penalty (default: 64)
  frequency_penalty?: number;   // frequency penalty (default: 0.0)
  presence_penalty?: number;    // presence penalty (default: 0.0)
  seed?: number;                // RNG seed (default: -1, random)
  grammar?: string;             // GBNF grammar for structured outpu
}

export interface LlamaMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
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
}

export interface Spec extends TurboModule {
  // Define your module's methods here
  multiply(a: number, b: number): number;

  // Initialize a Llama context with the given model parameters
  initLlama(params: LlamaModelParams): Promise<LlamaContextType & LlamaContextMethods>;

  // Load model info without creating a full contex
  loadLlamaModelInfo(modelPath: string): Promise<{
    n_params: number;
    n_vocab: number;
    n_context: number;
    n_embd: number;
    description: string;
    gpuSupported: boolean;
    optimalGpuLayers: number;
    quant_type?: string;
    architecture?: string;
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
 * Get information about a model without loading it fully
 */
export function loadLlamaModelInfo(
  modelPath: string
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
}> {
  return LlamaCppRn.loadLlamaModelInfo(modelPath);
}

export default LlamaCppRn;
