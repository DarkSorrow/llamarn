// Import everything from NativeRNLlamaCpp.ts
import {
  loadLlamaModelInfo,
  initLlama,
  // Import all necessary types as well
  type LlamaModel,
  type LlamaModelParams,
  type LlamaCompletionParams,
  type LlamaContextType,
  type LlamaMessage,
  type JsonSchemaObject,
  type JsonSchemaArray,
  type JsonSchemaScalar,
  type JsonSchemaProperty,
  type LlamaTool,
  type LlamaCompletionResult,
  type EmbeddingOptions,
  type EmbeddingResponse,
  type LlamaContextMethods,
  type Spec,
} from './NativeRNLlamaCpp';

// Export the helper functions directly
export {
  loadLlamaModelInfo,
  initLlama,
};

// Export the types directly
export type {
  LlamaModel,
  LlamaModelParams,
  LlamaCompletionParams,
  LlamaContextType,
  LlamaMessage,
  JsonSchemaObject,
  JsonSchemaArray,
  JsonSchemaScalar,
  JsonSchemaProperty,
  LlamaTool,
  LlamaCompletionResult,
  EmbeddingOptions,
  EmbeddingResponse,
  LlamaContextMethods,
  Spec,
};

// Optional: If direct access to the TurboModule instance is desired by library users.
// export { RNLlamaCppInstance as RNLlamaCppModule };
