// Import everything from NativeRNLlamaCpp.ts
import RNLlamaCppInstance, {
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

// Provide a way to access the raw native module's multiply method
export function multiply(a: number, b: number): number {
  return RNLlamaCppInstance.multiply(a, b);
}

// Optional: If direct access to the TurboModule instance is desired by library users.
// export { RNLlamaCppInstance as RNLlamaCppModule };
