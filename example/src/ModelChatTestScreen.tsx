import React, { useState, useRef, useEffect } from 'react';
import {
  StyleSheet,
  View,
  Text,
  Button,
  FlatList,
  ActivityIndicator,
  TextInput,
  Platform,
  TouchableOpacity,
  KeyboardAvoidingView,
  InteractionManager,
} from 'react-native';
import { initLlama, loadLlamaModelInfo } from '@novastera-oss/llamarn';
import type { LlamaModel, LlamaMessage, LlamaTool } from '@novastera-oss/llamarn';
import RNFS from 'react-native-fs';

// Use smaller model for Android to avoid build size issues
const modelFileName = Platform.OS === 'android' 
  ? "Qwen3.5-0.8B-Q4_K_M.gguf"  // 770MB - smaller for Android
  : "Qwen3.5-2B-Q4_K_M.gguf"; // 4.1GB - full model for iOS
//const modelFileName = "Qwen3-1.7B-Q4_K_M.gguf";

interface Message {
  /** Stable ID used by the native KV cache to skip re-encoding unchanged messages. */
  id?: string;
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string;
  /** Thinking/reasoning text extracted from <think>…</think> blocks (thinking models only). */
  reasoning_content?: string;
  name?: string;
  tool_call_id?: string;
  isToolCall?: boolean; // Simple flag to indicate if this is a tool call message
  /** Present on assistant turns that only requested tools (needed for follow-up completion). */
  tool_calls?: Array<{
    id?: string;
    type?: string;
    function?: { name?: string; arguments?: string };
  }>;
}

/** Generates a short stable ID for KV cache message tracking. */
const makeId = (): string =>
  Date.now().toString(36) + Math.random().toString(36).slice(2, 7);

type ModelMode = 'conversation' | 'tools' | 'embeddings';

interface ModelState {
  instance: LlamaModel;
  info: any;
  mode: ModelMode;
  samplingDefaults: {
    temperature?: number;
    top_p?: number;
    top_k?: number;
    min_p?: number;
    repeat_penalty?: number;
    repeat_last_n?: number;
  };
}

const extractResponseText = (response: any): string => {
  if (typeof response?.text === 'string' && response.text.trim().length > 0) {
    return response.text;
  }
  return response?.choices?.[0]?.message?.content || '';
};

/** Lets the native thread paint so batched chat updates appear step-by-step on device. */
const yieldToUI = () =>
  new Promise<void>(resolve => {
    InteractionManager.runAfterInteractions(() => {
      setTimeout(resolve, 50);
    });
  });

/** Normalize tool calls from chat completion (top-level or OpenAI choices shape). */
const extractToolCalls = (response: any): NonNullable<Message['tool_calls']> => {
  if (!response) {
    return [];
  }
  const top = response.tool_calls;
  if (Array.isArray(top) && top.length > 0) {
    return top;
  }
  const choice = response.choices?.[0];
  const fromMessage = choice?.message?.tool_calls;
  if (Array.isArray(fromMessage) && fromMessage.length > 0) {
    return fromMessage;
  }
  if (choice?.finish_reason === 'tool_calls' && Array.isArray(fromMessage)) {
    return fromMessage;
  }
  return [];
};

/**
 * Strip <think>…</think> from the start of model output.
 * Thinking models (Qwen3, DeepSeek-R1, etc.) embed reasoning in the content field.
 * This must be separated before adding the message to history so the chat template
 * receives properly structured messages ({content, reasoning_content}) on subsequent turns.
 */
const extractThinking = (content: string): { thinking: string | null; content: string } => {
  // Allow optional leading whitespace before <think> — models sometimes emit a newline first.
  const match = content.match(/^\s*<think>([\s\S]*?)<\/think>\s*/);
  if (!match) return { thinking: null, content };
  return { thinking: (match[1] ?? '').trim(), content: content.slice(match[0].length) };
};

/** Map our chat Message to the payload the native module expects (incl. tool_calls). */
const messageToApiPayload = (msg: Message): Record<string, unknown> => {
  const payload: Record<string, unknown> = {
    role: msg.role,
    ...(msg.id ? { id: msg.id } : {}),
  };
  const hasTools = Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0;
  // Always use a string for `content`. The GGUF chat template (e.g. Qwen3) runs in minimal Jinja and
  // does message.content.split(...).lstrip(...) — null/undefined is not a string there and crashes
  // with "Callee is not a function ... (hint: 'lstrip')". OpenAI allows null for tool-only
  // assistant turns; our native path needs "" instead.
  let content = msg.content ?? '';
  // The Qwen3 template crashes when an assistant message has </think> in content but no
  // reasoning_content: it runs split('</think>')[0].rstrip().split('<think>')[-1].lstrip(),
  // and if the before-</think> substring is empty, split('<think>') returns [] causing [-1]
  // to be out-of-bounds → Undefined → lstrip() throws. Strip any residual think blocks.
  if (msg.role === 'assistant' && !msg.reasoning_content && content.includes('</think>')) {
    content = content.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
  }
  payload.content = content;
  if (msg.reasoning_content) {
    payload.reasoning_content = msg.reasoning_content;
  }
  if (msg.name) {
    payload.name = msg.name;
  }
  if (msg.tool_call_id) {
    payload.tool_call_id = msg.tool_call_id;
  }
  if (hasTools) {
    payload.tool_calls = msg.tool_calls;
  }
  return payload;
};

const formatToolCallArgs = (tc: { function?: { name?: string; arguments?: unknown } }): string => {
  const raw = tc.function?.arguments;
  if (raw == null) {
    return '{}';
  }
  if (typeof raw === 'string') {
    return raw;
  }
  try {
    return JSON.stringify(raw, null, 2);
  } catch {
    return String(raw);
  }
};

const getPerformanceSummary = (response: any): string | null => {
  const perf = response?.performance || response?.timings || null;
  if (!perf || typeof perf !== 'object') {
    return null;
  }

  const tokensPerSecond =
    perf.tokens_per_second ?? perf.tokensPerSecond ?? perf.predicted_per_second;
  const promptTokens = perf.prompt_tokens ?? perf.promptTokens;
  const completionTokens = perf.completion_tokens ?? perf.completionTokens;

  const parts: string[] = [];
  if (typeof tokensPerSecond === 'number') {
    parts.push(`tok/s ${tokensPerSecond.toFixed(2)}`);
  }
  if (typeof promptTokens === 'number') {
    parts.push(`prompt ${promptTokens}`);
  }
  if (typeof completionTokens === 'number') {
    parts.push(`completion ${completionTokens}`);
  }

  return parts.length > 0 ? `\n\n[Performance] ${parts.join(' | ')}` : null;
};

const hashString = (value: string): string => {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, '0');
};

const stableStringify = (value: Record<string, unknown>): string => {
  const orderedKeys = Object.keys(value).sort();
  const ordered: Record<string, unknown> = {};
  orderedKeys.forEach(key => {
    ordered[key] = value[key];
  });
  return JSON.stringify(ordered);
};

const withTimeout = async <T,>(promise: Promise<T>, ms: number, label: string): Promise<T> => {
  let timeout: ReturnType<typeof setTimeout> | null = null;
  try {
    return await Promise.race([
      promise,
      new Promise<T>((_, reject) => {
        timeout = setTimeout(() => {
          reject(new Error(`${label} timed out after ${ms}ms`));
        }, ms);
      }),
    ]);
  } finally {
    if (timeout) {
      clearTimeout(timeout);
    }
  }
};

const isPromiseLike = (value: unknown): value is Promise<unknown> =>
  !!value && typeof (value as { then?: unknown }).then === 'function';

// Model Loading Component
const ModelLoader: React.FC<{
  onModelLoaded: (state: ModelState) => void;
}> = ({ onModelLoaded }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const getModelPath = async (): Promise<string> => {
    let modelPath = modelFileName;
    
    if (Platform.OS === 'ios') {
      modelPath = RNFS.MainBundlePath + '/' + modelFileName;
      console.log(`[iOS] Using full model path: ${modelPath}`);
    } else if (Platform.OS === 'android') {
      // For Android, copy asset to cache directory
      const tempDir = RNFS.CachesDirectoryPath;
      const tempModelPath = `${tempDir}/${modelFileName}`;
      
      console.log(`[Android] Copying asset to temp location: ${tempModelPath}`);
      
      const tempFileExists = await RNFS.exists(tempModelPath);
      
      if (!tempFileExists) {
        console.log(`[Android] Temp file doesn't exist, copying from assets...`);
        await RNFS.copyFileAssets(modelFileName, tempModelPath);
        console.log(`[Android] Successfully copied asset to: ${tempModelPath}`);
      } else {
        console.log(`[Android] Using existing temp file: ${tempModelPath}`);
      }
      
      modelPath = tempModelPath;
    }
    
    return modelPath;
  };

  const initializeModel = async (mode: ModelMode) => {
    setLoading(true);
    setError(null);
    
    try {
      const modelPath = await getModelPath();
      console.log('Initializing model from:', modelPath);
      
      // Get model info first
      const info = await loadLlamaModelInfo(modelPath);
      console.log('Model info:', info);
      
      // Initialize with mode-specific settings
      const initParams: any = {
        model: modelPath,
        n_ctx: mode === 'embeddings' ? 512 : 2048,
        n_batch: 128,
        n_gpu_layers: 0,
        use_mlock: true,
        embedding: mode === 'embeddings',
      };

      // Add mode-specific parameters
      if (mode === 'tools') {
        initParams.use_jinja = true;
      }

      console.log('Initializing model with settings:', initParams);
      const modelInstance = await initLlama(initParams);
      
      console.log('Model initialized successfully');
      
      onModelLoaded({
        instance: modelInstance,
        info,
        mode,
        samplingDefaults: info.samplingDefaults ?? {},
      });
      
    } catch (err) {
      console.error('Model initialization failed:', err);
      setError(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.initContainer}>
      <Text style={styles.title}>Load Model for</Text>
      
      <View style={styles.modeButtons}>
        <TouchableOpacity 
          style={[styles.modeButton, loading && styles.modeButtonDisabled]}
          onPress={() => initializeModel('conversation')}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator size="small" color="white" />
          ) : (
            <Text style={styles.modeButtonText}>Conversation</Text>
          )}
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={[styles.modeButton, loading && styles.modeButtonDisabled]}
          onPress={() => initializeModel('tools')}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator size="small" color="white" />
          ) : (
            <Text style={styles.modeButtonText}>Tools</Text>
          )}
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={[styles.modeButton, loading && styles.modeButtonDisabled]}
          onPress={() => initializeModel('embeddings')}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator size="small" color="white" />
          ) : (
            <Text style={styles.modeButtonText}>Embeddings</Text>
          )}
        </TouchableOpacity>
      </View>

      {loading && (
        <View style={styles.loaderContainer}>
          <ActivityIndicator size="large" color="#007bff" style={styles.loader} />
          <Text style={styles.loadingText}>Loading model...</Text>
        </View>
      )}

      {error && (
        <View style={styles.errorContainer}>
          <Text style={styles.error}>{error}</Text>
        </View>
      )}
    </View>
  );
};

// Model Header Component
const ModelHeader: React.FC<{
  modelInfo: any;
  mode: ModelMode;
  onUnload: () => void;
}> = ({ modelInfo, mode, onUnload }) => {
  return (
    <View style={styles.header}>
      <Text style={styles.modelName}>
        {modelInfo?.description || 'AI Assistant'} ({mode})
      </Text>
      <TouchableOpacity 
        style={styles.unloadButton} 
        onPress={onUnload}
      >
        <Text style={styles.unloadButtonText}>Unload Model</Text>
      </TouchableOpacity>
    </View>
  );
};

// Format tool response JSON for display
const formatToolResponse = (content: string): string => {
  try {
    // Try to parse as JSON for better display
    const data = JSON.parse(content);
    
    // Format weather data specially
    if (data.location && data.temperature !== undefined) {
      return `Location: ${data.location}\nTemperature: ${data.temperature}°${data.unit === 'fahrenheit' ? 'F' : 'C'}\nCondition: ${data.condition}\nHumidity: ${data.humidity}%`;
    }
    
    // Format location data specially
    if (data.city && data.coordinates) {
      return `City: ${data.city}, ${data.state}, ${data.country}\nCoordinates: ${data.coordinates.latitude}, ${data.coordinates.longitude}`;
    }
    
    // Format other JSON responses
    return Object.entries(data)
      .map(([key, value]) => `${key}: ${typeof value === 'object' ? JSON.stringify(value) : value}`)
      .join('\n');
  } catch (e) {
    // If not valid JSON, return as is
    return content;
  }
};

// Main Component
export default function ModelChatTestScreen() {
  const [modelState, setModelState] = useState<ModelState | null>(null);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([
    { id: makeId(), role: 'system', content: 'You are a helpful AI assistant.' }
  ]);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [streamingText, setStreamingText] = useState('');
  
  const listRef = useRef<FlatList<Message>>(null);
  const messagesRef = useRef<Message[]>(messages);

  const updateMessages = (updater: (prev: Message[]) => Message[]) => {
    setMessages(prev => {
      const next = updater(prev);
      messagesRef.current = next;
      return next;
    });
  };

  const replaceMessages = (next: Message[]) => {
    messagesRef.current = next;
    setMessages(next);
  };

  const toModelMessages = (history: Message[]): LlamaMessage[] =>
    history
      .filter(m => !m.isToolCall)
      .map(m => messageToApiPayload(m) as unknown as LlamaMessage);

  const logHistoryDebug = (history: Message[], label: string) => {
    const compact = history
      .filter(m => !m.isToolCall)
      .map(m => ({
        id: m.id,
        role: m.role,
        name: m.name,
        tool_call_id: m.tool_call_id,
        has_tool_calls: !!m.tool_calls?.length,
        content: (m.content ?? '').slice(0, 140),
      }));
    console.log(`[ChatDebug] ${label}:`, JSON.stringify(compact, null, 2));
  };

  // Handle tool call
  const handleToolCall = async (toolCall: any) => {
    console.log('Tool call received:', toolCall);
    
    try {
      // Extract function name and arguments
      const functionName = toolCall.function?.name;
      const functionArgs = toolCall.function?.arguments || '{}';
      
      // Improved argument parsing with multiple fallbacks
      let parsedArgs;
      try {
        // Try parsing as JSON string
        parsedArgs = typeof functionArgs === 'string' ? JSON.parse(functionArgs) : functionArgs;
      } catch (e) {
        console.log('Error parsing function arguments as JSON:', e);
        // Try extracting location from a non-JSON string
        if (typeof functionArgs === 'string') {
          const locationMatch = functionArgs.match(/location[\"']?\s*[:=]\s*[\"']([^\"']+)[\"']/i);
          if (locationMatch && locationMatch[1]) {
            parsedArgs = { location: locationMatch[1] };
          } else {
            // Very liberal fallback - assume the entire string is a location
            parsedArgs = { location: functionArgs.replace(/[{}"']/g, '').trim() };
          }
        } else {
          // Default to empty object if all else fails
          parsedArgs = {};
        }
      }
      
      console.log('Parsed arguments:', parsedArgs);
      
      if (functionName === 'get_weather' || functionName === 'weather') {
        // Parse the arguments
        const location = parsedArgs.location || parsedArgs.city || 'Unknown';
        const unit = parsedArgs.unit || 'celsius';
        
        console.log(`Getting weather for location: ${location}, unit: ${unit}`);
        
        // Simulate weather data
        const weatherData = {
          location,
          temperature: Math.floor(Math.random() * 30) + 5, // 5-35°C
          unit,
          condition: ['Sunny', 'Cloudy', 'Rainy', 'Partly cloudy', 'Stormy'][Math.floor(Math.random() * 5)],
          humidity: Math.floor(Math.random() * 60) + 30, // 30-90%
        };

        // Send raw JSON — the Qwen3 template wraps it in <tool_response> tags and the model
        // expects structured JSON, not emoji-formatted display text.
        const toolMessage: Message = {
          id: makeId(),
          role: 'tool',
          content: JSON.stringify(weatherData),
          name: functionName,
          tool_call_id: toolCall.id
        };
        
        return toolMessage;
      }
      else if (functionName === 'get_location') {
        // Simulate location data
        const locationData = {
          city: 'San Francisco',
          state: 'CA',
          country: 'USA',
          coordinates: {
            latitude: 37.7749,
            longitude: -122.4194
          }
        };
        
        // Send raw JSON — the model expects structured JSON in tool responses.
        const toolMessage: Message = {
          id: makeId(),
          role: 'tool',
          content: JSON.stringify(locationData),
          name: functionName,
          tool_call_id: toolCall.id
        };
        
        return toolMessage;
      }
      
      throw new Error(`Unknown tool: ${functionName}`);
    } catch (err) {
      console.error('Error handling tool call:', err);
      const errorMessage: Message = {
        id: makeId(),
        role: 'tool',
        content: JSON.stringify({ error: `Error processing tool call: ${err}` }),
        name: toolCall.function?.name || 'unknown',
        tool_call_id: toolCall.id
      };
      
      return errorMessage;
    }
  };

  // Handle model loaded event
  const handleModelLoaded = (state: ModelState) => {
    // Set appropriate initial system message based on mode
    let initialMessages: Message[] = [];
    
    if (state.mode === 'tools') {
      // System message for the weather tool
      initialMessages = [{
        id: makeId(),
        role: 'system',
        content:
`You are a helpful AI assistant with access to tools. Use tools only when necessary.

Rules:
1. For weather information: Use the get_weather tool
2. For location information: Use the get_location tool
3. For general questions: Answer directly without tools
4. Only call tools when necessary to answer the user's specific question
5. Don't make multiple tool calls for the same information
6. Keep responses concise and helpful`
      }];
    } else {
      // Default system message with Qwen3 thinking mode
      initialMessages = [{
        id: makeId(),
        role: 'system',
        content: 'You are a helpful AI assistant.'
      }];
    }
    
    replaceMessages(initialMessages);
    setModelState(state);
  };

  // Handle model unload
  const handleUnload = async () => {
    if (modelState?.instance) {
      try {
        await modelState.instance.release();
        console.log('Model unloaded successfully');
        setModelState(null);
        replaceMessages([{ id: makeId(), role: 'system', content: 'You are a helpful AI assistant.' }]);
        setStreamingText('');
      } catch (err) {
        console.error('Failed to unload model:', err);
        setError(`Release Error: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  };

  // Add streaming token for UI display
  const handleStreamingToken = (token: string) => {
    console.log('[Stream] token:', token);
    setStreamingText(prev => prev + token);
  };

  const runNativeSafetyChecks = async () => {
    if (!modelState?.instance) return;

    setGenerating(true);
    setError(null);
    setStreamingText('');

    const checkLog: string[] = [];
    try {
      const toolNames =
        modelState.mode === 'tools' ? ['get_location', 'get_weather'] : [];

      const promptSignature = {
        mode: modelState.mode,
        systemPromptVersion: 'safety-check-v1',
        toolNames: toolNames.join(','),
      };

      const baseConfig = {
        temperature: 0.6,
        top_p: 0.95,
        top_k: 20,
        min_p: 0.05,
        repeat_penalty: 1.1,
        repeat_last_n: 64,
        presence_penalty: 0.0,
        token_rate_cap: 30,
        token_buffer_size: 4,
      };

      const prompt_id = `prompt-${hashString(stableStringify(promptSignature))}`;
      const config_id = `config-${hashString(stableStringify(baseConfig))}`;

      const history: Message[] = [
        { id: 'chk-sys', role: 'system', content: 'You are a concise assistant.' },
        { id: 'chk-u1', role: 'user', content: 'Reply with exactly READY' },
      ];

      const optionsBase = {
        temperature: baseConfig.temperature,
        top_p: baseConfig.top_p,
        top_k: baseConfig.top_k,
        min_p: baseConfig.min_p,
        repeat_penalty: baseConfig.repeat_penalty,
        repeat_last_n: baseConfig.repeat_last_n,
        presence_penalty: baseConfig.presence_penalty,
        token_rate_cap: baseConfig.token_rate_cap,
        token_buffer_size: baseConfig.token_buffer_size,
        max_tokens: 64,
        prompt_id,
        config_id,
      };

      const first = await withTimeout(
        modelState.instance.completion({
          ...optionsBase,
          messages: history.map(m => messageToApiPayload(m)) as unknown as LlamaMessage[],
        }),
        25000,
        'Warmup completion',
      );
      const firstText = extractResponseText(first);
      if (!firstText.trim()) {
        throw new Error('Warmup completion returned empty content');
      }
      checkLog.push('PASS: warmup completion succeeded');

      history.push({
        id: 'chk-a1',
        role: 'assistant',
        content: firstText,
      });
      history.push({
        id: 'chk-u2',
        role: 'user',
        content: 'Now reply with exactly SECOND',
      });

      const second = await withTimeout(
        modelState.instance.completion({
          ...optionsBase,
          messages: history.map(m => messageToApiPayload(m)) as unknown as LlamaMessage[],
        }),
        25000,
        'Cache reuse completion',
      );
      const secondText = extractResponseText(second);
      if (!secondText.trim()) {
        throw new Error('Cache reuse completion returned empty content');
      }
      checkLog.push('PASS: reuse-path completion succeeded');

      const newConfig = {
        ...baseConfig,
        temperature: 0.3,
      };
      const changedConfigId = `config-${hashString(stableStringify(newConfig))}`;
      const third = await withTimeout(
        modelState.instance.completion({
          ...optionsBase,
          temperature: newConfig.temperature,
          config_id: changedConfigId,
          messages: history.map(m => messageToApiPayload(m)) as unknown as LlamaMessage[],
        }),
        25000,
        'Config shift completion',
      );
      const thirdText = extractResponseText(third);
      if (!thirdText.trim()) {
        throw new Error('Config shift completion returned empty content');
      }
      checkLog.push('PASS: config_id shift completion succeeded');

      const abortPrompt = `${'abort-check '.repeat(4000)}END`;
      const abortStart = Date.now();
      const abortRun = withTimeout(
        modelState.instance.completion({
          prompt: abortPrompt,
          max_tokens: 256,
          token_buffer_size: 4,
          token_rate_cap: 20,
          temperature: 0.6,
        }),
        30000,
        'Abort responsiveness completion',
      );

      setTimeout(() => {
        try {
          const stopResult = modelState.instance.stopCompletion();
          if (isPromiseLike(stopResult)) {
            stopResult.catch(e => {
              console.warn('stopCompletion failed during safety check:', e);
            });
          }
        } catch (e) {
          console.warn('stopCompletion threw during safety check:', e);
        }
      }, 150);

      await abortRun;
      const abortElapsed = Date.now() - abortStart;
      checkLog.push(`PASS: stopCompletion path returned in ${abortElapsed}ms`);

      setError(`Native safety checks complete:\n${checkLog.join('\n')}`);
    } catch (err) {
      setError(`Native safety checks failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setGenerating(false);
    }
  };

  const sendMessage = async () => {
    if (!modelState?.instance || !input.trim()) return;
    
    const userMessage: Message = { id: makeId(), role: 'user', content: input.trim() };
    const currentMessages = [...messagesRef.current, userMessage];
    replaceMessages(currentMessages);
    setInput('');
    setGenerating(true);
    setError(null);
    setStreamingText('');

    // Build sampling params: GGUF model defaults, then mode-specific overrides.
    // JS-supplied values always win; omitting a field lets the native layer use initLlama defaults.
    const sd = modelState.samplingDefaults;
    const baseSampling = {
      temperature:    sd.temperature    ?? 0.8,
      top_p:          sd.top_p          ?? 0.9,
      top_k:          sd.top_k          ?? 40,
      min_p:          sd.min_p          ?? 0.05,
      repeat_penalty: sd.repeat_penalty ?? 1.1,
      repeat_last_n:  sd.repeat_last_n  ?? 64,
    };
    
    try {
      const completionOptions: any = {
        messages: toModelMessages(currentMessages),
        // Qwen3 thinking mode settings (better for complex reasoning about tool usage)
        temperature: 0.6,
        top_p: 0.95,
        top_k: 20,
        min_p: 0.05,          // filter very unlikely tokens — helps small models stay coherent
        repeat_penalty: 1.1,  // primary anti-repetition: penalises recently-seen tokens
        repeat_last_n: 64,    // window for repeat_penalty
        presence_penalty: 0.0, // presence_penalty on top of repeat_penalty is redundant and can over-penalise
        max_tokens: 8192,       // More space for thinking + response
        stop: ["</s>", "<|im_end|>", "<|eot_id|>", "<|eom_id|>"],
      };
      logHistoryDebug(currentMessages, 'Pre-completion history');
      
      // Add tools and tool configuration only in tools mode
      if (modelState.mode === 'tools') {
        completionOptions.tools = [
          // Weather tool
          {
            type: 'function',
            function: {
              name: 'get_weather',
              description: 'Gets the current weather information for a specific city',
              parameters: {
                type: 'object',
                properties: {
                  location: {
                    type: 'string',
                    description: 'The city name for which to get weather information',
                  },
                  unit: {
                    type: 'string',
                    enum: ['celsius', 'fahrenheit'],
                    description: 'The unit of temperature to use (default: celsius)',
                  }
                },
                required: ['location'],
              },
            }
          },
          // Location tool
          {
            type: 'function',
            function: {
              name: 'get_location',
              description: 'Gets the user\'s current location',
              parameters: {
                type: 'object',
                properties: { },
                required: [],
              }
            }
          }
        ] as LlamaTool[];
        completionOptions.tool_choice = "auto";
      }
      
      console.log('Completion options:', completionOptions);
      
      // Get the initial assistant response
      const response = await modelState.instance.completion(
        completionOptions,
        (data: { token: string }) => {
          handleStreamingToken(data.token);
        }
      );
      
      console.log('Response with tool calls:', response);

      const rawContent = extractResponseText(response);
      console.log('[Stream] completion resolved text:', rawContent);
      const { thinking, content: cleanContent } = extractThinking(rawContent);
      const perfSummary = getPerformanceSummary(response);
      const toolCalls = extractToolCalls(response);
      console.log('Tool calls extracted:', toolCalls?.length ?? 0, toolCalls);

      const assistantMessage: Message = {
        id: makeId(),
        role: 'assistant',
        content: cleanContent || 'Sorry, I couldn\'t generate a response.',
        ...(thinking ? { reasoning_content: thinking } : {}),
      };

      // Tool path: show tool call → tool result(s) in chat, then final answer (with yields so UI updates on device).
      if (toolCalls.length > 0) {
        const toolCallsPayload = JSON.parse(JSON.stringify(toolCalls)) as NonNullable<Message['tool_calls']>;
        const assistantToolTurn: Message = {
          id: makeId(),
          role: 'assistant',
          content: rawContent.trim() ? cleanContent : '',
          tool_calls: toolCallsPayload,
          ...(thinking ? { reasoning_content: thinking } : {}),
        };

        // Always track the assistant tool turn in messages for history integrity,
        // even when it has no visible content (model went straight to tool call).
        updateMessages(prev => [...prev, assistantToolTurn]);
        await yieldToUI();

        for (const tc of toolCalls) {
          const callLine: Message = {
            role: 'assistant',
            content: `🔧 Tool call: ${tc.function?.name ?? 'unknown'}\n${formatToolCallArgs(tc)}`,
            isToolCall: true,
          };
          updateMessages(prev => [...prev, callLine]);
          await yieldToUI();
        }

        const toolResultMessages: Message[] = [];
        for (const tc of toolCalls) {
          const toolResponse = await handleToolCall(tc);
          toolResultMessages.push(toolResponse);
          updateMessages(prev => [...prev, toolResponse]);
          await yieldToUI();
        }

        // Build the API conversation for the follow-up: exclude display-only callLine messages.
        const allMessages: Message[] = [
          ...currentMessages.filter(m => !m.isToolCall),
          assistantToolTurn,
          ...toolResultMessages,
        ];
        logHistoryDebug(allMessages, 'Tool follow-up history');

        setStreamingText('');
        const finalResponse = await modelState.instance.completion(
          {
            ...completionOptions,
            messages: toModelMessages(allMessages),
            tools: undefined,
            tool_choice: undefined,
            temperature: 0.6,
            top_p: 0.95,
            top_k: 20,
            min_p: 0.05,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            presence_penalty: 0.0,
            max_tokens: 4096,
          },
          (data: { token: string }) => {
            handleStreamingToken(data.token);
          },
        );

        console.log('Final response after tool call:', finalResponse);

        const finalRaw = extractResponseText(finalResponse);
        console.log('[Stream] tool follow-up resolved text:', finalRaw);
        const { thinking: finalThinking, content: finalClean } = extractThinking(finalRaw);
        const finalMessage: Message = {
          id: makeId(),
          role: 'assistant',
          content: finalClean || 'I couldn\'t process the tool response.',
          ...(finalThinking ? { reasoning_content: finalThinking } : {}),
        };
        updateMessages(prev => [...prev, finalMessage]);
      } else {
        updateMessages(prev => [...prev, assistantMessage]);
      }

      if (perfSummary) {
        console.log('Performance summary:', perfSummary);
      }
      setStreamingText('');
    } catch (err) {
      console.error('Completion error:', err);
      setError(`Error generating response: ${err instanceof Error ? err.message : String(err)}`);
      setStreamingText('');
    } finally {
      setGenerating(false);
    }
  };

  // Test embedding
  const testEmbedding = async () => {
    if (!modelState?.instance) return;
    
    try {
      const testText = input || 'Hello, world';
      
      console.log(`Testing embedding with text: "${testText}"`);
      
      // Generate embedding with float format
      console.log('Generating embedding with float format...');
      const embedding = await modelState.instance.embedding({
        content: testText,
        encoding_format: 'float'
      });
      
      // Log the embedding details
      const embeddingData = embedding.data[0]?.embedding;
      console.log(`Generated embedding with ${Array.isArray(embeddingData) ? embeddingData.length : 'unknown'} dimensions`);
      
      // Verify each value is a valid number
      const allValid = Array.isArray(embeddingData) && embeddingData.every((val: number) => typeof val === 'number' && !isNaN(val));
      console.log('All values are valid numbers:', allValid);
      
      // Calculate some stats
      let statsMessage = '';
      if (Array.isArray(embeddingData)) {
        const min = Math.min(...embeddingData);
        const max = Math.max(...embeddingData);
        const avg = embeddingData.reduce((sum: number, val: number) => sum + val, 0) / embeddingData.length;
        
        const firstFew = embeddingData.slice(0, 5);
        const lastFew = embeddingData.slice(-5);
        statsMessage = 
          `- First values: [${firstFew.map(v => v.toFixed(6)).join(', ')}...]\n` +
          `- Last values: [...${lastFew.map(v => v.toFixed(6)).join(', ')}]\n` +
          `- All values are valid numbers: ${allValid}\n` +
          `- Range: Min=${min.toFixed(6)}, Max=${max.toFixed(6)}, Avg=${avg.toFixed(6)}\n\n`;
      }
      
      // Format results for display
      let resultMessage = 
        `Embedding generated successfully!\n\n` +
        `Float format:\n` +
        `- Vector dimension: ${Array.isArray(embeddingData) ? embeddingData.length : 'N/A'}\n` +
        `- Token count: ${embedding.usage.prompt_tokens}\n` +
        `- Model: ${embedding.model}\n\n` +
        statsMessage;
      
      setError(resultMessage);
      
    } catch (err) {
      console.error('Embedding error:', err);
      setError(`Embedding Error: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  // Test tokenizer
  const testTokenizer = async () => {
    if (!modelState?.instance) return;
    
    try {
      console.log('Testing tokenizer with input:', input || 'Hello, how are you?');
      const testText = input || 'Hello, how are you?';
      
      const result = await modelState.instance.tokenize({
        content: testText,
        add_special: false,
        with_pieces: true
      });
      
      console.log('Tokenization result:', result);
      
      if (result && result.tokens) {
        setError(`Tokenized to ${result.tokens.length} tokens: ${JSON.stringify(result.tokens)}`);
      } else {
        setError('Tokenization returned empty result');
      }
    } catch (err) {
      console.error('Tokenization error:', err);
      setError(`Tokenization Error: ${err instanceof Error ? err.message : String(err)}`);
    }
  };
  
  // Test detokenizer
  const testDetokenize = async () => {
    if (!modelState?.instance) return;
    
    try {
      console.log('Testing detokenize with input:', input || 'Hello, how are you?');
      const testText = input || 'Hello, how are you?';
      
      // First tokenize the text
      const tokenizeResult = await modelState.instance.tokenize({
        content: testText,
        add_special: false
      });
      
      console.log('Tokenization for detokenize test:', tokenizeResult);
      
      if (!tokenizeResult || !tokenizeResult.tokens || !tokenizeResult.tokens.length) {
        setError('Tokenization returned empty result for detokenize test');
        return;
      }
      
      // Extract token IDs and detokenize
      const tokenIds = tokenizeResult.tokens.map((token: any) => 
        typeof token === 'number' ? token : token.id
      );
      
      // Call detokenize with the token IDs
      const detokenizeResult = await modelState.instance.detokenize({
        tokens: tokenIds
      });
      
      console.log('Detokenize result:', detokenizeResult);
      
      if (detokenizeResult && detokenizeResult.content) {
        setError(`Original: "${testText}"\nDetokenized: "${detokenizeResult.content}"`);
      } else {
        setError('Detokenize returned empty result');
      }
    } catch (err) {
      console.error('Detokenize error:', err);
      setError(`Detokenize Error: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  // Keep chat pinned to latest item during updates.
  useEffect(() => {
    setTimeout(() => {
      listRef.current?.scrollToEnd({ animated: true });
    }, 60);
  }, [messages, streamingText, generating]);

  // When in tools mode, show example question and debug buttons
  const renderToolModeControls = () => {
    if (modelState?.mode !== 'tools') return null;
    
    return (
      <View style={styles.toolModeContainer}>
        <Text style={styles.toolModeText}>
          <Text style={styles.toolModeHighlight}>Tool Mode Active:</Text> Ask about weather or location
        </Text>
        <View style={styles.toolButtonsContainer}>
          <TouchableOpacity
            style={styles.exampleButton}
            onPress={() => setInput("What's the weather in Paris?")}
          >
            <Text style={styles.exampleButtonText}>Weather Example</Text>
          </TouchableOpacity>
          
          <TouchableOpacity
            style={styles.exampleButton}
            onPress={() => setInput("Where am I located?")}
          >
            <Text style={styles.exampleButtonText}>Location Example</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  };

  // Render a message
  const renderMessage = (msg: Message, index: number) => {
    // Skip display-only assistant turns that have no visible content (e.g. empty tool-call turns).
    if (msg.role === 'assistant' && !msg.isToolCall && !msg.content && !msg.tool_calls?.length) {
      return null;
    }

    const getMessageStyle = () => {
      if (msg.role === 'user') return styles.userMessage;
      if (msg.role === 'tool') return styles.toolResponseMessage;
      if (msg.isToolCall) return styles.toolCallMessage;
      return styles.assistantMessage;
    };

    const getMessageLabel = () => {
      if (msg.role === 'user') return 'You';
      if (msg.role === 'tool') return `Tool Response (${msg.name || 'unknown'})`;
      if (msg.isToolCall) return 'Tool Call';
      return 'Assistant';
    };

    // Tool result messages store raw JSON for the model; format them nicely for display.
    const displayContent = msg.role === 'tool'
      ? formatToolResponse(msg.content)
      : msg.content;

    return (
      <View
        key={index}
        style={[
          styles.messageWrapper,
          getMessageStyle()
        ]}
      >
        <Text style={styles.messageSender}>
          {getMessageLabel()}
        </Text>
        <Text style={styles.messageContent}>{displayContent}</Text>
      </View>
    );
  };

  // Render the appropriate interface based on model state and mode
  const renderInterface = () => {
    if (!modelState) {
      return <ModelLoader onModelLoaded={handleModelLoaded} />;
    }

    return (
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.chatContainer}
        keyboardVerticalOffset={100}
      >
        <ModelHeader 
          modelInfo={modelState.info}
          mode={modelState.mode}
          onUnload={handleUnload}
        />

        {modelState.mode === 'embeddings' ? (
          <View style={styles.testTabContainer}>
            <Text style={styles.testTabTitle}>Embedding Tests</Text>
            
            <TextInput
              style={styles.testInput}
              value={input}
              onChangeText={setInput}
              placeholder="Enter text to embed..."
              multiline
            />
            
            <View style={styles.testButtonContainer}>
              <Button 
                title="Generate Embedding" 
                onPress={testEmbedding} 
                disabled={!input.trim()}
              />
            </View>
          </View>
        ) : (
          <>
            {/* Add the tool mode controls */}
            {renderToolModeControls()}

            <View style={styles.safetyCheckContainer}>
              <TouchableOpacity
                style={[styles.safetyCheckButton, generating && styles.modeButtonDisabled]}
                onPress={runNativeSafetyChecks}
                disabled={generating}
              >
                <Text style={styles.safetyCheckButtonText}>Run Native Safety Checks</Text>
              </TouchableOpacity>
            </View>

            {modelState.mode === 'conversation' && (
              <View style={styles.tokenToolsContainer}>
                <TextInput
                  style={styles.testInput}
                  value={input}
                  onChangeText={setInput}
                  placeholder="Enter text to tokenize/detokenize..."
                  multiline
                />
                
                <View style={styles.testButtonContainer}>
                  <Button 
                    title="Tokenize" 
                    onPress={testTokenizer} 
                    disabled={!input.trim()}
                  />
                  <View style={styles.buttonSpacer} />
                  <Button 
                    title="Detokenize" 
                    onPress={testDetokenize} 
                    disabled={!input.trim()}
                  />
                </View>
              </View>
            )}

            <FlatList
              style={styles.messagesContainer}
              ref={listRef}
              contentContainerStyle={styles.messagesContent}
              data={messages.filter(msg => msg.role !== 'system')}
              renderItem={({ item, index }) => renderMessage(item, index)}
              keyExtractor={(item, index) => item.id ?? `${item.role}-${index}`}
              keyboardShouldPersistTaps="handled"
              ListFooterComponent={
                <View>
                  {!!streamingText && (
                    <View style={[styles.messageWrapper, styles.assistantMessage]}>
                      <Text style={styles.messageSender}>Assistant</Text>
                      <Text style={styles.messageContent}>{streamingText}</Text>
                    </View>
                  )}
                  {generating && !streamingText && (
                    <View style={[styles.messageWrapper, styles.assistantMessage]}>
                      <ActivityIndicator size="small" color="#333" />
                    </View>
                  )}
                </View>
              }
            />
            
            <View style={styles.inputContainer}>
              <TextInput
                style={styles.input}
                value={input}
                onChangeText={setInput}
                placeholder="Type a message..."
                multiline
                maxLength={500}
                editable={!generating}
              />
              <TouchableOpacity 
                style={[
                  styles.sendButton,
                  (!input.trim() || generating) && styles.sendButtonDisabled
                ]}
                onPress={sendMessage}
                disabled={!input.trim() || generating}
              >
                <Text style={styles.sendButtonText}>Send</Text>
              </TouchableOpacity>
            </View>
          </>
        )}

        {error && (
          <View style={styles.errorContainer}>
            <Text style={styles.error}>{error}</Text>
          </View>
        )}
      </KeyboardAvoidingView>
    );
  };

  return <View style={styles.container}>{renderInterface()}</View>;
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    paddingTop: 60,
    paddingBottom: 60,
    backgroundColor: '#f8f9fa',
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 16,
    textAlign: 'center',
  },
  initContainer: {
    alignItems: 'center',
    paddingVertical: 20,
  },
  loader: {
    marginTop: 16,
  },
  error: {
    color: 'red',
    marginTop: 16,
    textAlign: 'center',
  },
  errorContainer: {
    marginTop: 16,
    padding: 12,
    backgroundColor: '#ffeeee',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#ffcccc',
  },
  chatContainer: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 10,
    marginBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#dee2e6',
  },
  modelName: {
    fontWeight: 'bold',
    fontSize: 16,
    flex: 1,
  },
  unloadButton: {
    backgroundColor: '#dc3545',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 4,
  },
  unloadButtonText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 12,
  },
  messagesContainer: {
    flex: 1,
    marginBottom: 10,
  },
  messagesContent: {
    paddingVertical: 10,
  },
  userMessage: {
    alignSelf: 'flex-end',
    backgroundColor: '#007bff',
  },
  assistantMessage: {
    alignSelf: 'flex-start',
    backgroundColor: '#e9ecef',
  },
  inputContainer: {
    flexDirection: 'row',
    marginTop: 10,
    marginBottom: 10,
    paddingHorizontal: 5,
  },
  input: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#ced4da',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 8,
    maxHeight: 100,
    backgroundColor: 'white',
  },
  sendButton: {
    backgroundColor: '#007bff',
    borderRadius: 20,
    width: 60,
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: 8,
  },
  sendButtonDisabled: {
    backgroundColor: '#6c757d',
    opacity: 0.7,
  },
  sendButtonText: {
    color: 'white',
    fontWeight: 'bold',
  },
  loaderContainer: {
    marginTop: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
  loadingText: {
    marginTop: 8,
    fontSize: 16,
    color: '#333',
  },
  modeButtons: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    width: '100%',
    marginTop: 20,
  },
  modeButton: {
    backgroundColor: '#007bff',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 8,
    minWidth: 120,
    alignItems: 'center',
  },
  modeButtonText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 16,
  },
  modeButtonDisabled: {
    backgroundColor: '#6c757d',
    opacity: 0.7,
  },
  testTabContainer: {
    flex: 1,
    padding: 16,
  },
  testTabTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  testInput: {
    borderWidth: 1,
    borderColor: '#ced4da',
    borderRadius: 8,
    padding: 12,
    marginBottom: 16,
    minHeight: 100,
    backgroundColor: 'white',
  },
  testButtonContainer: {
    marginBottom: 16,
  },
  tokenToolsContainer: {
    padding: 16,
    backgroundColor: '#f8f9fa',
    borderBottomWidth: 1,
    borderBottomColor: '#dee2e6',
  },
  safetyCheckContainer: {
    paddingHorizontal: 16,
    paddingBottom: 10,
  },
  safetyCheckButton: {
    backgroundColor: '#198754',
    borderRadius: 8,
    paddingVertical: 10,
    alignItems: 'center',
  },
  safetyCheckButtonText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 13,
  },
  toolModeContainer: {
    marginVertical: 10,
    padding: 12,
    backgroundColor: '#e9ecef',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#dee2e6',
    flexDirection: 'column',
    alignItems: 'flex-start',
  },
  toolModeText: {
    color: '#212529',
    fontSize: 14,
    marginBottom: 10,
    width: '100%',
  },
  toolModeHighlight: {
    fontWeight: 'bold',
  },
  toolButtonsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    width: '100%',
    marginTop: 5,
  },
  exampleButton: {
    backgroundColor: '#007bff',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 4,
    marginRight: 8,
    marginBottom: 6,
  },
  exampleButtonText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 11,
  },
  buttonSpacer: {
    height: 8,
  },
  messageWrapper: {
    maxWidth: '95%',
    borderRadius: 12,
    padding: 12,
    marginVertical: 4,
    alignSelf: 'center',
  },
  messageSender: {
    fontWeight: 'bold',
    color: '#007bff',
  },
  messageContent: {
    color: '#212529',
    fontSize: 16,
    flexWrap: 'wrap',
  },
  toolCallMessage: {
    alignSelf: 'center',
    backgroundColor: '#FFEB99',
    borderColor: '#FFD700',
    borderWidth: 1,
  },
  toolResponseMessage: {
    alignSelf: 'center',
    backgroundColor: '#E8F5E8',
    borderColor: '#4CAF50',
    borderWidth: 1,
  }
}); 