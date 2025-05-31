import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  StyleSheet,
  View,
  Text,
  Button,
  ScrollView,
  ActivityIndicator,
  TextInput,
  SafeAreaView,
  Platform,
  NativeModules,
  TouchableOpacity,
  KeyboardAvoidingView,
} from 'react-native';
import { initLlama, loadLlamaModelInfo } from '@novastera-oss/llamarn';
import type { LlamaModel, LlamaMessage, LlamaTool } from '@novastera-oss/llamarn';
import RNFS from 'react-native-fs';

const { AssetCheckModule } = NativeModules;

// Use smaller model for Android to avoid build size issues
const modelFileName = Platform.OS === 'android' 
  ? "Llama-3.2-1B-Instruct-Q4_K_M.gguf"  // 770MB - smaller for Android
  : "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"; // 4.1GB - full model for iOS

interface Message {
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string;
  name?: string;
  tool_call_id?: string;
  isToolCall?: boolean; // Simple flag to indicate if this is a tool call message
}

type ModelMode = 'conversation' | 'tools' | 'embeddings';

interface ModelState {
  instance: LlamaModel;
  info: any;
  mode: ModelMode;
}

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
        mode
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
    { role: 'system', content: 'You are a helpful AI assistant.' }
  ]);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [streamingTokens, setStreamingTokens] = useState<string[]>([]);
  
  const scrollViewRef = useRef<ScrollView>(null);

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
        
        // Create tool message
        const toolMessage: Message = {
          role: 'tool',
          content: `🌤️ Weather Result:\n${formatToolResponse(JSON.stringify(weatherData))}`,
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
        
        // Create tool message
        const toolMessage: Message = {
          role: 'tool',
          content: `📍 Location Result:\n${formatToolResponse(JSON.stringify(locationData))}`,
          name: functionName,
          tool_call_id: toolCall.id
        };
        
        return toolMessage;
      }
      
      throw new Error(`Unknown tool: ${functionName}`);
    } catch (err) {
      console.error('Error handling tool call:', err);
      const errorMessage: Message = {
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
        role: 'system',
        content: 
`You are a helpful AI assistant. When asked for specific information like weather or location, you MUST use your available tools to find it. Respond with the information found by the tool.
You must use the tool get_weather each time someone ask you about the weather. YOU WILL ALWAYS HAVE ACCESS TO THE get_weather tools`
      }];
    } else {
      // Default system message
      initialMessages = [{
        role: 'system',
        content: 'You are a helpful AI assistant.'
      }];
    }
    
    setMessages(initialMessages);
    setModelState(state);
  };

  // Handle model unload
  const handleUnload = async () => {
    if (modelState?.instance) {
      try {
        await modelState.instance.release();
        console.log('Model unloaded successfully');
        setModelState(null);
        setMessages([{ role: 'system', content: 'You are a helpful AI assistant.' }]);
      } catch (err) {
        console.error('Failed to unload model:', err);
        setError(`Release Error: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  };

  // Add streaming token for UI display
  const handleStreamingToken = (token: string) => {
    console.log('Token received:', token);
    setStreamingTokens(prev => [...prev, token]);
  };

  // Send a message to the model
  const sendMessage = async () => {
    if (!modelState?.instance || !input.trim()) return;
    
    const userMessage: Message = { role: 'user', content: input.trim() };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setGenerating(true);
    setError(null);
    setStreamingTokens([]); // Reset streaming tokens for each new message
    
    try {
      // Get current messages including the new user message
      const currentMessages = [...messages, userMessage];
      
      const completionOptions: any = {
        messages: currentMessages.map(msg => ({
          role: msg.role,
          content: msg.content,
          ...(msg.name && { name: msg.name }),
          ...(msg.tool_call_id && { tool_call_id: msg.tool_call_id })
        })) as LlamaMessage[],
        temperature: 0.3,
        top_p: 0.85,
        top_k: 40,
        max_tokens: 400,
        stop: ["</s>", "<|im_end|>", "<|eot_id|>"],
      };
      
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
      
             // Extract the assistant's response content
       let responseContent = response.text; // Direct text property
       
       // If not available, try to get from choices array (standard OpenAI format)
       if (!responseContent && response.choices && response.choices.length > 0) {
         responseContent = response.choices[0]?.message?.content || '';
       }
       
       // Add the assistant response to the messages
       const assistantMessage: Message = { 
         role: 'assistant', 
         content: responseContent || 'Sorry, I couldn\'t generate a response.' 
       };
       
       // Check if there are tool calls to process
       let toolCalls: any[] = [];
       
       // Try to find tool calls in different possible locations
       if (response.tool_calls && Array.isArray(response.tool_calls) && response.tool_calls.length > 0) {
         toolCalls = response.tool_calls;
         console.log('Found tool_calls at top level:', toolCalls);
       } else if (response.choices && response.choices.length > 0) {
         const choice = response.choices[0];
         
         // Check if finish_reason indicates tool call
         if (choice && choice.finish_reason === 'tool_calls') {
           console.log('Finish reason indicates tool calls');
           
           // Try to find tool_calls in the message object
           if (choice.message && Array.isArray(choice.message.tool_calls) && choice.message.tool_calls.length > 0) {
             toolCalls = choice.message.tool_calls;
             console.log('Found tool_calls in choices[0].message.tool_calls');
           }
         }
       }
      
      console.log('Tool calls extracted:', toolCalls);
      
      // Process tool calls if found
      if (toolCalls.length > 0) {
        // Add assistant message with the tool call request
        setMessages(prev => [...prev, assistantMessage]);
        
        // Add tool call messages to show what tools are being called
        const toolCallMessages: Message[] = toolCalls.map((toolCall, index) => ({
          role: 'assistant' as const,
          content: `🔧 Calling tool: ${toolCall.function?.name}\nArguments: ${toolCall.function?.arguments || '{}'}`,
          isToolCall: true
        }));
        
        setMessages(prev => [...prev, ...toolCallMessages]);
        
        // Process all tool calls sequentially
        const toolMessages: Message[] = [];
        
        for (const toolCall of toolCalls) {
          const toolResponse = await handleToolCall(toolCall);
          toolMessages.push(toolResponse);
        }
        
        // Add all tool responses to messages
        setMessages(prev => [...prev, ...toolMessages]);
        
        // Now get a follow-up response with the tool results included
        const allMessages = [
          ...currentMessages,
          assistantMessage,
          ...toolMessages
        ];
        
        // Make a second completion call with the tool results
        const finalResponse = await modelState.instance.completion({
          ...completionOptions,
          messages: allMessages.map(msg => ({
            role: msg.role,
            content: msg.content,
            ...(msg.name && { name: msg.name }),
            ...(msg.tool_call_id && { tool_call_id: msg.tool_call_id })
          })) as LlamaMessage[],
          // Disable tools for the final response to prevent infinite loops
          tools: undefined,
          tool_choice: undefined
        },
        (data: { token: string }) => {
          handleStreamingToken(data.token);
        });
        
        console.log('Final response after tool call:', finalResponse);
        
                 // Extract text from either text property or choices array
         let finalText = finalResponse.text;
         if (!finalText && finalResponse.choices && finalResponse.choices.length > 0) {
           finalText = finalResponse.choices[0]?.message?.content || '';
         }
         
         // Add the final assistant response
         const finalMessage: Message = { 
           role: 'assistant', 
           content: finalText || 'I couldn\'t process the tool response.'
         };
        setMessages(prev => [...prev, finalMessage]);
      } else {
        // No tool calls - just add the assistant message
        setMessages(prev => [...prev, assistantMessage]);
      }
    } catch (err) {
      console.error('Completion error:', err);
      setError(`Error generating response: ${err instanceof Error ? err.message : String(err)}`);
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

  // Scroll to bottom of messages
  useEffect(() => {
    setTimeout(() => {
      scrollViewRef.current?.scrollToEnd({ animated: true });
    }, 100);
  }, [messages]);

  // When in tools mode, show example question and debug buttons
  const renderToolModeControls = () => {
    if (modelState?.mode !== 'tools') return null;
    
    return (
      <View style={styles.toolModeContainer}>
        <Text style={styles.toolModeText}>
          <Text style={styles.toolModeHighlight}>Tool Mode Active:</Text> Ask about the weather
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
        <Text style={styles.messageContent}>{msg.content}</Text>
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

            <ScrollView 
              style={styles.messagesContainer}
              ref={scrollViewRef}
              contentContainerStyle={styles.messagesContent}
            >
              {messages.filter(msg => msg.role !== 'system').map((msg, index) => renderMessage(msg, index))}
              {streamingTokens.length > 0 && (
                <View style={styles.streamingContainer}>
                  <Text style={styles.streamingTitle}>Latest tokens:</Text>
                  <Text style={styles.streamingTokens}>
                    {streamingTokens.map((token, i) => (
                      <Text key={i} style={styles.streamingToken}>
                        {token.replace(/\n/g, '⏎')}
                      </Text>
                    )).reverse()}
                  </Text>
                </View>
              )}
              {generating && (
                <View style={[styles.messageWrapper, styles.assistantMessage]}>
                  <ActivityIndicator size="small" color="#333" />
                </View>
              )}
            </ScrollView>
            
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

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.container}>
        {renderInterface()}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  container: {
    flex: 1,
    padding: 16,
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
  streamingContainer: {
    marginTop: 10,
    padding: 10,
    backgroundColor: '#e9ecef',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#dee2e6',
  },
  streamingTitle: {
    fontWeight: 'bold',
    marginBottom: 8,
  },
  streamingTokens: {
    color: '#212529',
    fontSize: 14,
  },
  streamingToken: {
    marginBottom: 4,
  },
  tokenToolsContainer: {
    padding: 16,
    backgroundColor: '#f8f9fa',
    borderBottomWidth: 1,
    borderBottomColor: '#dee2e6',
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