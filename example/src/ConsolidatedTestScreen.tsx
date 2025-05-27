import React, { useState, useCallback } from 'react';
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
} from 'react-native';
import { loadLlamaModelInfo } from '@novastera-oss/llamarn';
import type { LlamaModel } from '@novastera-oss/llamarn';
import RNFS from 'react-native-fs';

const { AssetCheckModule } = NativeModules;

// Use smaller model for Android to avoid build size issues
const modelFileName = Platform.OS === 'android' 
  ? "Llama-3.2-1B-Instruct-Q4_K_M.gguf"  // 770MB - smaller for Android
  : "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"; // 4.1GB - full model for iOS

export default function ConsolidatedTestScreen() {
  const [llamaContext, setLlamaContext] = useState<LlamaModel | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('');
  const [completionResult, setCompletionResult] = useState<string | null>(null);
  const [prompt, setPrompt] = useState('This is a test prompt.');
  const [error, setError] = useState<string | null>(null);
  const [fileCheckResult, setFileCheckResult] = useState<string | null>(null);
  const [bundleFiles, setBundleFiles] = useState<string[]>([]);

  const checkFileExists = useCallback(async () => {
    setError(null);
    setFileCheckResult(null);
    setBundleFiles([]);
    setIsLoading(true);
    setLoadingMessage('Checking file existence...');
    
    try {
      let modelPath = modelFileName;
      let bundlePath = '';
      
      if (Platform.OS === 'ios') {
        bundlePath = RNFS.MainBundlePath;
        modelPath = RNFS.MainBundlePath + '/' + modelFileName;
        console.log(`[iOS] Bundle path: ${bundlePath}`);
        console.log(`[iOS] Full model path: ${modelPath}`);
      } else if (Platform.OS === 'android') {
        // For Android, we can't directly check asset existence with RNFS
        // Assets are bundled and not accessible through regular file system
        modelPath = `file:///android_asset/${modelFileName}`;
        console.log(`[Android] Android asset path: ${modelPath}`);
        console.log(`[Android] Note: Assets cannot be checked with RNFS, will test via native loading`);
      }

      let fileExists = false;
      if (Platform.OS === 'ios') {
        // Check if the specific model file exists (iOS only)
        fileExists = await RNFS.exists(modelPath);
        console.log(`File exists check for ${modelPath}: ${fileExists}`);
      } else {
        // For Android, use our custom native module to check assets
        try {
          fileExists = await AssetCheckModule.doesAssetExist(modelFileName);
          console.log(`[Android] Asset exists check for ${modelFileName}: ${fileExists}`);
        } catch (error) {
          console.error(`[Android] Error checking asset: ${error}`);
          fileExists = false;
        }
      }
      
      let resultMessage = `File: ${modelFileName}\nPath: ${modelPath}\nExists: ${fileExists ? 'YES (or assumed for Android)' : 'NO'}`;

      // Try to list files in the bundle directory (iOS only)
      try {
        if (Platform.OS === 'ios') {
          const files = await RNFS.readDir(bundlePath);
          const fileNames = files.map(file => file.name);
          setBundleFiles(fileNames);
          resultMessage += `\n\nBundle contains ${files.length} files:\n${fileNames.join(', ')}`;
        } else if (Platform.OS === 'android') {
          // For Android, use our custom native module to list assets
          try {
            const assetList = await AssetCheckModule.listAssets('');
            setBundleFiles(assetList);
            resultMessage += `\n\nAndroid assets (${assetList.length} files):\n${assetList.join(', ')}`;
          } catch (error) {
            console.error('Error listing Android assets:', error);
            resultMessage += `\n\nCould not list Android assets: ${error}`;
          }
        }
      } catch (listError) {
        console.log('Could not list bundle files:', listError);
        resultMessage += '\n\nCould not list bundle contents.';
      }

      setFileCheckResult(resultMessage);
      
    } catch (e) {
      const err = e as Error;
      console.error("Error checking file:", err);
      setError(`File check failed: ${err.message}`);
    }
    setIsLoading(false);
  }, []);

  const handleLoadModel = useCallback(async () => {
    setError(null);
    setCompletionResult(null);
    setLlamaContext(null);
    setIsLoading(true);
    setLoadingMessage(`Loading model: ${modelFileName}...`);
    try {
      let modelPath = modelFileName;
      if (Platform.OS === 'ios') {
        modelPath = RNFS.MainBundlePath + '/' + modelFileName;
        console.log(`[iOS] Using full model path: ${modelPath}`);
      } else if (Platform.OS === 'android') {
        // For Android, try different asset path formats
        // Different native libraries expect different formats
        modelPath = `file:///android_asset/${modelFileName}`;
        console.log(`[Android] Using android_asset path: ${modelPath}`);
      }

      console.log(`Attempting to load model using path: ${modelPath}`);
      
      let context;
            if (Platform.OS === 'android') {
        // For Android, we need to copy the asset to a temporary location
        // because the native C++ code can't directly access Android assets
        const tempDir = RNFS.CachesDirectoryPath;
        const tempModelPath = `${tempDir}/${modelFileName}`;
        
        console.log(`[Android] Copying asset to temp location: ${tempModelPath}`);
        
        try {
          // First check if we already have the file in cache
          const tempFileExists = await RNFS.exists(tempModelPath);
          
          if (!tempFileExists) {
            console.log(`[Android] Temp file doesn't exist, copying from assets...`);
            // Copy from assets to cache directory
            await RNFS.copyFileAssets(modelFileName, tempModelPath);
            console.log(`[Android] Successfully copied asset to: ${tempModelPath}`);
          } else {
            console.log(`[Android] Using existing temp file: ${tempModelPath}`);
          }
          
          // Now try to load model info from the temp file
          context = await loadLlamaModelInfo(tempModelPath);
          console.log(`[Android] Success with temp file path: ${tempModelPath}`);
          
        } catch (copyError) {
          console.error(`[Android] Failed to copy asset or load model:`, copyError);
          throw new Error(`Failed to access Android asset: ${(copyError as Error).message}`);
        }
      } else {
        // iOS - use the single path
        context = await loadLlamaModelInfo(modelPath);
      }
      
      console.log("Model info loaded successfully:", context);
      if (context === undefined) {
        console.error("CRITICAL: loadLlamaModelInfo resolved with undefined!");
        throw new Error("loadLlamaModelInfo resolved with undefined. Native module issue.");
      }
      setLoadingMessage('Model info loaded successfully!');
      // Just display the info, don't set as context since this is just info loading
      console.log("Model info details:", JSON.stringify(context, null, 2));
    } catch (e) {
      const err = e as Error;
      console.error("Error loading Llama model:", err);
      setError(`Failed to load model: ${err.message}`);
      setLlamaContext(null);
    }
    setIsLoading(false);
  }, []);

  const handleSimpleCompletion = useCallback(async () => {
    if (!llamaContext) {
      setError("Model context not loaded. Load the model first.");
      return;
    }
    setError(null);
    setCompletionResult(null);
    setIsLoading(true);
    setLoadingMessage('Performing completion...');
    try {
      console.log(`Attempting completion with prompt: "${prompt}"`);
      const result = await llamaContext.completion({
        prompt: prompt,
        n_predict: 30,
        temperature: 0.2,
      });
      console.log("Completion successful:", result);
      let responseText = "No text in result";
      if (result.text) {
        responseText = result.text;
      } else if (result.choices?.[0]?.message?.content) {
        responseText = result.choices[0].message.content;
      }
      setCompletionResult(responseText);
      setLoadingMessage('Completion finished.');
    } catch (e) {
      const err = e as Error;
      console.error("Error during completion:", err);
      setError(`Completion failed: ${err.message}`);
    }
    setIsLoading(false);
  }, [llamaContext, prompt]);

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.title}>Consolidated Model Tests</Text>

        <View style={styles.sectionContainer}>
          <Button title="Check File Exists" onPress={checkFileExists} disabled={isLoading} />
        </View>

        <View style={styles.sectionContainer}>
          <Button title={`Get Model Info (${modelFileName})`} onPress={handleLoadModel} disabled={isLoading} />
        </View>

        {isLoading && (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color="#007AFF" />
            <Text style={styles.loadingText}>{loadingMessage}</Text>
          </View>
        )}

        {error && (
          <View style={styles.errorContainer}>
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}

        {fileCheckResult && !isLoading && (
          <View style={styles.resultsContainer}>
            <Text style={styles.subTitle}>File Check Result:</Text>
            <Text style={styles.resultText}>{fileCheckResult}</Text>
          </View>
        )}
        
        {llamaContext && !isLoading && (
          <View style={styles.sectionContainer}>
            <Text style={styles.subTitle}>Test Completion</Text>
            <TextInput
              style={styles.input}
              value={prompt}
              onChangeText={setPrompt}
              placeholder="Enter prompt here"
              multiline
            />
            <Button title="Generate Completion" onPress={handleSimpleCompletion} disabled={isLoading} />
          </View>
        )}

        {completionResult && !isLoading && (
          <View style={styles.resultsContainer}>
            <Text style={styles.subTitle}>Completion Result:</Text>
            <Text style={styles.resultText}>{completionResult}</Text>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#f8f8f8',
  },
  container: {
    flexGrow: 1,
    padding: 20,
    alignItems: 'center',
  },
  title: {
    fontSize: 22,
    fontWeight: 'bold',
    marginBottom: 20,
    textAlign: 'center',
  },
  subTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 10,
    textAlign: 'center',
  },
  sectionContainer: {
    width: '100%',
    padding: 15,
    backgroundColor: '#ffffff',
    borderRadius: 8,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 1.41,
    elevation: 2,
  },
  loadingContainer: {
    alignItems: 'center',
    marginVertical: 20,
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#333',
  },
  errorContainer: {
    marginVertical: 15,
    padding: 10,
    backgroundColor: '#ffebee',
    borderRadius: 8,
    width: '100%',
  },
  errorText: {
    color: '#c62828',
    fontSize: 16,
    textAlign: 'center',
  },
  input: {
    borderWidth: 1,
    borderColor: '#ddd',
    padding: 10,
    fontSize: 16,
    borderRadius: 6,
    marginBottom: 15,
    minHeight: 80, // For multiline
    textAlignVertical: 'top',
  },
  resultsContainer: {
    width: '100%',
    padding: 15,
    backgroundColor: '#e3f2fd',
    borderRadius: 8,
    marginTop: 10,
  },
  resultText: {
    fontSize: 16,
    color: '#0d47a1',
  },
}); 