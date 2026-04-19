import React, { useState } from 'react';
import {
  View,
  Text,
  Button,
  TextInput,
  ScrollView,
  StyleSheet,
  ActivityIndicator,
} from 'react-native';
import { initLlama } from '@novastera-oss/llamarn';
import type { LlamaModel } from '@novastera-oss/llamarn';

// Update these paths to your local model files
const MODEL_FILE  = '/path/to/llava-v1.5-7b-q4_k.gguf';
const MMPROJ_FILE = '/path/to/mmproj-model-f16.gguf';

export default function VisionChatScreen() {
  const [model, setModel]     = useState<LlamaModel | null>(null);
  const [imagePath, setImage] = useState('');
  const [response, setResp]   = useState('');
  const [loading, setLoading] = useState(false);
  const [status, setStatus]   = useState('');

  async function loadModel() {
    setLoading(true);
    setStatus('Loading model...');
    try {
      const ctx = await initLlama({
        model:        MODEL_FILE,
        mmproj:       MMPROJ_FILE,
        n_ctx:        2048,
        use_jinja:    true,
        capabilities: ['vision-chat'],
      });
      setModel(ctx);
      const mods = await ctx.getSupportedModalities();
      setStatus(`Ready — vision: ${mods.vision}, audio: ${mods.audio}`);
    } catch (e: any) {
      setStatus(`Error loading: ${e.message}`);
    } finally {
      setLoading(false);
    }
  }

  async function describe() {
    if (!model || !imagePath.trim()) return;
    setLoading(true);
    setResp('');
    setStatus('Running...');
    try {
      const r = await model.completion({
        messages: [{
          role: 'user',
          content: [
            { type: 'text', text: 'Describe this image in detail.' },
            { type: 'image_url', image_url: { url: imagePath.trim() } },
          ],
        }],
        n_predict: 256,
      });
      setResp(r.text ?? JSON.stringify(r));
      setStatus('Done');
    } catch (e: any) {
      setStatus(`Error: ${e.message}`);
    } finally {
      setLoading(false);
    }
  }

  async function embedImage() {
    if (!model || !imagePath.trim()) return;
    setLoading(true);
    setStatus('Embedding...');
    try {
      const r = await (model as any).embedImage(imagePath.trim(), { normalize: true });
      setResp(`Embedding: ${r.n_tokens} tokens × ${r.n_embd} dims\nFirst 5: ${r.embedding.slice(0, 5).map((v: number) => v.toFixed(4)).join(', ')}...`);
      setStatus('Done');
    } catch (e: any) {
      setStatus(`Error: ${e.message}`);
    } finally {
      setLoading(false);
    }
  }

  async function releaseModel() {
    if (!model) return;
    await model.release();
    setModel(null);
    setStatus('Released');
  }

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Vision Chat Demo</Text>

      <View style={styles.row}>
        <Button title="Load Model" onPress={loadModel} disabled={loading || !!model} />
        <Button title="Release" onPress={releaseModel} disabled={!model || loading} />
      </View>

      {!!status && <Text style={styles.status}>{status}</Text>}
      {loading && <ActivityIndicator style={styles.spinner} />}

      <TextInput
        style={styles.input}
        placeholder="file:///path/to/image.jpg or data:image/...;base64,..."
        value={imagePath}
        onChangeText={setImage}
        autoCapitalize="none"
        autoCorrect={false}
      />

      <View style={styles.row}>
        <Button title="Describe" onPress={describe} disabled={!model || !imagePath.trim() || loading} />
        <Button title="Embed" onPress={embedImage} disabled={!model || !imagePath.trim() || loading} />
      </View>

      {!!response && (
        <View style={styles.responseBox}>
          <Text style={styles.responseText}>{response}</Text>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container:   { padding: 16, flexGrow: 1 },
  title:       { fontSize: 20, fontWeight: 'bold', marginBottom: 12 },
  row:         { flexDirection: 'row', gap: 8, flexWrap: 'wrap', marginBottom: 8 },
  status:      { fontSize: 13, color: '#555', marginBottom: 4 },
  spinner:     { marginVertical: 8 },
  input:       { borderWidth: 1, borderColor: '#ccc', borderRadius: 6, padding: 8, marginBottom: 10, fontSize: 12 },
  responseBox: { marginTop: 8, padding: 8, backgroundColor: '#f5f5f5', borderRadius: 6 },
  responseText:{ fontSize: 13, lineHeight: 20 },
});
