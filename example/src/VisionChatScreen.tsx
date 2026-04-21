import { useState } from 'react';
import {
  View,
  Text,
  Button,
  TextInput,
  ScrollView,
  Image,
  StyleSheet,
  ActivityIndicator,
  Platform,
} from 'react-native';
import RNFS from 'react-native-fs';
import { initLlama } from '@novastera-oss/llamarn';
import type { LlamaModel } from '@novastera-oss/llamarn';

const MODEL_FILENAME  = Platform.OS === 'android' ? 'Qwen3.5-0.8B-Q4_K_M.gguf' : 'Qwen3.5-2B-Q4_K_M.gguf';
const MMPROJ_FILENAME = 'mmproj-F16.gguf';

function getDefaultModelPath(): string {
  if (Platform.OS === 'ios') return `${RNFS.MainBundlePath}/${MODEL_FILENAME}`;
  return `${RNFS.CachesDirectoryPath}/${MODEL_FILENAME}`;
}

function getDefaultMmprojPath(): string {
  if (Platform.OS === 'ios') return `${RNFS.MainBundlePath}/${MMPROJ_FILENAME}`;
  return `${RNFS.CachesDirectoryPath}/${MMPROJ_FILENAME}`;
}

export default function VisionChatScreen() {
  const [model, setModel]         = useState<LlamaModel | null>(null);
  const [modelPath, setModelPath] = useState(getDefaultModelPath());
  const [mmprojPath, setMmprojPath] = useState(getDefaultMmprojPath());
  const [imagePath, setImage]     = useState('');
  const [response, setResp]       = useState('');
  const [loading, setLoading]     = useState(false);
  const [status, setStatus]       = useState('');

  async function ensureModel(): Promise<LlamaModel> {
    if (model) return model;
    const mp = modelPath.trim();
    if (!mp) throw new Error('Enter a model path first');
    setStatus('Loading model…');

    // Android: copy assets to cache on first run (same pattern as ModelChatTestScreen)
    let resolvedMmproj = mmprojPath.trim();
    if (Platform.OS === 'android' && resolvedMmproj) {
      const cached = `${RNFS.CachesDirectoryPath}/${MMPROJ_FILENAME}`;
      if (!(await RNFS.exists(cached))) {
        setStatus('Copying mmproj to cache…');
        await RNFS.copyFileAssets(MMPROJ_FILENAME, cached);
      }
      resolvedMmproj = cached;
    }

    // Vision models need larger context: image tokens alone can be 256–784 for a 448px image.
    // n_gpu_layers: 0 ensures CPU fallback on iOS simulator where Metal embedding-input
    // compute paths may not be fully supported.
    const params: Parameters<typeof initLlama>[0] = {
      model: mp,
      n_ctx: 4096,
      use_jinja: true,
      n_gpu_layers: 0,
    };
    const mmproj = resolvedMmproj;
    if (mmproj) {
      params.mmproj       = mmproj;
      params.capabilities = ['vision-chat', 'image-encode'];
    }
    const ctx = await initLlama(params);
    setModel(ctx);
    if (mmproj) {
      const mods = await ctx.getSupportedModalities();
      setStatus(`Model ready — vision: ${mods.vision}`);
    } else {
      setStatus('Model ready (text-only, no mmproj)');
    }
    return ctx;
  }

  async function useTestImage() {
    try {
      const dest = `${RNFS.CachesDirectoryPath}/dataset.png`;
      if (Platform.OS === 'android') {
        await RNFS.copyFileAssets('dataset.png', dest);
      } else {
        const { uri } = Image.resolveAssetSource(require('../assets/dataset.png'));
        if (uri.startsWith('http')) {
          await RNFS.downloadFile({ fromUrl: uri, toFile: dest }).promise;
        } else {
          await RNFS.copyFile(uri.replace('file://', ''), dest);
        }
      }
      setImage(`file://${dest}`);
      setStatus('Test image ready');
    } catch (e: any) {
      setStatus(`Failed to load test image: ${e.message}`);
    }
  }

  async function describe() {
    if (!imagePath.trim()) { setStatus('Load an image first'); return; }
    setLoading(true);
    setResp('');
    try {
      const ctx = await ensureModel();
      setStatus('Running…');
      const r = await ctx.completion({
        messages: [{
          role: 'user',
          content: [
            { type: 'text', text: 'Describe this image in detail.' },
            { type: 'image_url', image_url: { url: imagePath.trim() } },
          ],
        }],
        n_predict: 256,
      });
      // chat completion returns OpenAI format; fall through candidates in order
      const text: string =
        (r as any).text ??
        (r as any).choices?.[0]?.message?.content ??
        JSON.stringify(r);
      setResp(text);
      setStatus('Done');
    } catch (e: any) {
      setStatus(`Error: ${e.message}`);
    } finally {
      setLoading(false);
    }
  }

  async function embedImage() {
    if (!imagePath.trim()) { setStatus('Load an image first'); return; }
    setLoading(true);
    try {
      const ctx = await ensureModel();
      setStatus('Embedding…');
      const r = await (ctx as any).embedImage(imagePath.trim(), { normalize: true });
      setResp(
        `Embedding: ${r.n_tokens} tokens × ${r.n_embd} dims\n` +
        `First 5: ${(r.embedding as number[]).slice(0, 5).map((v) => v.toFixed(4)).join(', ')}…`
      );
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

      <Text style={styles.label}>Model path</Text>
      <TextInput
        style={styles.input}
        value={modelPath}
        onChangeText={setModelPath}
        autoCapitalize="none"
        autoCorrect={false}
        editable={!model && !loading}
      />

      <Text style={styles.label}>Mmproj path (optional — vision models only)</Text>
      <TextInput
        style={styles.input}
        placeholder="leave blank for text-only test"
        value={mmprojPath}
        onChangeText={setMmprojPath}
        autoCapitalize="none"
        autoCorrect={false}
        editable={!model && !loading}
      />

      <Text style={model ? styles.badgeOk : styles.badgeOff}>
        {model ? '● Model loaded' : '○ Model not loaded — loads automatically on first use'}
      </Text>

      <View style={styles.row}>
        <Button title="Pre-load Model" onPress={async () => { setLoading(true); try { await ensureModel(); } catch(e:any) { setStatus(`Error: ${e.message}`); } finally { setLoading(false); }}} disabled={loading || !!model} />
        <Button title="Release"        onPress={releaseModel} disabled={!model || loading} />
      </View>

      {!!status && <Text style={styles.status}>{status}</Text>}
      {loading && <ActivityIndicator style={styles.spinner} />}

      <View style={styles.row}>
        <Button title="Use Test Image" onPress={useTestImage} disabled={loading} />
      </View>

      <TextInput
        style={styles.input}
        placeholder="or paste file:///path/to/image.jpg"
        value={imagePath}
        onChangeText={setImage}
        autoCapitalize="none"
        autoCorrect={false}
      />

      {!!imagePath && (
        <Image source={{ uri: imagePath }} style={styles.preview} resizeMode="contain" />
      )}

      <View style={styles.row}>
        <Button title="Describe" onPress={describe} disabled={!imagePath.trim() || loading} />
        <Button title="Embed"    onPress={embedImage} disabled={!imagePath.trim() || loading} />
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
  container:    { padding: 16, flexGrow: 1 },
  title:        { fontSize: 20, fontWeight: 'bold', marginBottom: 12 },
  label:        { fontSize: 12, color: '#666', marginBottom: 2 },
  row:          { flexDirection: 'row', gap: 8, flexWrap: 'wrap', marginBottom: 8 },
  status:       { fontSize: 13, color: '#555', marginBottom: 4 },
  badgeOk:      { fontSize: 12, color: '#2a8a2a', fontWeight: '600', marginBottom: 6 },
  badgeOff:     { fontSize: 12, color: '#999',    marginBottom: 6 },
  spinner:      { marginVertical: 8 },
  input:        { borderWidth: 1, borderColor: '#ccc', borderRadius: 6, padding: 8, marginBottom: 10, fontSize: 11 },
  preview:      { width: '100%', height: 180, marginBottom: 10, borderRadius: 6, backgroundColor: '#eee' },
  responseBox:  { marginTop: 8, padding: 8, backgroundColor: '#f5f5f5', borderRadius: 6 },
  responseText: { fontSize: 13, lineHeight: 20 },
});
