import React, { useRef, useState } from 'react';
import {
  SafeAreaView,
  ScrollView,
  Text,
  Button,
  ActivityIndicator,
  StyleSheet,
  PermissionsAndroid,
  Switch,
  View,
} from 'react-native';
import RNFS from 'react-native-fs';
import { initWhisper } from 'whisper.rn';
import { RealtimeTranscriber } from 'whisper.rn/src/realtime-transcription';
import { AudioPcmStreamAdapter } from 'whisper.rn/src/realtime-transcription/adapters/AudioPcmStreamAdapter';

const MODEL_PATH = `${RNFS.ExternalDirectoryPath}/ggml-tiny.en.bin`;
const AUDIO_PATH = `${RNFS.ExternalDirectoryPath}/jfk.wav`;
const REC_PATH = `${RNFS.ExternalDirectoryPath}/recording.wav`;
const SAMPLE_RATE = 16000;
const BYTES_PER_SEC = SAMPLE_RATE * 2; // 16-bit mono
const MAX_SECONDS = 600; // batch recordings stop automatically at 10 minutes

const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';

function toBase64(bytes: Uint8Array): string {
  const parts: string[] = [];
  let chunk = '';
  const len = bytes.length;
  let i = 0;
  for (; i + 2 < len; i += 3) {
    const n = (bytes[i] << 16) | (bytes[i + 1] << 8) | bytes[i + 2];
    chunk += B64[(n >> 18) & 63] + B64[(n >> 12) & 63] + B64[(n >> 6) & 63] + B64[n & 63];
    if (chunk.length >= 8192) {
      parts.push(chunk);
      chunk = '';
    }
  }
  const rem = len - i;
  if (rem === 1) {
    const n = bytes[i] << 16;
    chunk += B64[(n >> 18) & 63] + B64[(n >> 12) & 63] + '==';
  } else if (rem === 2) {
    const n = (bytes[i] << 16) | (bytes[i + 1] << 8);
    chunk += B64[(n >> 18) & 63] + B64[(n >> 12) & 63] + B64[(n >> 6) & 63] + '=';
  }
  parts.push(chunk);
  return parts.join('');
}

function makeWavHeader(dataLen: number): Uint8Array {
  const buf = new ArrayBuffer(44);
  const v = new DataView(buf);
  const w = (o: number, s: string) => {
    for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i));
  };
  w(0, 'RIFF');
  v.setUint32(4, 36 + dataLen, true);
  w(8, 'WAVE');
  w(12, 'fmt ');
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); // PCM
  v.setUint16(22, 1, true); // mono
  v.setUint32(24, SAMPLE_RATE, true);
  v.setUint32(28, BYTES_PER_SEC, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  w(36, 'data');
  v.setUint32(40, dataLen, true);
  return new Uint8Array(buf);
}

function toBytes(d: any): Uint8Array | null {
  const x = d?.data ?? d;
  if (x instanceof Uint8Array) return new Uint8Array(x); // copy
  if (x instanceof ArrayBuffer) return new Uint8Array(x.slice(0));
  if (ArrayBuffer.isView(x)) {
    return new Uint8Array(x.buffer.slice(x.byteOffset, x.byteOffset + x.byteLength));
  }
  if (Array.isArray(x)) return Uint8Array.from(x);
  return null;
}

export default function App() {
  const [status, setStatus] = useState('Ready');
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState(false);
  const [batchMode, setBatchMode] = useState(true);

  const ctxRef = useRef<any>(null);
  const transcriberRef = useRef<any>(null); // live mode
  const adapterRef = useRef<any>(null); // batch mode
  const chunksRef = useRef<Uint8Array[]>([]);
  const bytesRef = useRef(0);
  const lastSecRef = useRef(-1);
  const stoppingRef = useRef(false);
  const activeModeRef = useRef<'live' | 'batch'>('batch');

  const getContext = async () => {
    if (!ctxRef.current) {
      if (!(await RNFS.exists(MODEL_PATH))) {
        throw new Error(`Model not found at ${MODEL_PATH}`);
      }
      setStatus('Loading model...');
      ctxRef.current = await initWhisper({ filePath: MODEL_PATH });
    }
    return ctxRef.current;
  };

  const ensureMic = async () => {
    const perm = await PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.RECORD_AUDIO);
    if (perm !== PermissionsAndroid.RESULTS.GRANTED) {
      setStatus('Microphone permission denied');
      return false;
    }
    return true;
  };

  // ---------- test file ----------
  const transcribeFile = async () => {
    setLoading(true);
    setText('');
    try {
      if (!(await RNFS.exists(AUDIO_PATH))) {
        throw new Error(`Audio not found at ${AUDIO_PATH}`);
      }
      const ctx = await getContext();
      setStatus('Transcribing...');
      const start = Date.now();
      const { promise } = ctx.transcribe(AUDIO_PATH, { language: 'en' });
      const { result } = await promise;
      setText(result.trim());
      setStatus(`Done in ${((Date.now() - start) / 1000).toFixed(1)}s`);
    } catch (e: any) {
      setStatus(`Error: ${e?.message ?? String(e)}`);
    } finally {
      setLoading(false);
    }
  };

  // ---------- live mode ----------
  const startLive = async () => {
    try {
      setText('');
      setLoading(true);
      const ctx = await getContext();
      setLoading(false);
      const transcriber = new RealtimeTranscriber(
        { whisperContext: ctx, audioStream: new AudioPcmStreamAdapter(), fs: RNFS },
        { audioSliceSec: 10, transcribeOptions: { language: 'en' } },
        {
          onTranscribe: (event: any) => {
            const t = event?.data?.result?.trim();
            if (t) setText(prev => (prev ? prev + ' ' + t : t));
          },
          onError: (err: any) => setStatus(`Error: ${err?.message ?? String(err)}`),
        },
      );
      transcriberRef.current = transcriber;
      await transcriber.start();
      setRecording(true);
      setStatus('Live: listening... text appears every ~10 seconds');
    } catch (e: any) {
      setLoading(false);
      setStatus(`Error: ${e?.message ?? String(e)}`);
    }
  };

  const stopLive = async () => {
    try {
      await transcriberRef.current?.stop();
    } catch (e: any) {
      setStatus(`Error: ${e?.message ?? String(e)}`);
    }
    transcriberRef.current = null;
    setRecording(false);
    setStatus('Stopped');
  };

  // ---------- batch mode ----------
  const stopBatch = async () => {
    if (stoppingRef.current) return;
    stoppingRef.current = true;
    setRecording(false);
    setLoading(true);
    try {
      const adapter = adapterRef.current;
      adapterRef.current = null;
      try {
        await adapter?.stop();
        await adapter?.release?.();
      } catch {}

      const total = bytesRef.current;
      const seconds = total / BYTES_PER_SEC;
      if (total < BYTES_PER_SEC / 2) {
        throw new Error('Recording too short (no audio captured)');
      }

      setStatus('Saving audio...');
      const all = new Uint8Array(44 + total);
      all.set(makeWavHeader(total), 0);
      let off = 44;
      for (const c of chunksRef.current) {
        all.set(c, off);
        off += c.length;
      }
      chunksRef.current = [];
      await RNFS.writeFile(REC_PATH, toBase64(all), 'base64');

      const ctx = await getContext();
      setStatus(`Transcribing ${seconds.toFixed(0)}s of audio...`);
      const start = Date.now();
      const { promise } = ctx.transcribe(REC_PATH, { language: 'en' });
      const { result } = await promise;
      setText(result.trim());
      setStatus(
        `Done: ${seconds.toFixed(0)}s of audio transcribed in ${((Date.now() - start) / 1000).toFixed(1)}s`,
      );
    } catch (e: any) {
      setStatus(`Error: ${e?.message ?? String(e)}`);
    } finally {
      setLoading(false);
    }
  };

  const startBatch = async () => {
    try {
      setText('');
      chunksRef.current = [];
      bytesRef.current = 0;
      lastSecRef.current = -1;
      stoppingRef.current = false;

      const adapter: any = new AudioPcmStreamAdapter();
      await adapter.initialize({
        sampleRate: SAMPLE_RATE,
        channels: 1,
        bitsPerSample: 16,
        audioSource: 6,
        bufferSize: 16 * 1024,
      });
      adapter.onData((d: any) => {
        const b = toBytes(d);
        if (!b) {
          setStatus(`Error: unexpected audio data format (${typeof d?.data})`);
          return;
        }
        chunksRef.current.push(b);
        bytesRef.current += b.length;
        const sec = Math.floor(bytesRef.current / BYTES_PER_SEC);
        if (sec !== lastSecRef.current) {
          lastSecRef.current = sec;
          setStatus(`Recording... ${sec}s (tap Stop when done)`);
        }
        if (sec >= MAX_SECONDS) stopBatch();
      });
      adapter.onError?.((err: any) => setStatus(`Error: ${err?.message ?? String(err)}`));
      adapterRef.current = adapter;
      await adapter.start();
      setRecording(true);
      setStatus('Recording... 0s (tap Stop when done)');
    } catch (e: any) {
      setStatus(`Error: ${e?.message ?? String(e)}`);
    }
  };

  // ---------- button ----------
  const toggleRecording = async () => {
    if (recording) {
      if (activeModeRef.current === 'batch') await stopBatch();
      else await stopLive();
      return;
    }
    if (!(await ensureMic())) return;
    activeModeRef.current = batchMode ? 'batch' : 'live';
    if (batchMode) await startBatch();
    else await startLive();
  };

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.title}>Offline Transcriber</Text>

      <View style={styles.row}>
        <Text style={styles.modeText}>
          {batchMode ? 'Batch: transcribe after you stop' : 'Live: text while you speak'}
        </Text>
        <Switch value={batchMode} onValueChange={setBatchMode} disabled={recording || loading} />
      </View>

      <Button
        title="Transcribe test file"
        onPress={transcribeFile}
        disabled={loading || recording}
      />
      <Button
        title={recording ? 'Stop recording' : 'Record from microphone'}
        onPress={toggleRecording}
        disabled={loading}
        color={recording ? '#c0392b' : undefined}
      />
      {loading && <ActivityIndicator style={styles.spinner} size="large" />}
      <Text style={styles.status}>{status}</Text>
      <ScrollView style={styles.box}>
        <Text style={styles.result}>{text}</Text>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 20, gap: 12 },
  title: { fontSize: 24, fontWeight: 'bold' },
  row: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  modeText: { fontSize: 15, flex: 1 },
  spinner: { marginVertical: 8 },
  status: { fontSize: 14, color: '#555' },
  box: { flex: 1, borderWidth: 1, borderColor: '#ccc', borderRadius: 8, padding: 10 },
  result: { fontSize: 18 },
});