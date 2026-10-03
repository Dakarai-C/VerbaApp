import React, { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Modal,
  PermissionsAndroid,
  Pressable,
  SafeAreaView,
  ScrollView,
  Share,
  StatusBar,
  StyleSheet,
  Switch,
  Text,
  TextInput,
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

// Design tokens from the Figma file
const C = {
  canvas: '#0d0d0f',
  surface: '#141417',
  raised: '#1c1c21',
  border: '#2a2a32',
  borderSubtle: '#1e1e24',
  amber: '#f5a623',
  amberDim: '#f5a62330',
  amberMuted: '#f5a62318',
  green: '#34c97d',
  text: '#e8e8ee',
  muted: '#6b6b7a',
  faint: '#3a3a45',
  red: '#e05252',
};
const MONO = 'monospace'; // swap for Azeret Mono once the font is bundled

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

const pad = (n: number) => String(n).padStart(2, '0');
const fmt = (secs: number) => `${pad(Math.floor(secs / 60))}:${pad(Math.floor(secs % 60))}`;

type Entry = {
  id: string;
  title: string;
  date: string;
  duration: string;
  words: number;
  text: string;
  segs?: { t: number; text: string }[];
};

// ---------- small UI pieces ----------
function Waveform({ active, levelRef }: { active: boolean; levelRef: React.MutableRefObject<number> }) {
  const N = 40;
  const [bars, setBars] = useState<number[]>(() => Array(N).fill(0.06));
  useEffect(() => {
    if (!active) {
      setBars(Array(N).fill(0.06));
      return;
    }
    const id = setInterval(() => {
      const lvl = Math.min(1, levelRef.current * 4 + 0.08);
      setBars(prev => [...prev.slice(1), Math.max(0.05, lvl * (0.6 + Math.random() * 0.4))]);
    }, 80);
    return () => clearInterval(id);
  }, [active, levelRef]);
  return (
    <View style={s.wave}>
      {bars.map((h, i) => (
        <View
          key={i}
          style={{
            width: 3,
            height: `${h * 100}%`,
            borderRadius: 2,
            backgroundColor: active ? C.amber : C.faint,
            opacity: active ? 0.85 + h * 0.15 : 0.5,
          }}
        />
      ))}
    </View>
  );
}

function MenuRow(p: { icon: string; title: string; sub: string; onPress: () => void; right?: React.ReactNode }) {
  return (
    <Pressable onPress={p.onPress} style={({ pressed }) => [s.menuRow, pressed && { backgroundColor: '#ffffff0d' }]}>
      <View style={s.menuIcon}>
        <Text style={{ color: C.amber, fontSize: 16 }}>{p.icon}</Text>
      </View>
      <View style={{ flex: 1 }}>
        <Text style={s.menuTitle}>{p.title}</Text>
        <Text style={s.mono9}>{p.sub}</Text>
      </View>
      {p.right}
    </Pressable>
  );
}

export default function App() {
  const [status, setStatus] = useState('Ready');
  const [text, setText] = useState(''); // live preview only
  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState(false);
  const [batchMode, setBatchMode] = useState(true);
  const [elapsed, setElapsed] = useState(0);
  const [entries, setEntries] = useState<Entry[]>([]);
  const [selected, setSelected] = useState<Entry | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const [freeHrs, setFreeHrs] = useState<number | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [licensesOpen, setLicensesOpen] = useState(false);
  const [deleteAudio, setDeleteAudio] = useState(false);
  const [showTimestamps, setShowTimestamps] = useState(true);
  const [renaming, setRenaming] = useState<Entry | null>(null);
  const [draftName, setDraftName] = useState('');

  const ctxRef = useRef<any>(null);
  const transcriberRef = useRef<any>(null); // live mode
  const adapterRef = useRef<any>(null); // batch mode
  const chunksRef = useRef<Uint8Array[]>([]);
  const bytesRef = useRef(0);
  const lastSecRef = useRef(-1);
  const stoppingRef = useRef(false);
  const activeModeRef = useRef<'live' | 'batch'>('batch');
  const liveTextRef = useRef('');
  const elapsedRef = useRef(0);
  const levelRef = useRef(0);

  // timer
  useEffect(() => {
    if (!recording) return;
    elapsedRef.current = 0;
    setElapsed(0);
    const id = setInterval(() => {
      elapsedRef.current += 1;
      setElapsed(elapsedRef.current);
    }, 1000);
    return () => clearInterval(id);
  }, [recording]);

  // recording time left, from free storage
  useEffect(() => {
    RNFS.getFSInfo()
      .then(i => setFreeHrs(i.freeSpace / BYTES_PER_SEC / 3600))
      .catch(() => {});
  }, [recording, entries.length]);

  const nextName = () => `untitled-${String(entries.length + 1).padStart(4, '0')}`;

  const addEntry = (raw: string, secs: number, title?: string, segments?: any[]) => {
    const t = raw.trim();
    if (!t) {
      setStatus('Done: no speech detected');
      return;
    }
    const entry: Entry = {
      id: String(Date.now()),
      title: title ?? nextName(),
      date: new Date().toISOString().slice(0, 10),
      duration: fmt(secs),
      words: t.split(/\s+/).length,
      text: t,
      segs: segments?.length
        ? segments.map((g: any) => ({ t: (g.t0 ?? 0) / 100, text: String(g.text ?? '').trim() }))
        : undefined,
    };
    setEntries(cur => [entry, ...cur]);
    setDraftName(entry.title);
    setRenaming(entry);
  };

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
    try {
      if (!(await RNFS.exists(AUDIO_PATH))) {
        throw new Error(`Audio not found at ${AUDIO_PATH}`);
      }
      const ctx = await getContext();
      setStatus('Transcribing...');
      const start = Date.now();
      const { promise } = ctx.transcribe(AUDIO_PATH, { language: 'en' });
      const out: any = await promise;
      addEntry(out.result, 11, 'jfk-test', out.segments);
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
      liveTextRef.current = '';
      setLoading(true);
      const ctx = await getContext();
      setLoading(false);
      const transcriber = new RealtimeTranscriber(
        { whisperContext: ctx, audioStream: new AudioPcmStreamAdapter(), fs: RNFS },
        { audioSliceSec: 10, transcribeOptions: { language: 'en' } },
        {
          onTranscribe: (event: any) => {
            const t = event?.data?.result?.trim();
            if (t) {
              liveTextRef.current = liveTextRef.current ? liveTextRef.current + ' ' + t : t;
              setText(liveTextRef.current);
            }
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
    addEntry(liveTextRef.current, elapsedRef.current);
    liveTextRef.current = '';
    setText('');
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
      const out: any = await promise;
      addEntry(out.result, seconds, undefined, out.segments);
      if (deleteAudio) RNFS.unlink(REC_PATH).catch(() => {});
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

        // loudness for the waveform
        const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
        let sum = 0;
        let n = 0;
        for (let i = 0; i + 1 < b.length; i += 32) {
          const v = dv.getInt16(i, true) / 32768;
          sum += v * v;
          n++;
        }
        levelRef.current = n ? Math.sqrt(sum / n) : 0;

        const sec = Math.floor(bytesRef.current / BYTES_PER_SEC);
        if (sec !== lastSecRef.current) {
          lastSecRef.current = sec;
          setStatus(`Recording... ${sec}s`);
        }
        if (sec >= MAX_SECONDS) stopBatch();
      });
      adapter.onError?.((err: any) => setStatus(`Error: ${err?.message ?? String(err)}`));
      adapterRef.current = adapter;
      await adapter.start();
      setRecording(true);
      setStatus('Recording... 0s');
    } catch (e: any) {
      setStatus(`Error: ${e?.message ?? String(e)}`);
    }
  };

  // ---------- record button ----------
  const toggleRecording = async () => {
    if (recording) {
      if (activeModeRef.current === 'batch') await stopBatch();
      else await stopLive();
      return;
    }
    if (!(await ensureMic())) return;
    setText('');
    activeModeRef.current = batchMode ? 'batch' : 'live';
    if (batchMode) await startBatch();
    else await startLive();
  };

  const exportEntry = (e: Entry) =>
    Share.share({ message: `${e.title}\n${e.date} · ${e.duration}\n\n${e.text}` });

  const confirmRename = (keep: boolean) => {
    if (!renaming) return;
    const name = keep ? renaming.title : draftName.trim() || renaming.title;
    const updated = { ...renaming, title: name };
    setEntries(cur => cur.map(e => (e.id === updated.id ? updated : e)));
    setRenaming(null);
    setSelected(updated);
  };

  const isError = status.startsWith('Error');

  return (
    <SafeAreaView style={s.container}>
      <StatusBar barStyle="light-content" backgroundColor={C.canvas} />

      {/* Header */}
      <View style={s.header}>
        <Text style={[s.mono10, { letterSpacing: 2, fontWeight: '500' }]}>TRANSCRIBE</Text>
        <View style={s.headerRight}>
          <Text style={s.mono10}>↓ OFFLINE</Text>
          <View style={{ flexDirection: 'row', alignItems: 'flex-end', gap: 2 }}>
            {[1, 2, 3, 4].map(b => (
              <View
                key={b}
                style={{ width: 3, height: 8 + b * 2, borderRadius: 1, backgroundColor: b <= 3 ? C.text : C.faint }}
              />
            ))}
          </View>
        </View>
      </View>

      <ScrollView contentContainerStyle={{ paddingBottom: 24 }}>
        {/* Record panel */}
        <View
          style={[
            s.panel,
            recording && { backgroundColor: C.amberMuted, borderColor: C.amberDim },
          ]}
        >
          <View style={s.panelTop}>
            <View>
              <Text style={[s.label, { color: recording ? C.amber : C.muted }]}>
                {recording ? 'RECORDING' : loading ? 'WORKING' : 'READY'}
              </Text>
              <Text style={s.timer}>{fmt(recording ? elapsed : 0)}</Text>
            </View>
            <Pressable
              onPress={toggleRecording}
              disabled={loading}
              accessibilityLabel={recording ? 'Stop recording' : 'Start recording'}
              style={({ pressed }) => [
                s.recBtn,
                {
                  backgroundColor: recording ? C.amber : C.raised,
                  borderColor: recording ? C.amber : C.faint,
                  opacity: loading ? 0.4 : 1,
                  transform: [{ scale: pressed ? 0.95 : 1 }],
                },
                recording && { shadowColor: C.amber, elevation: 10 },
              ]}
            >
              {recording ? (
                <View style={{ width: 16, height: 16, borderRadius: 3, backgroundColor: C.canvas }} />
              ) : (
                <View style={{ width: 18, height: 18, borderRadius: 9, backgroundColor: C.amber }} />
              )}
            </Pressable>
          </View>

          <Waveform active={recording} levelRef={levelRef} />

          {!!text && recording && (
            <Text style={s.livePreview} numberOfLines={4}>
              {text}
            </Text>
          )}

          <View style={s.statusRow}>
            {loading && <ActivityIndicator size="small" color={C.amber} />}
            <Text style={[s.mono10, { flex: 1, color: isError ? C.red : C.muted }]}>{status}</Text>
          </View>

          <View style={s.panelFoot}>
            <Text style={s.mono10}>{nextName().toUpperCase()}</Text>
            <View style={{ alignItems: 'flex-end' }}>
              <Text style={s.hours}>{freeHrs === null ? '--' : freeHrs.toFixed(2)} HRS</Text>
              <Text style={s.mono9}>RECORDING TIME LEFT</Text>
            </View>
          </View>
        </View>

        {/* Recent */}
        <View style={{ paddingHorizontal: 16, marginTop: 24 }}>
          <Text style={[s.mono10, { letterSpacing: 2, fontWeight: '500', marginBottom: 12 }]}>RECENT</Text>
          {entries.length === 0 && (
            <Text style={s.empty}>No transcripts yet. Tap the record button to make your first one.</Text>
          )}
          <View style={{ gap: 8 }}>
            {entries.map(e => (
              <Pressable
                key={e.id}
                onPress={() => setSelected(e)}
                style={({ pressed }) => [s.card, pressed && { borderColor: C.faint }]}
              >
                <Text style={s.cardTitle} numberOfLines={1}>
                  {e.title}
                </Text>
                <Text style={[s.mono9, { marginTop: 6 }]}>
                  {e.duration} · {e.words.toLocaleString()} WORDS
                </Text>
              </Pressable>
            ))}
          </View>
        </View>
      </ScrollView>

      {/* Bottom nav */}
      <View style={s.nav}>
        <View style={s.navItem}>
          <Text style={{ color: C.amber, fontSize: 16 }}>▤</Text>
          <Text style={[s.mono9, { color: C.amber, letterSpacing: 1 }]}>HOME</Text>
        </View>
        <Pressable onPress={() => setMenuOpen(true)} accessibilityLabel="Add audio" style={s.fab}>
          <Text style={{ color: C.amber, fontSize: 28, lineHeight: 30 }}>+</Text>
        </Pressable>
        <Pressable
          style={s.navItem}
          onPress={() => setSettingsOpen(true)}
          accessibilityLabel="Settings"
        >
          <Text style={{ color: C.faint, fontSize: 16 }}>◈</Text>
          <Text style={[s.mono9, { letterSpacing: 1 }]}>SETTINGS</Text>
        </Pressable>
      </View>

      {/* Add audio menu */}
      <Modal visible={menuOpen} transparent animationType="fade" onRequestClose={() => setMenuOpen(false)}>
        <View style={s.modalEnd}>
          <Pressable style={s.backdrop} onPress={() => setMenuOpen(false)} />
          <View style={s.menu}>
            <View style={s.menuHead}>
              <Text style={[s.mono9, { letterSpacing: 2 }]}>ADD AUDIO</Text>
              <Text style={s.menuTitle}>Choose a source</Text>
            </View>
            <View style={{ padding: 8 }}>
              <MenuRow
                icon="↓"
                title="Transcribe test file"
                sub="JFK SAMPLE ON DEVICE"
                onPress={() => {
                  setMenuOpen(false);
                  transcribeFile();
                }}
              />
              <MenuRow
                icon="∿"
                title="Toggle real-time"
                sub="LIVE TRANSCRIPTION"
                onPress={() => !recording && !loading && setBatchMode(v => !v)}
                right={
                  <Switch
                    value={!batchMode}
                    onValueChange={v => setBatchMode(!v)}
                    disabled={recording || loading}
                    trackColor={{ false: C.faint, true: C.amber }}
                    thumbColor={C.canvas}
                  />
                }
              />
              <MenuRow
                icon="●"
                title="New recording"
                sub="START CAPTURING NOW"
                onPress={() => {
                  setMenuOpen(false);
                  if (!recording && !loading) toggleRecording();
                }}
              />
            </View>
          </View>
        </View>
      </Modal>

      {/* Rename dialog */}
      <Modal visible={!!renaming} transparent animationType="fade" onRequestClose={() => confirmRename(true)}>
        <View style={[s.modalEnd, { justifyContent: 'flex-start', paddingTop: 120, paddingHorizontal: 20 }]}>
          <Pressable style={s.backdrop} onPress={() => confirmRename(true)} />
          <View style={s.dialog}>
            <View style={{ padding: 20 }}>
              <Text style={[s.mono9, { color: C.amber, letterSpacing: 2 }]}>TRANSCRIPT READY</Text>
              <Text style={[s.menuTitle, { fontSize: 18, marginTop: 4 }]}>Rename this file?</Text>
              <Text style={s.help}>Give your recording a useful name, or keep the generated one.</Text>
              <Text style={[s.mono9, { letterSpacing: 1.5, marginTop: 20, marginBottom: 8 }]}>FILE NAME</Text>
              <TextInput
                value={draftName}
                onChangeText={setDraftName}
                autoFocus
                selectTextOnFocus
                autoCorrect={false}
                autoCapitalize="none"
                placeholderTextColor={C.muted}
                style={s.input}
                accessibilityLabel="Recording file name"
              />
            </View>
            <View style={s.btnRow}>
              <Pressable style={[s.btn, { borderWidth: 1, borderColor: C.border }]} onPress={() => confirmRename(true)}>
                <Text style={[s.btnText, { color: C.muted }]}>KEEP NAME</Text>
              </Pressable>
              <Pressable style={[s.btn, { backgroundColor: C.amber }]} onPress={() => confirmRename(false)}>
                <Text style={[s.btnText, { color: C.canvas }]}>SAVE NAME</Text>
              </Pressable>
            </View>
          </View>
        </View>
      </Modal>

      {/* Settings sheet */}
      <Modal
        visible={settingsOpen}
        transparent
        animationType="slide"
        onRequestClose={() => {
          setSettingsOpen(false);
          setLicensesOpen(false);
        }}
      >
        <View style={s.modalEnd}>
          <Pressable
            style={s.backdrop}
            onPress={() => {
              setSettingsOpen(false);
              setLicensesOpen(false);
            }}
          />
          <View style={s.sheet}>
            <View style={s.handle} />
            <View style={[s.sheetHead, { alignItems: 'center' }]}>
              {licensesOpen && (
                <Pressable onPress={() => setLicensesOpen(false)} style={s.backBtn} accessibilityLabel="Back to settings">
                  <Text style={{ color: C.muted, fontSize: 22, lineHeight: 24 }}>‹</Text>
                </Pressable>
              )}
              <View>
                <Text style={[s.mono9, { color: C.amber, letterSpacing: 2 }]}>
                  {licensesOpen ? 'OPEN SOURCE' : 'PREFERENCES'}
                </Text>
                <Text style={[s.menuTitle, { fontSize: 16, marginTop: 2 }]}>{licensesOpen ? 'Licenses' : 'Settings'}</Text>
              </View>
            </View>
            <ScrollView style={{ flexShrink: 1 }} contentContainerStyle={{ padding: 16 }}>
              {licensesOpen ? (
                <>
                  <Text style={[s.help, { marginBottom: 16 }]}>
                    Transcribe runs on open-source software. Everything stays on this device.
                  </Text>
                  <View style={s.group}>
                    {[
                      ['whisper.cpp', 'MIT'],
                      ['whisper.rn', 'MIT'],
                      ['Whisper model weights (OpenAI)', 'MIT'],
                      ['React Native', 'MIT'],
                      ['react-native-fs', 'MIT'],
                    ].map(([name, lic], i) => (
                      <View key={name} style={[s.licRow, i > 0 && { borderTopWidth: 1, borderTopColor: C.borderSubtle }]}>
                        <Text style={[s.menuTitle, { flex: 1, fontWeight: '500' }]}>{name}</Text>
                        <Text style={s.badge}>{lic}</Text>
                      </View>
                    ))}
                  </View>
                </>
              ) : (
                <>
                  <Text style={[s.mono9, { letterSpacing: 1.5, marginBottom: 8, marginLeft: 4 }]}>PRIVACY & STORAGE</Text>
                  <View style={s.group}>
                    <MenuRow
                      icon="⌫"
                      title="Delete audio after transcription"
                      sub="KEEP TRANSCRIPT ONLY"
                      onPress={() => setDeleteAudio(v => !v)}
                      right={
                        <Switch
                          value={deleteAudio}
                          onValueChange={setDeleteAudio}
                          trackColor={{ false: C.faint, true: C.amber }}
                          thumbColor={C.canvas}
                        />
                      }
                    />
                  </View>
                  <Text style={[s.mono9, { letterSpacing: 1.5, marginTop: 24, marginBottom: 8, marginLeft: 4 }]}>ABOUT</Text>
                  <View style={s.group}>
                    <MenuRow
                      icon="i"
                      title="Licenses"
                      sub="OPEN-SOURCE ATTRIBUTIONS"
                      onPress={() => setLicensesOpen(true)}
                      right={<Text style={{ color: C.muted, fontSize: 20 }}>›</Text>}
                    />
                  </View>
                </>
              )}
            </ScrollView>
          </View>
        </View>
      </Modal>

      {/* Transcript sheet */}
      <Modal visible={!!selected} transparent animationType="slide" onRequestClose={() => setSelected(null)}>
        <View style={s.modalEnd}>
          <Pressable style={s.backdrop} onPress={() => setSelected(null)} />
          {selected && (
            <View style={s.sheet}>
              <View style={s.handle} />
              <View style={s.sheetHead}>
                <View style={{ flex: 1 }}>
                  <Text style={[s.mono9, { color: C.amber, letterSpacing: 2 }]}>TRANSCRIPT</Text>
                  <Text style={[s.menuTitle, { fontSize: 16, marginTop: 4 }]} numberOfLines={1}>
                    {selected.title}
                  </Text>
                  <Text style={[s.mono9, { marginTop: 4 }]}>
                    {selected.date} · {selected.duration} · {selected.words.toLocaleString()} WORDS
                  </Text>
                </View>
                <Pressable onPress={() => exportEntry(selected)} style={s.exportBtn}>
                  <Text style={[s.mono9, { color: C.amber }]}>↑ EXPORT</Text>
                </Pressable>
              </View>
              {!!selected.segs && (
                <View style={s.tsRow}>
                  <Text style={[s.mono9, { letterSpacing: 1.5 }]}>TIMESTAMPS</Text>
                  <Switch
                    value={showTimestamps}
                    onValueChange={setShowTimestamps}
                    trackColor={{ false: C.faint, true: C.amber }}
                    thumbColor={C.canvas}
                  />
                </View>
              )}
              <ScrollView style={{ flexShrink: 1, paddingHorizontal: 20 }} contentContainerStyle={{ paddingVertical: 20, gap: 16 }}>
                {selected.segs ? (
                  selected.segs.map((g, i) => (
                    <View key={i} style={{ flexDirection: 'row', gap: 12 }}>
                      {showTimestamps && <Text style={s.tsTime}>{fmt(g.t)}</Text>}
                      <Text style={[s.transcript, { flex: 1 }]} selectable>
                        {g.text}
                      </Text>
                    </View>
                  ))
                ) : (
                  <Text style={s.transcript} selectable>
                    {selected.text}
                  </Text>
                )}
              </ScrollView>
            </View>
          )}
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const s = StyleSheet.create({
  container: { flex: 1, backgroundColor: C.canvas },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: 16,
    paddingBottom: 8,
    borderBottomWidth: 1,
    borderBottomColor: C.borderSubtle,
  },
  headerRight: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  mono10: { fontFamily: MONO, fontSize: 10, color: C.muted },
  mono9: { fontFamily: MONO, fontSize: 9, color: C.muted },

  panel: {
    marginHorizontal: 16,
    marginTop: 16,
    padding: 20,
    gap: 16,
    borderRadius: 16,
    backgroundColor: C.surface,
    borderWidth: 1,
    borderColor: C.border,
  },
  panelTop: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  label: { fontSize: 12, fontWeight: '500', marginBottom: 2 },
  timer: { fontFamily: MONO, fontSize: 30, fontWeight: '300', color: C.text },
  recBtn: {
    width: 56,
    height: 56,
    borderRadius: 28,
    borderWidth: 2,
    alignItems: 'center',
    justifyContent: 'center',
  },
  wave: { height: 40, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 2 },
  livePreview: { fontFamily: MONO, fontSize: 12, lineHeight: 20, color: C.text },
  statusRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  panelFoot: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: C.borderSubtle,
  },
  hours: { fontFamily: MONO, fontSize: 14, fontWeight: '500', color: C.text },

  empty: { fontSize: 13, lineHeight: 20, color: C.muted },
  card: {
    paddingHorizontal: 16,
    paddingVertical: 14,
    borderRadius: 12,
    backgroundColor: C.surface,
    borderWidth: 1,
    borderColor: C.border,
  },
  cardTitle: { fontSize: 14, fontWeight: '600', color: C.text },

  nav: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-around',
    paddingVertical: 16,
    backgroundColor: C.canvas,
    borderTopWidth: 1,
    borderTopColor: C.border,
  },
  navItem: { flex: 1, alignItems: 'center', gap: 4 },
  fab: {
    width: 52,
    height: 52,
    marginTop: -20,
    borderRadius: 26,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: C.raised,
    borderWidth: 1.5,
    borderColor: C.faint,
  },

  modalEnd: { flex: 1, justifyContent: 'flex-end' },
  backdrop: { ...StyleSheet.absoluteFillObject, backgroundColor: '#000000a6' },
  menu: {
    marginHorizontal: 16,
    marginBottom: 96,
    borderRadius: 16,
    overflow: 'hidden',
    backgroundColor: C.raised,
    borderWidth: 1,
    borderColor: C.faint,
  },
  menuHead: { paddingHorizontal: 16, paddingVertical: 12, borderBottomWidth: 1, borderBottomColor: C.border },
  menuTitle: { fontSize: 14, fontWeight: '600', color: C.text },
  menuRow: { flexDirection: 'row', alignItems: 'center', gap: 12, padding: 12, borderRadius: 12 },
  menuIcon: {
    width: 40,
    height: 40,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: C.amberMuted,
  },

  sheet: {
    maxHeight: '82%',
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    overflow: 'hidden',
    backgroundColor: C.surface,
    borderWidth: 1,
    borderBottomWidth: 0,
    borderColor: C.faint,
  },
  handle: { alignSelf: 'center', width: 40, height: 4, borderRadius: 2, marginVertical: 10, backgroundColor: C.faint },
  sheetHead: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 16,
    paddingHorizontal: 20,
    paddingBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: C.border,
  },
  exportBtn: {
    paddingHorizontal: 10,
    height: 32,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: C.border,
  },
  transcript: { fontFamily: MONO, fontSize: 12, lineHeight: 24, color: C.text },
  tsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 6,
    backgroundColor: C.raised,
    borderBottomWidth: 1,
    borderBottomColor: C.border,
  },
  tsTime: { width: 38, paddingTop: 7, fontFamily: MONO, fontSize: 9, color: C.amber },

  dialog: { borderRadius: 16, overflow: 'hidden', backgroundColor: C.raised, borderWidth: 1, borderColor: C.faint },
  help: { marginTop: 4, fontSize: 12, lineHeight: 20, color: C.muted },
  input: {
    paddingHorizontal: 14,
    paddingVertical: 12,
    borderRadius: 12,
    fontFamily: MONO,
    fontSize: 14,
    color: C.text,
    backgroundColor: C.canvas,
    borderWidth: 1,
    borderColor: C.border,
  },
  btnRow: { flexDirection: 'row', gap: 8, padding: 12, borderTopWidth: 1, borderTopColor: C.border },
  btn: { flex: 1, paddingVertical: 12, borderRadius: 12, alignItems: 'center' },
  btnText: { fontSize: 12, fontWeight: '600' },

  group: { borderRadius: 16, overflow: 'hidden', backgroundColor: C.raised, borderWidth: 1, borderColor: C.border },
  backBtn: { width: 32, height: 32, alignItems: 'center', justifyContent: 'center' },
  licRow: { flexDirection: 'row', alignItems: 'center', gap: 12, paddingHorizontal: 16, paddingVertical: 14 },
  badge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
    overflow: 'hidden',
    fontFamily: MONO,
    fontSize: 9,
    color: C.amber,
    backgroundColor: C.amberMuted,
  },
});
