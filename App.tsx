import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Animated,
  Easing,
  Keyboard,
  Modal,
  PanResponder,
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
  useWindowDimensions,
  View,
} from 'react-native';
import RNFS from 'react-native-fs';
import { initWhisper } from 'whisper.rn';
import { RealtimeTranscriber } from 'whisper.rn/src/realtime-transcription';
import { AudioPcmStreamAdapter } from 'whisper.rn/src/realtime-transcription/adapters/AudioPcmStreamAdapter';

const MODEL_PATH = `${RNFS.ExternalDirectoryPath}/ggml-tiny.en.bin`;
const AUDIO_PATH = `${RNFS.ExternalDirectoryPath}/jfk.wav`;
const SAMPLE_RATE = 16000;
const BYTES_PER_SEC = SAMPLE_RATE * 2; // 16-bit mono
const MAX_SECONDS = 600; // batch recordings stop automatically at 10 minutes

// Saved data lives in the app's private folder, so transcripts survive closing the app.
const ENTRIES_PATH = `${RNFS.DocumentDirectoryPath}/entries.json`;
// Every saved recording gets its own file here (internal app storage, private to this app).
// Entries refer to files by name only, so the folder can move without breaking saved transcripts.
const AUDIO_DIR = `${RNFS.DocumentDirectoryPath}/audio`;
const SETTINGS_PATH = `${RNFS.DocumentDirectoryPath}/settings.json`;

// Speech model: downloaded once from Hugging Face, then everything runs offline.
const MODEL_URL = 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin';
const MODEL_MIN_BYTES = 70000000; // ggml-tiny.en.bin is about 75 MB; anything smaller is a partial file

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

// Android system-bar inset (3-button / gesture nav). React Native core has no inset API on Android
// (its SafeAreaView is a no-op there), so use react-native-safe-area-context when it is installed.
// If it is not installed this quietly falls back to 0 and the app behaves exactly as before.
let SafeArea: any = null;
try {
  SafeArea = require('react-native-safe-area-context');
} catch {}
const useBottomInset: () => number = SafeArea ? () => SafeArea.useSafeAreaInsets().bottom : () => 0;

// one-line context around the first match, for search results that matched on transcript text
function snippetFor(text: string, q: string): string {
  const i = text.toLowerCase().indexOf(q);
  if (i < 0) return '';
  const start = Math.max(0, i - 24);
  return (start > 0 ? '…' : '') + text.slice(start, i + q.length + 40).replace(/\s+/g, ' ') + '…';
}

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
  // --- audio association (all optional, so transcripts saved by older versions still load) ---
  audioFile?: string; // file name inside AUDIO_DIR; missing = no audio kept for this entry
  source?: 'mic' | 'live' | 'sample';
  status?: 'done' | 'failed' | 'untranscribed'; // missing = done
};

const countWords = (t: string) => (t ? t.split(/\s+/).length : 0);
const mapSegs = (segments?: any[]) =>
  segments?.length
    ? segments.map((g: any) => ({ t: (g.t0 ?? 0) / 100, text: String(g.text ?? '').trim() }))
    : undefined;

function friendlyFsError(e: any): string {
  const m = String(e?.message ?? e);
  return /ENOSPC|No space left/i.test(m)
    ? 'Not enough free storage to save this recording. Free up some space and try again.'
    : m;
}

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

function AppContent() {
  const bottomInset = useBottomInset();
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
  const [sheetExpanded, setSheetExpanded] = useState(false); // transcript sheet: settled state (drag or tap the handle)
  const [searchOpen, setSearchOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [kbOpen, setKbOpen] = useState(false);

  const [loaded, setLoaded] = useState(false); // saved transcripts and settings have been read
  const [modelState, setModelState] = useState<'unknown' | 'installed' | 'missing' | 'downloading' | 'error'>('unknown');
  const [modelBytes, setModelBytes] = useState(0);
  const [modelProgress, setModelProgress] = useState(0); // 0..1
  const [modelError, setModelError] = useState('');

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

  // Latest values for code that runs from long-lived callbacks (e.g. the 10-minute auto-stop), which would
  // otherwise see the values from the render when recording started.
  const deleteAudioRef = useRef(false);
  deleteAudioRef.current = deleteAudio;
  const entriesRef = useRef<Entry[]>([]);
  entriesRef.current = entries;

  const downloadJobRef = useRef<number | null>(null);
  const cancelledRef = useRef(false);

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

  // ---------- transcript sheet: draggable height ----------
  // The sheet's height is an Animated.Value that follows the finger while the handle is dragged,
  // then animates to the collapsed or expanded height on release.
  const { height: winH } = useWindowDimensions();
  const SHEET_MIN = Math.round(winH * 0.6); // collapsed
  const SHEET_MAX = Math.round(winH * 0.94); // expanded
  const boundsRef = useRef({ min: SHEET_MIN, max: SHEET_MAX });
  boundsRef.current = { min: SHEET_MIN, max: SHEET_MAX };
  const sheetH = useRef(new Animated.Value(SHEET_MIN)).current;
  const sheetHVal = useRef(SHEET_MIN); // latest height, kept in sync by the listener below
  const dragStartH = useRef(SHEET_MIN);
  const dragStartExpanded = useRef(false);

  useEffect(() => {
    const id = sheetH.addListener(({ value }) => {
      sheetHVal.current = value;
    });
    return () => sheetH.removeListener(id);
  }, [sheetH]);

  const snapSheet = (expand: boolean) => {
    setSheetExpanded(expand);
    Animated.timing(sheetH, {
      toValue: expand ? boundsRef.current.max : boundsRef.current.min,
      duration: 220,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: false, // height is a layout property
    }).start();
  };

  const toggleSheet = () => {
    const { min, max } = boundsRef.current;
    snapSheet(sheetHVal.current < (min + max) / 2);
  };

  // Only attached to the handle, so it never competes with the transcript ScrollView.
  // Every drag starts from the sheet's current height (dragStartH) and applies the gesture's total dy
  // to it, clamped between collapsed and expanded, so nothing accumulates across gestures.
  const sheetPan = useMemo(
    () =>
      PanResponder.create({
        onStartShouldSetPanResponder: () => true,
        onMoveShouldSetPanResponder: () => true,
        onPanResponderTerminationRequest: () => false,
        onPanResponderGrant: () => {
          sheetH.stopAnimation();
          const { min, max } = boundsRef.current;
          dragStartH.current = sheetHVal.current;
          dragStartExpanded.current = sheetHVal.current > (min + max) / 2;
        },
        onPanResponderMove: (_, g) => {
          const { min, max } = boundsRef.current;
          // finger up (dy < 0) = taller sheet
          sheetH.setValue(Math.max(min, Math.min(max, dragStartH.current - g.dy)));
        },
        onPanResponderRelease: (_, g) => {
          const { min, max } = boundsRef.current;
          if (Math.abs(g.dx) < 6 && Math.abs(g.dy) < 6) {
            toggleSheet(); // plain tap
            return;
          }
          // a real flick decides by direction, but only if it also travelled a meaningful distance
          if (Math.abs(g.vy) > 0.5 && Math.abs(g.dy) > 24) {
            snapSheet(g.vy < 0);
            return;
          }
          // otherwise it has to be dragged far enough to switch state (hysteresis), so small accidental
          // movements fall back to where the sheet started
          const progress = (sheetHVal.current - min) / (max - min);
          snapSheet(dragStartExpanded.current ? progress > 0.7 : progress > 0.3);
        },
        onPanResponderTerminate: () => {
          const { min, max } = boundsRef.current;
          snapSheet(sheetHVal.current >= (min + max) / 2);
        },
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  // always open a transcript at the collapsed size
  useEffect(() => {
    sheetH.stopAnimation();
    sheetH.setValue(boundsRef.current.min);
    setSheetExpanded(false);
  }, [selected?.id, sheetH]);

  // recording time left, from free storage
  useEffect(() => {
    RNFS.getFSInfo()
      .then(i => setFreeHrs(i.freeSpace / BYTES_PER_SEC / 3600))
      .catch(() => {});
  }, [recording, entries.length]);

  // search: hide the bottom nav while the keyboard is up so it doesn't ride on top of it
  useEffect(() => {
    const a = Keyboard.addListener('keyboardDidShow', () => setKbOpen(true));
    const b = Keyboard.addListener('keyboardDidHide', () => setKbOpen(false));
    return () => {
      a.remove();
      b.remove();
    };
  }, []);

  const q = searchOpen ? query.trim().toLowerCase() : '';
  const visible = useMemo(
    () => (q ? entries.filter(e => e.title.toLowerCase().includes(q) || e.text.toLowerCase().includes(q)) : entries),
    [entries, q],
  );
  const toggleSearch = () => {
    setSearchOpen(v => !v);
    setQuery('');
  };
  const openEntry = (e: Entry) => {
    Keyboard.dismiss();
    if (e.status === 'failed' || e.status === 'untranscribed') {
      Alert.alert(
        'Not transcribed yet',
        'This audio is saved on your device but has no transcript. Transcribe it now?',
        [
          { text: 'Not now', style: 'cancel' },
          { text: 'Transcribe', onPress: () => retryTranscription(e) },
        ],
      );
      return;
    }
    setSelected(e);
  };

  // next free "untitled-0001" style name; never reuses a number even if entries are renamed or removed later
  const nextName = () => {
    const list = entriesRef.current;
    const maxUsed = list.reduce((m, e) => {
      const x = /^untitled-(\d+)$/.exec(e.title);
      return x ? Math.max(m, Number(x[1])) : m;
    }, 0);
    return `untitled-${String(Math.max(maxUsed, list.length) + 1).padStart(4, '0')}`;
  };

  const addEntry = (raw: string, secs: number, title?: string, segments?: any[], extra: Partial<Entry> = {}) => {
    const t = raw.trim();
    // nothing to show: no speech and no audio kept
    if (!t && !extra.audioFile) {
      setStatus('Done: no speech detected');
      return;
    }
    const entry: Entry = {
      id: String(Date.now()),
      title: title ?? nextName(),
      date: new Date().toISOString().slice(0, 10),
      duration: fmt(secs),
      words: countWords(t),
      text: t,
      segs: mapSegs(segments),
      ...extra,
    };
    setEntries(cur => [entry, ...cur]);
    // only offer the rename dialog when there is a real transcript to name
    if (t) {
      setDraftName(entry.title);
      setRenaming(entry);
    }
  };

  // Transcribe audio that was already saved (a failed attempt, or a file recovered after the app was closed
  // mid-transcription). The retention setting applies once this succeeds, exactly as for a new recording.
  const retryTranscription = async (e: Entry) => {
    if (!e.audioFile || loading || recording) return;
    setLoading(true);
    try {
      const path = `${AUDIO_DIR}/${e.audioFile}`;
      if (!(await RNFS.exists(path))) throw new Error('The saved audio file could not be found.');
      const ctx = await getContext();
      setStatus('Transcribing...');
      const start = Date.now();
      const { promise } = ctx.transcribe(path, { language: 'en' });
      const out: any = await promise;
      const t = String(out.result ?? '').trim();
      const keep = !deleteAudioRef.current;
      if (!keep) await RNFS.unlink(path).catch(() => {});
      const updated: Entry = {
        ...e,
        text: t,
        words: countWords(t),
        segs: mapSegs(out.segments),
        status: 'done',
        audioFile: keep ? e.audioFile : undefined,
      };
      setEntries(cur => cur.map(x => (x.id === e.id ? updated : x)));
      setStatus(`Done in ${((Date.now() - start) / 1000).toFixed(1)}s`);
      if (t) {
        setDraftName(updated.title);
        setRenaming(updated);
      }
    } catch (err: any) {
      setStatus(`Error: ${err?.message ?? String(err)} Your audio is still saved.`);
    } finally {
      setLoading(false);
    }
  };

  const getContext = async () => {
    if (!ctxRef.current) {
      if (!(await RNFS.exists(MODEL_PATH))) {
        throw new Error('Speech model not downloaded. Open Settings > Speech model to get it.');
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

  // ---------- speech model: check / download ----------

  const checkModel = async (): Promise<boolean> => {
    try {
      if (await RNFS.exists(MODEL_PATH)) {
        const st = await RNFS.stat(MODEL_PATH);
        if (Number(st.size) >= MODEL_MIN_BYTES) {
          setModelBytes(Number(st.size));
          setModelState('installed');
          return true;
        }
      }
    } catch {}
    setModelState('missing');
    return false;
  };

  const downloadModel = async () => {
    if (downloadJobRef.current !== null) return;
    cancelledRef.current = false;
    setModelError('');
    setModelProgress(0);
    setModelState('downloading');
    const part = `${MODEL_PATH}.part`; // download to a temp name so a half-finished file is never used
    try {
      await RNFS.unlink(part).catch(() => {});
      const job = RNFS.downloadFile({
        fromUrl: MODEL_URL,
        toFile: part,
        progressInterval: 500,
        progress: ({ bytesWritten, contentLength }) => {
          if (contentLength > 0) setModelProgress(bytesWritten / contentLength);
        },
      });
      downloadJobRef.current = job.jobId;
      const res = await job.promise;
      downloadJobRef.current = null;
      if (res.statusCode !== 200) throw new Error(`Server answered ${res.statusCode}`);
      const st = await RNFS.stat(part);
      if (Number(st.size) < MODEL_MIN_BYTES) throw new Error('Download was incomplete');
      if (await RNFS.exists(MODEL_PATH)) await RNFS.unlink(MODEL_PATH);
      await RNFS.moveFile(part, MODEL_PATH);
      ctxRef.current = null; // load the fresh file next time
      await checkModel();
      setStatus('Speech model ready');
    } catch (e: any) {
      downloadJobRef.current = null;
      RNFS.unlink(part).catch(() => {});
      if (cancelledRef.current) {
        setModelState('missing');
      } else {
        setModelError(e?.message ?? String(e));
        setModelState('error');
      }
    }
  };

  const cancelDownload = () => {
    cancelledRef.current = true;
    if (downloadJobRef.current !== null) RNFS.stopDownload(downloadJobRef.current);
  };

  const confirmDownload = () =>
    Alert.alert(
      'Download speech model?',
      'It is about 75 MB and downloads once. After that, viva voce works fully offline. Wi-Fi is best.',
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Download', onPress: downloadModel },
      ],
    );

  // the Settings row: check for the model, and offer the download if it is missing
  const onModelPress = async () => {
    if (modelState === 'downloading') {
      cancelDownload();
      return;
    }
    const ok = await checkModel();
    if (!ok) confirmDownload();
  };

  // ---------- saved data: transcripts + settings ----------

  useEffect(() => {
    (async () => {
      try {
        let list: Entry[] = [];
        if (await RNFS.exists(ENTRIES_PATH)) {
          const data = JSON.parse(await RNFS.readFile(ENTRIES_PATH, 'utf8'));
          if (Array.isArray(data)) list = data;
        }
        // Audio files that no transcript points to (the app was closed mid-transcription) are never deleted:
        // they come back as "not transcribed" entries so the user can transcribe or remove them.
        try {
          await RNFS.mkdir(AUDIO_DIR);
          const known = new Set(list.map(e => e.audioFile).filter(Boolean));
          const found = (await RNFS.readDir(AUDIO_DIR)).filter(f => f.isFile() && /\.wav$/i.test(f.name) && !known.has(f.name));
          const recovered: Entry[] = found.map((f, i) => ({
            id: `${f.name.replace(/\.wav$/i, '')}-${i}`,
            title: `recovered-${f.name.replace(/^rec-/, '').replace(/\.wav$/i, '')}`,
            date: (f.mtime ?? new Date()).toISOString().slice(0, 10),
            duration: fmt(Math.max(0, (Number(f.size) - 44) / BYTES_PER_SEC)),
            words: 0,
            text: '',
            audioFile: f.name,
            source: 'mic',
            status: 'untranscribed',
          }));
          list = [...recovered, ...list];
        } catch {}
        setEntries(list);
      } catch {}
      try {
        if (await RNFS.exists(SETTINGS_PATH)) {
          const st = JSON.parse(await RNFS.readFile(SETTINGS_PATH, 'utf8'));
          if (typeof st.deleteAudio === 'boolean') setDeleteAudio(st.deleteAudio);
          if (typeof st.showTimestamps === 'boolean') setShowTimestamps(st.showTimestamps);
          if (typeof st.batchMode === 'boolean') setBatchMode(st.batchMode);
        }
      } catch {}
      setLoaded(true);
      if (!(await checkModel())) setStatus('Speech model not downloaded. Open Settings to get it.');
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // write after every change, but only once the saved copy has been read (so we never overwrite it with an empty list)
  useEffect(() => {
    if (!loaded) return;
    RNFS.writeFile(ENTRIES_PATH, JSON.stringify(entries), 'utf8').catch(() => {});
  }, [entries, loaded]);

  useEffect(() => {
    if (!loaded) return;
    RNFS.writeFile(SETTINGS_PATH, JSON.stringify({ deleteAudio, showTimestamps, batchMode }), 'utf8').catch(() => {});
  }, [deleteAudio, showTimestamps, batchMode, loaded]);

  // re-check the model whenever Settings is opened
  useEffect(() => {
    if (settingsOpen && modelState !== 'downloading') checkModel();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settingsOpen]);

  const modelSub =
    modelState === 'installed'
      ? `INSTALLED · ${(modelBytes / 1e6).toFixed(0)} MB`
      : modelState === 'downloading'
      ? `DOWNLOADING ${Math.round(modelProgress * 100)}% · TAP TO CANCEL`
      : modelState === 'error'
      ? 'DOWNLOAD FAILED · TAP TO RETRY'
      : modelState === 'missing'
      ? 'NOT DOWNLOADED · TAP TO DOWNLOAD'
      : 'CHECKING…';

  const modelRight =
    modelState === 'installed' ? (
      <Text style={{ color: C.green, fontSize: 16 }}>✓</Text>
    ) : modelState === 'downloading' ? (
      <ActivityIndicator size="small" color={C.amber} />
    ) : modelState === 'error' ? (
      <Text style={{ color: C.red, fontSize: 16 }}>!</Text>
    ) : modelState === 'missing' ? (
      <Text style={{ color: C.amber, fontSize: 16 }}>↓</Text>
    ) : null;

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
      addEntry(out.result, 11, 'jfk-test', out.segments, { source: 'sample' });
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
    addEntry(liveTextRef.current, elapsedRef.current, undefined, undefined, { source: 'live' });
    liveTextRef.current = '';
    setText('');
  };

  // ---------- batch mode ----------
  const stopBatch = async () => {
    if (stoppingRef.current) return;
    stoppingRef.current = true;
    setRecording(false);
    setLoading(true);
    // Retention policy: the setting is read once, when recording stops. Changing it later never affects this file.
    const discardAfter = deleteAudioRef.current;
    const id = String(Date.now());
    let audioFile: string | undefined; // set only once the audio is safely on disk
    let seconds = 0;
    try {
      const adapter = adapterRef.current;
      adapterRef.current = null;
      try {
        await adapter?.stop();
        await adapter?.release?.();
      } catch {}

      const total = bytesRef.current;
      seconds = total / BYTES_PER_SEC;
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
      const name = `rec-${id}.wav`;
      const path = `${AUDIO_DIR}/${name}`;
      try {
        await RNFS.mkdir(AUDIO_DIR);
        await RNFS.writeFile(path, toBase64(all), 'base64');
      } catch (err: any) {
        RNFS.unlink(path).catch(() => {}); // don't leave a half-written file behind
        throw new Error(friendlyFsError(err));
      }
      audioFile = name;

      const ctx = await getContext();
      setStatus(`Transcribing ${seconds.toFixed(0)}s of audio...`);
      const start = Date.now();
      const { promise } = ctx.transcribe(path, { language: 'en' });
      const out: any = await promise;

      // Transcription succeeded, so the policy can now be applied. (Before this point nothing is ever deleted.)
      if (discardAfter) await RNFS.unlink(path).catch(() => {});
      addEntry(out.result ?? '', seconds, undefined, out.segments, {
        id,
        source: 'mic',
        audioFile: discardAfter ? undefined : audioFile,
        status: 'done',
      });
      setStatus(
        `Done: ${seconds.toFixed(0)}s of audio transcribed in ${((Date.now() - start) / 1000).toFixed(1)}s`,
      );
    } catch (e: any) {
      const msg = e?.message ?? String(e);
      if (audioFile) {
        // The audio is saved but transcription failed: keep the file and show it in Recent so it can be retried.
        addEntry('', seconds, undefined, undefined, { id, source: 'mic', audioFile, status: 'failed' });
        setStatus(`Error: ${msg} Your audio was saved; tap it in Recent to try again.`);
      } else {
        setStatus(`Error: ${msg}`);
      }
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

  const closeSettings = () => {
    setSettingsOpen(false);
    setLicensesOpen(false);
  };
  const noop = () => {}; // inner Pressables use this so taps on a dialog don't reach the dismiss layer

  const isError = status.startsWith('Error');

  // rename dialog: compare against the name the dialog opened with (renaming.title)
  const trimmedDraft = draftName.trim();
  const nameChanged = trimmedDraft.length > 0 && trimmedDraft !== (renaming?.title ?? '');

  return (
    <SafeAreaView style={s.container}>
      <StatusBar barStyle="light-content" backgroundColor={C.canvas} />

      <ScrollView contentContainerStyle={{ paddingTop: 8, paddingBottom: 24 }} keyboardShouldPersistTaps="handled">
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
          <View style={s.recentHead}>
            <Text style={[s.mono10, { letterSpacing: 2, fontWeight: '500' }]}>RECENT</Text>
            {entries.length > 0 && (
              <Pressable
                onPress={toggleSearch}
                style={s.exportBtn}
                accessibilityLabel={searchOpen ? 'Close search' : 'Search transcripts'}
              >
                <Text style={[s.mono9, { color: C.amber }]}>{searchOpen ? '✕ CLOSE' : '⌕ SEARCH'}</Text>
              </Pressable>
            )}
          </View>
          {searchOpen && (
            <View style={s.searchBox}>
              <Text style={{ color: C.muted, fontSize: 14 }}>⌕</Text>
              <TextInput
                value={query}
                onChangeText={setQuery}
                autoFocus
                autoCorrect={false}
                autoCapitalize="none"
                returnKeyType="search"
                placeholder="Search names and text"
                placeholderTextColor={C.muted}
                style={s.searchInput}
                accessibilityLabel="Search transcripts"
              />
              {!!query && (
                <Pressable onPress={() => setQuery('')} hitSlop={10} accessibilityLabel="Clear search">
                  <Text style={{ color: C.muted, fontSize: 14 }}>✕</Text>
                </Pressable>
              )}
            </View>
          )}
          {entries.length === 0 && (
            <Text style={s.empty}>No transcripts yet. Tap the record button to make your first one.</Text>
          )}
          {entries.length > 0 && visible.length === 0 && (
            <Text style={s.empty}>No transcripts match “{query.trim()}”.</Text>
          )}
          <View style={{ gap: 8 }}>
            {visible.map(e => (
              <Pressable
                key={e.id}
                onPress={() => openEntry(e)}
                style={({ pressed }) => [s.card, pressed && { borderColor: C.faint }]}
              >
                <Text style={s.cardTitle} numberOfLines={1}>
                  {e.title}
                </Text>
                <Text style={[s.mono9, { marginTop: 6 }]}>
                  {e.status === 'failed' || e.status === 'untranscribed'
                    ? `${e.duration} · NOT TRANSCRIBED · AUDIO SAVED`
                    : e.words === 0
                    ? `${e.duration} · NO SPEECH DETECTED${e.audioFile ? ' · AUDIO SAVED' : ''}`
                    : `${e.duration} · ${e.words.toLocaleString()} WORDS${e.audioFile ? ' · AUDIO SAVED' : ''}`}
                </Text>
                {!!q && !e.title.toLowerCase().includes(q) && (
                  <Text style={[s.mono9, { marginTop: 6, color: C.text }]} numberOfLines={2}>
                    {snippetFor(e.text, q)}
                  </Text>
                )}
              </Pressable>
            ))}
          </View>
        </View>
      </ScrollView>

      {/* Bottom nav */}
      {!(kbOpen && searchOpen) && (
      <View style={[s.nav, { paddingBottom: 16 + bottomInset }]}>
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
      )}

      {/* Add audio menu */}
      <Modal
        visible={menuOpen}
        transparent
        statusBarTranslucent
        animationType="fade"
        onRequestClose={() => setMenuOpen(false)}
      >
        <Pressable style={s.overlay} onPress={() => setMenuOpen(false)}>
          <Pressable style={s.menu} onPress={noop}>
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
          </Pressable>
        </Pressable>
      </Modal>

      {/* Rename dialog */}
      <Modal
        visible={!!renaming}
        transparent
        statusBarTranslucent
        animationType="fade"
        onRequestClose={() => confirmRename(true)}
      >
        <Pressable
          style={[s.overlay, { justifyContent: 'flex-start', paddingTop: 120, paddingHorizontal: 20 }]}
          onPress={() => confirmRename(true)}
        >
          <Pressable style={s.dialog} onPress={noop}>
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
                returnKeyType="done"
                onSubmitEditing={() => confirmRename(!nameChanged)}
                placeholderTextColor={C.muted}
                style={s.input}
                accessibilityLabel="Recording file name"
              />
            </View>
            <View style={s.btnRow}>
              <Pressable
                style={[s.btn, { backgroundColor: nameChanged ? C.amber : C.faint }]}
                onPress={() => confirmRename(!nameChanged)}
                accessibilityLabel={nameChanged ? 'Save name' : 'Keep name'}
              >
                <Text style={[s.btnText, { color: nameChanged ? C.canvas : C.muted }]}>
                  {nameChanged ? 'SAVE NAME' : 'KEEP NAME'}
                </Text>
              </Pressable>
            </View>
          </Pressable>
        </Pressable>
      </Modal>

      {/* Settings sheet */}
      <Modal
        visible={settingsOpen}
        transparent
        statusBarTranslucent
        animationType="slide"
        onRequestClose={closeSettings}
      >
        <Pressable style={s.overlay} onPress={closeSettings}>
          <Pressable style={s.sheet} onPress={noop}>
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
                    viva voce is built on open-source software. Your audio and transcripts stay on this device.
                  </Text>
                  <View style={s.group}>
                    {[
                      ['whisper.cpp', 'MIT'],
                      ['whisper.rn', 'MIT'],
                      ['Whisper model weights (OpenAI)', 'MIT'],
                      ['React Native', 'MIT'],
                      ['react-native-fs', 'MIT'],
                      ['@fugood/react-native-audio-pcm-stream', 'MIT'],
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
                  <Text style={[s.help, { marginLeft: 4, marginTop: 8 }]}>
                    Applies to new recordings, after they are transcribed. Audio you have already saved is never
                    deleted by changing this, and audio from a failed transcription is always kept.
                  </Text>
                  <Text style={[s.mono9, { letterSpacing: 1.5, marginTop: 24, marginBottom: 8, marginLeft: 4 }]}>SPEECH MODEL</Text>
                  <View style={s.group}>
                    <MenuRow
                      icon="◉"
                      title="Speech model"
                      sub={modelSub}
                      onPress={onModelPress}
                      right={modelRight}
                    />
                    {modelState === 'downloading' && (
                      <View style={s.modelBar}>
                        <View style={[s.modelBarFill, { width: `${Math.max(2, Math.round(modelProgress * 100))}%` }]} />
                      </View>
                    )}
                  </View>
                  {modelState === 'error' && !!modelError && (
                    <Text style={[s.help, { marginLeft: 4, color: C.red }]}>{modelError}</Text>
                  )}
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
          </Pressable>
        </Pressable>
      </Modal>

      {/* Transcript sheet */}
      <Modal
        visible={!!selected}
        transparent
        statusBarTranslucent
        animationType="slide"
        onRequestClose={() => setSelected(null)}
      >
        {/* Backdrop and sheet are siblings: no Pressable wraps the sheet, so the only touch handlers inside it
            are the handle's PanResponder and the transcript ScrollView, and they never compete. */}
        <View style={s.overlay}>
          <Pressable
            style={StyleSheet.absoluteFill}
            onPress={() => setSelected(null)}
            accessibilityLabel="Close transcript"
          />
          {selected && (
             <Animated.View style={[s.sheet, { height: sheetH, maxHeight: winH }]}>
              {/* Drag the handle to resize the sheet (it follows the finger); a tap also toggles */}
              <View
                {...sheetPan.panHandlers}
                style={s.handleHit}
                accessible
                accessibilityRole="button"
                accessibilityLabel={sheetExpanded ? 'Collapse transcript' : 'Expand transcript'}
                onAccessibilityTap={toggleSheet}
              >
                <View style={[s.handle, { marginVertical: 0 }]} />
              </View>
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
              {/* flex: 1 fills exactly the space left under the header, whatever the sheet height is */}
              <ScrollView
                style={{ flex: 1, paddingHorizontal: 20 }}
                contentContainerStyle={{ paddingTop: 20, paddingBottom: 56 + bottomInset, gap: 16 }}
                nestedScrollEnabled
                showsVerticalScrollIndicator
              >
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
                    {selected.text || 'No speech was detected in this audio.'}
                  </Text>
                )}
              </ScrollView>
             </Animated.View>
          )}
        </View>
      </Modal>
    </SafeAreaView>
  );
}

export default function App() {
  return SafeArea ? (
    <SafeArea.SafeAreaProvider>
      <AppContent />
    </SafeArea.SafeAreaProvider>
  ) : (
    <AppContent />
  );
}

const s = StyleSheet.create({
  container: { flex: 1, backgroundColor: C.canvas },
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

  recentHead: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 },
  searchBox: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginBottom: 12,
    paddingHorizontal: 14,
    borderRadius: 12,
    backgroundColor: C.surface,
    borderWidth: 1,
    borderColor: C.border,
  },
  searchInput: { flex: 1, paddingVertical: 10, fontFamily: MONO, fontSize: 13, color: C.text },
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

  // full-screen dim layer; tapping anywhere on it (outside the dialog) dismisses
  overlay: { flex: 1, justifyContent: 'flex-end', backgroundColor: '#000000a6' },
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
  handleHit: { alignSelf: 'stretch', alignItems: 'center', justifyContent: 'center', height: 48 }, // big grab target
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

  modelBar: { height: 3, marginHorizontal: 12, marginBottom: 12, borderRadius: 2, backgroundColor: C.faint, overflow: 'hidden' },
  modelBarFill: { height: 3, borderRadius: 2, backgroundColor: C.amber },
});