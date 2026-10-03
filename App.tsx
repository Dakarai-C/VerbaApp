import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Animated,
  AppState,
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
import Sound from 'react-native-sound';
import { errorCodes, isErrorWithCode, keepLocalCopy, pick, types as pickerTypes } from '@react-native-documents/picker';
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
// Audio files whose transcript the user deleted while choosing to KEEP the audio. Without this list, the startup
// recovery below would find those files again and list them as "not transcribed" recordings.
const DETACHED_PATH = `${RNFS.DocumentDirectoryPath}/detached-audio.json`;

// ---- Audio import ----
// whisper.rn reliably reads WAV files. Compressed formats (MP3, M4A, AAC...) need a separate decoding step that
// this app does not have yet, so only these extensions are accepted. Add 'mp3' here to experiment.
const IMPORT_EXTENSIONS = ['wav'];
// Whisper expects 16 kHz mono 16-bit audio. While true, WAV files in any other layout are refused with a clear message.
// Set to false to let Whisper try them anyway (useful to find out whether the native decoder resamples).
const STRICT_WAV_FORMAT = true;
// Whisper loads the whole file into memory, which is risky on phones with little RAM. Conservative limit; tune on device.
const MAX_IMPORT_BYTES = 200 * 1000 * 1000;
const IMPORT_FREE_SPACE_MARGIN = 50 * 1000 * 1000; // keep this much free after copying

// Speech model: downloaded once from Hugging Face, then everything runs offline.
const MODEL_URL = 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin';
const MODEL_MIN_BYTES = 70000000; // ggml-tiny.en.bin is about 75 MB; anything smaller is a partial file

// Test clip for "Transcribe test file": downloaded on first use, then it works offline. 11 s, 16 kHz mono 16-bit, 352,078 bytes.
const SAMPLE_URL = 'https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav';
const SAMPLE_MIN_BYTES = 300000; // anything smaller is a partial or error page

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

// ---------- sound effects ----------
// Short clips bundled with the app in android/app/src/main/res/raw/. Use these names, in any format Android plays
// (mp3, ogg, wav); the extension is ignored. A clip that is missing is skipped silently, so the app works without them.
type SfxName = 'tap' | 'error' | 'record' | 'delete' | 'toggle';
const SFX_FILES: Record<SfxName, string> = {
  tap: 'sfx_tap',
  error: 'sfx_error',
  record: 'sfx_record', // plays when recording starts and again when it stops
  delete: 'sfx_delete',
  toggle: 'sfx_toggle', // plays when a switch is turned ON
};
const sfxBank: Partial<Record<SfxName, Sound>> = {};
let sfxEnabled = true; // the Settings switch
let sfxMuted = false; // true while the microphone is open, so effects never end up in a recording
let sfxStarted = false;

function sfxInit() {
  if (sfxStarted) return;
  sfxStarted = true;
  (Object.keys(SFX_FILES) as SfxName[]).forEach(name => {
    const snd: Sound = new Sound(SFX_FILES[name], Sound.MAIN_BUNDLE, (err: any) => {
      if (err) {
        console.warn(`Sound effect "${name}" not loaded: is ${SFX_FILES[name]} in android/app/src/main/res/raw?`);
        return;
      }
      sfxBank[name] = snd;
    });
  });
}

// Fire-and-forget for most effects. The returned promise resolves when the clip ends (or after 1.5 s at most), so the
// recording cue can be awaited. `force` plays even while muted; only the recording cues use it.
function playSfx(name: SfxName, force = false): Promise<void> {
  const snd = sfxBank[name];
  if (!snd || !sfxEnabled || (sfxMuted && !force)) return Promise.resolve();
  return new Promise<void>(resolve => {
    const t = setTimeout(resolve, 1500);
    try {
      snd.stop(() => snd.play(() => {
        clearTimeout(t);
        resolve();
      }));
    } catch {
      clearTimeout(t);
      resolve();
    }
  });
}

// Android system-bar inset (3-button / gesture nav). React Native core has no inset API on Android
// (its SafeAreaView is a no-op there), so use react-native-safe-area-context when it is installed.
// If it is not installed this quietly falls back to 0 and the app behaves exactly as before.
let SafeArea: any = null;
try {
  SafeArea = require('react-native-safe-area-context');
} catch {}
// App lock uses the system biometric prompt (fingerprint / face, with the phone's PIN or pattern as fallback).
// Loaded the same optional way: if the package is not installed, the lock switch explains that instead of crashing.
let Biometrics: any = null;
try {
  Biometrics = require('@sbaiahmed1/react-native-biometrics');
} catch {}
// Error text that means "this phone has no screen lock / enrolled biometrics to check against" (as opposed to the user
// cancelling or failing a scan). Heuristic on the library's message: tune here if the J8 reports it differently.
const NO_SECURITY_RE = /not.?enrolled|none.?enrolled|no.?(device.?)?credential|no.?biometric|not.?available|no.?hardware|passcode.?not.?set/i;
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
  source?: 'mic' | 'live' | 'sample' | 'import';
  status?: 'done' | 'failed' | 'untranscribed'; // missing = done
};

const countWords = (t: string) => (t ? t.split(/\s+/).length : 0);
const mapSegs = (segments?: any[]) =>
  segments?.length
    ? segments.map((g: any) => ({ t: (g.t0 ?? 0) / 100, text: String(g.text ?? '').trim() }))
    : undefined;

function fromBase64(b64: string): Uint8Array {
  const clean = b64.replace(/[^A-Za-z0-9+/]/g, '');
  const len = clean.length;
  const out = new Uint8Array(Math.floor((len * 3) / 4));
  const ix = (i: number) => (i < len ? B64.indexOf(clean[i]) : 0);
  let o = 0;
  for (let i = 0; i < len; i += 4) {
    const n = (ix(i) << 18) | (ix(i + 1) << 12) | (ix(i + 2) << 6) | ix(i + 3);
    out[o++] = (n >> 16) & 255;
    if (i + 2 < len) out[o++] = (n >> 8) & 255;
    if (i + 3 < len) out[o++] = n & 255;
  }
  return out;
}

type WavInfo = { format: number; channels: number; rate: number; bits: number; seconds: number };

// Reads the RIFF/WAVE header. Returns null if this is not a readable WAV file.
function parseWav(b: Uint8Array, fileSize: number): WavInfo | null {
  const tag = (o: number) => String.fromCharCode(b[o], b[o + 1], b[o + 2], b[o + 3]);
  if (b.length < 12 || tag(0) !== 'RIFF' || tag(8) !== 'WAVE') return null;
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  let fmtChunk: { format: number; channels: number; rate: number; byteRate: number; bits: number } | null = null;
  let o = 12;
  while (o + 8 <= b.length) {
    const id = tag(o);
    const size = dv.getUint32(o + 4, true);
    if (id === 'fmt ' && o + 24 <= b.length) {
      fmtChunk = {
        format: dv.getUint16(o + 8, true),
        channels: dv.getUint16(o + 10, true),
        rate: dv.getUint32(o + 12, true),
        byteRate: dv.getUint32(o + 16, true),
        bits: dv.getUint16(o + 22, true),
      };
    } else if (id === 'data') {
      if (!fmtChunk) return null;
      const avail = Math.max(0, fileSize - (o + 8));
      // streamed or truncated files can report 0 / 0xFFFFFFFF / too much: trust the real file size then
      const dataBytes = !size || size === 0xffffffff || size > avail ? avail : size;
      const byteRate = fmtChunk.byteRate || (fmtChunk.rate * fmtChunk.channels * fmtChunk.bits) / 8;
      return {
        format: fmtChunk.format,
        channels: fmtChunk.channels,
        rate: fmtChunk.rate,
        bits: fmtChunk.bits,
        seconds: byteRate > 0 ? dataBytes / byteRate : 0,
      };
    }
    o += 8 + size + (size & 1);
  }
  return null;
}

function friendlyFsError(e: any): string {
  const m = String(e?.message ?? e);
  return /ENOSPC|No space left/i.test(m)
    ? 'Not enough free storage to save this recording. Free up some space and try again.'
    : m;
}

// Turns a failed download into a message a person can act on. The raw error is only ever logged, never shown.
// No network-status library: when the phone is offline the DNS lookup itself fails, and that is the signal.
// The patterns cover the Android/Java wording RNFS passes through, plus the errors thrown by downloadModel itself.
function friendlyNetError(e: any, what = 'speech model', size = 'about 75 MB'): string {
  const m = String(e?.message ?? e);
  if (/ENOSPC|No space left/i.test(m)) {
    return `Not enough free storage for the ${what} (${size}). Free up some space and try again.`;
  }
  if (/UnknownHost|resolve host|ENOTFOUND|EAI_AGAIN|nodename nor servname|No address associated/i.test(m)) {
    return `You appear to be offline. Connect to Wi-Fi or mobile data to download the ${what}. Everything else in Viva Voce works offline.`;
  }
  if (/time(d)?[ -]?out/i.test(m)) {
    return 'The connection timed out. Check your signal and try again, ideally on Wi-Fi.';
  }
  if (/SSL|certificate|handshake/i.test(m)) {
    return 'A secure connection could not be made. Check the phone\'s date and time, and that you are not on a Wi-Fi network that needs a sign-in page.';
  }
  const status = /Server answered (\d+)/.exec(m);
  if (status) {
    const code = Number(status[1]);
    if (code === 404 || code === 403 || code === 410) {
      return `The ${what} could not be found on the download server. The link may have changed.`;
    }
    if (code === 429 || code >= 500) return 'The download server is busy or unavailable. Please try again later.';
    return `The download server refused the request (code ${code}). Please try again later.`;
  }
  if (/incomplete|unexpected end|Connection (closed|reset|abort)|ECONNRESET|Broken pipe|Software caused/i.test(m)) {
    return 'The download was interrupted before it finished. Check your connection and try again.';
  }
  if (/ConnectException|Failed to connect|ECONNREFUSED|Unreachable|Network request failed|SocketException/i.test(m)) {
    return 'Could not reach the download server. Check your Wi-Fi or mobile data and try again.';
  }
  return 'The download failed. Please try again.';
}

// An entry's audioFile must be a bare file name. Anything with a path in it is never touched.
const isPlainFileName = (n: unknown): n is string =>
  typeof n === 'string' && n.length > 0 && !/[\\/]/.test(n) && n !== '.' && n !== '..';

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

function MenuRow(p: { icon: string; title: string; sub: string; onPress: () => void; right?: React.ReactNode; silent?: boolean }) {
  return (
    <Pressable
      onPress={() => {
        if (!p.silent) playSfx('tap'); // switches and the record row have their own sounds
        p.onPress();
      }} style={({ pressed }) => [s.menuRow, pressed && { backgroundColor: '#ffffff0d' }]}>
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

// ---------- audio playback UI (styled after the Figma Make "Offline audio transcription" reference) ----------
// Figma's card play button: 32px round, faint outline, muted icon; amber fill with a canvas-coloured icon while playing;
// 25% opacity when there is no audio.
type PlayerState = { id: string | null; status: 'idle' | 'loading' | 'playing' | 'paused'; pos: number; dur: number };

function PlayButton(p: { playing: boolean; loading: boolean; disabled: boolean; onPress: () => void; label: string }) {
  const fg = p.playing || p.loading ? C.canvas : C.muted;
  return (
    <Pressable
      onPress={() => {
        playSfx('tap');
        p.onPress();
      }}
      disabled={p.disabled}
      hitSlop={8}
      accessibilityRole="button"
      accessibilityLabel={p.label}
      accessibilityState={{ disabled: p.disabled }}
      style={({ pressed }) => [
        s.playBtn,
        {
          backgroundColor: p.playing || p.loading ? C.amber : 'transparent',
          borderColor: p.playing || p.loading ? C.amber : C.faint,
          opacity: p.disabled ? 0.25 : 1,
          transform: [{ scale: pressed ? 0.95 : 1 }],
        },
      ]}
    >
      {p.loading ? (
        <ActivityIndicator size="small" color={fg} />
      ) : p.playing ? (
        <View style={{ flexDirection: 'row', gap: 4 }}>
          <View style={{ width: 4, height: 12, borderRadius: 2, backgroundColor: fg }} />
          <View style={{ width: 4, height: 12, borderRadius: 2, backgroundColor: fg }} />
        </View>
      ) : (
        // play triangle drawn with borders (no SVG library in the project)
        <View
          style={{
            marginLeft: 2,
            borderLeftWidth: 7,
            borderTopWidth: 4.5,
            borderBottomWidth: 4.5,
            borderLeftColor: fg,
            borderTopColor: 'transparent',
            borderBottomColor: 'transparent',
          }}
        />
      )}
    </Pressable>
  );
}

const clamp01 = (n: number) => Math.max(0, Math.min(1, n));

// Progress bar you can tap or drag. The seek happens when the finger lifts, so the audio is not re-seeked on every move.
function SeekBar(p: { pos: number; dur: number; onSeek: (sec: number) => void }) {
  const [scrub, setScrub] = useState<number | null>(null);
  const [w, setW] = useState(0);
  const wRef = useRef(0);
  const durRef = useRef(p.dur);
  durRef.current = p.dur;
  const onSeekRef = useRef(p.onSeek);
  onSeekRef.current = p.onSeek;
  const startRatio = useRef(0);
  const ratioRef = useRef(0);

  const pan = useMemo(
    () =>
      PanResponder.create({
        onStartShouldSetPanResponder: () => durRef.current > 0,
        onMoveShouldSetPanResponder: () => durRef.current > 0,
        onPanResponderTerminationRequest: () => false, // keep the gesture even if a ScrollView wants it
        onPanResponderGrant: e => {
          const r = clamp01(e.nativeEvent.locationX / (wRef.current || 1));
          startRatio.current = r;
          ratioRef.current = r;
          setScrub(r * durRef.current);
        },
        onPanResponderMove: (_, g) => {
          const r = clamp01(startRatio.current + g.dx / (wRef.current || 1));
          ratioRef.current = r;
          setScrub(r * durRef.current);
        },
        onPanResponderRelease: () => {
          onSeekRef.current(ratioRef.current * durRef.current);
          setScrub(null);
        },
        onPanResponderTerminate: () => setScrub(null),
      }),
    [],
  );

  const shown = scrub ?? p.pos;
  const ratio = p.dur > 0 ? clamp01(shown / p.dur) : 0;
  return (
    <View style={{ flex: 1 }}>
      {/* the touch target is the whole 28px-high row; children ignore touches so locationX stays relative to it */}
      <View
        {...pan.panHandlers}
        style={s.seekHit}
        onLayout={e => {
          wRef.current = e.nativeEvent.layout.width;
          setW(e.nativeEvent.layout.width);
        }}
        accessible
        accessibilityRole="adjustable"
        accessibilityLabel="Seek"
        accessibilityValue={{ min: 0, max: Math.round(p.dur), now: Math.round(shown) }}
      >
        <View style={s.seekTrack} pointerEvents="none">
          <View style={{ width: `${ratio * 100}%`, height: 4, borderRadius: 2, backgroundColor: C.amber }} />
        </View>
        <View
          pointerEvents="none"
          style={[s.seekThumb, { left: Math.max(0, Math.min(w - 12, ratio * w - 6)), opacity: p.dur > 0 ? 1 : 0.4 }]}
        />
      </View>
      <View style={s.seekTimes}>
        <Text style={[s.mono9, { color: C.amber }]}>{fmt(shown)}</Text>
        <Text style={s.mono9}>{fmt(p.dur)}</Text>
      </View>
    </View>
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
  const [sfxOn, setSfxOn] = useState(true); // Settings > Sound effects
  const startingRef = useRef(false); // a recording is about to start (the start cue is still playing)
  const [deleting, setDeleting] = useState<Entry | null>(null); // entry the delete dialog is open for
  const [deleteBusy, setDeleteBusy] = useState(false);
  const deleteBusyRef = useRef(false);
  // ---- app lock ----
  const [appLock, setAppLock] = useState(false); // the setting
  const [locked, setLocked] = useState(true); // starts locked so nothing flashes before the setting is read
  const [lockMsg, setLockMsg] = useState('');
  const appLockRef = useRef(false);
  appLockRef.current = appLock;
  const lockedRef = useRef(true);
  lockedRef.current = locked;
  const authBusyRef = useRef(false);
  // System screens we open ourselves (file picker, share sheet, permission prompt, the biometric prompt) send the app to
  // the background for a moment. That must not count as "the user left the app".
  const suppressLockRef = useRef(false);
  const withoutLock = async <T,>(fn: () => Promise<T>): Promise<T> => {
    suppressLockRef.current = true;
    try {
      return await fn();
    } finally {
      setTimeout(() => {
        suppressLockRef.current = false;
      }, 600);
    }
  };
  // Highest "untitled-NNNN" number ever handed out. Saved in settings.json so deleting entries never frees a number.
  const [nameCounter, setNameCounter] = useState(0);
  const nameCounterRef = useRef(0);
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

  useEffect(() => {
    sfxInit();
  }, []);
  useEffect(() => {
    sfxEnabled = sfxOn;
  }, [sfxOn]);
  useEffect(() => {
    sfxMuted = recording;
  }, [recording]);
  useEffect(() => {
    if (status.startsWith('Error')) playSfx('error');
  }, [status]);

  const downloadJobRef = useRef<number | null>(null);
  const cancelledRef = useRef(false);

  // ---------- audio playback ----------
  // One shared player: playing a different recording replaces the current one. The Sound object lives in a ref;
  // `player` is just what the UI shows.
  const [player, setPlayer] = useState<PlayerState>({ id: null, status: 'idle', pos: 0, dur: 0 });
  const playerRef = useRef(player);
  playerRef.current = player;
  const soundRef = useRef<Sound | null>(null);
  const loadTokenRef = useRef(0); // lets a slow load be cancelled by a newer tap
  const lastSeekRef = useRef(0);

  const releaseSound = () => {
    const snd = soundRef.current;
    soundRef.current = null;
    if (snd) {
      try {
        snd.release();
      } catch {}
    }
  };

  // stop and unload whatever is playing (also used before recording and, later, before deleting)
  const stopPlayback = () => {
    loadTokenRef.current++;
    releaseSound();
    setPlayer({ id: null, status: 'idle', pos: 0, dur: 0 });
  };

  const playbackFailed = (why: string) => {
    console.warn('Playback failed:', why);
    playSfx('error');
    stopPlayback();
    Alert.alert('Audio unavailable', 'The audio for this recording is missing or cannot be played.');
  };

  const startPlayback = () => {
    const snd = soundRef.current;
    if (!snd) return;
    setPlayer(p => ({ ...p, status: 'playing' }));
    snd.play(ok => {
      // called once when playback reaches the end (or fails)
      if (soundRef.current !== snd) return;
      if (ok) {
        snd.setCurrentTime(0);
        setPlayer(p => ({ ...p, status: 'paused', pos: 0 }));
      } else {
        playbackFailed('player reported an error');
      }
    });
  };

  const togglePlay = async (e: Entry) => {
    if (!e.audioFile) return;
    const cur = playerRef.current;
    if (cur.id === e.id && soundRef.current) {
      if (cur.status === 'playing') {
        soundRef.current.pause();
        setPlayer(p => ({ ...p, status: 'paused' }));
      } else if (cur.status === 'paused') {
        startPlayback();
      }
      return;
    }
    if (cur.id === e.id && cur.status === 'loading') return;

    const token = ++loadTokenRef.current;
    releaseSound();
    setPlayer({ id: e.id, status: 'loading', pos: 0, dur: 0 });
    const path = `${AUDIO_DIR}/${e.audioFile}`;
    let exists = false;
    try {
      exists = await RNFS.exists(path);
    } catch {}
    if (token !== loadTokenRef.current) return;
    if (!exists) {
      playbackFailed(`file not found: ${path}`);
      return;
    }
    // '' as basePath: the path is already absolute
    const snd: Sound = new Sound(path, '', (err: any) => {
      if (token !== loadTokenRef.current) {
        snd.release();
        return;
      }
      if (err) {
        snd.release();
        playbackFailed(`could not load: ${JSON.stringify(err)}`);
        return;
      }
      soundRef.current = snd;
      setPlayer({ id: e.id, status: 'paused', pos: 0, dur: Math.max(0, snd.getDuration()) });
      startPlayback();
    });
  };

  const seekTo = (sec: number) => {
    const snd = soundRef.current;
    if (!snd) return;
    const d = playerRef.current.dur;
    const t = Math.max(0, d > 0 ? Math.min(sec, d - 0.05) : sec);
    lastSeekRef.current = Date.now();
    snd.setCurrentTime(t);
    setPlayer(p => ({ ...p, pos: t }));
  };

  // progress: ask the native player for its position a few times a second while playing
  useEffect(() => {
    if (player.status !== 'playing') return;
    const id = setInterval(() => {
      const snd = soundRef.current;
      if (!snd) return;
      snd.getCurrentTime((t: number) => {
        if (soundRef.current !== snd || Date.now() - lastSeekRef.current < 500) return; // ignore a stale reading right after a seek
        setPlayer(p => (p.status === 'playing' ? { ...p, pos: t } : p));
      });
    }, 250);
    return () => clearInterval(id);
  }, [player.status]);

  // pause when the app goes to the background; release the player when the app screen unmounts
  useEffect(() => {
    const sub = AppState.addEventListener('change', st => {
      if (st !== 'active' && playerRef.current.status === 'playing') {
        soundRef.current?.pause();
        setPlayer(p => ({ ...p, status: 'paused' }));
      }
    });
    return () => {
      sub.remove();
      loadTokenRef.current++;
      releaseSound();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
    playSfx('tap');
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
          { text: 'Delete', style: 'destructive', onPress: () => setDeleting(e) },
          { text: 'Transcribe', onPress: () => retryTranscription(e) },
        ],
      );
      return;
    }
    setSelected(e);
  };

  // next free "untitled-0001" style name; the number only ever goes up, even if entries are renamed or deleted
  const nextNumber = () => {
    const maxUsed = entriesRef.current.reduce((m, e) => {
      const x = /^untitled-(\d+)$/.exec(e.title);
      return x ? Math.max(m, Number(x[1])) : m;
    }, 0);
    return Math.max(maxUsed, nameCounterRef.current) + 1;
  };
  const nameFor = (n: number) => `untitled-${String(n).padStart(4, '0')}`;
  const nextName = () => nameFor(nextNumber());

  const addEntry = (raw: string, secs: number, title?: string, segments?: any[], extra: Partial<Entry> = {}) => {
    const t = raw.trim();
    // nothing to show: no speech and no audio kept
    if (!t && !extra.audioFile) {
      setStatus('Done: no speech detected');
      return;
    }
    let finalTitle = title;
    if (finalTitle === undefined) {
      const n = nextNumber();
      nameCounterRef.current = n; // reserved for good, even if this entry is deleted later
      setNameCounter(n);
      finalTitle = nameFor(n);
    }
    const entry: Entry = {
      id: String(Date.now()),
      title: finalTitle,
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
    const perm = await withoutLock(() => PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.RECORD_AUDIO));
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
        console.warn('Model download failed:', e);
        setModelError(friendlyNetError(e));
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
      let counterBase = 0;
      let lockOn = false;
      try {
        let list: Entry[] = [];
        let detached: string[] = [];
        try {
          if (await RNFS.exists(DETACHED_PATH)) {
            const d = JSON.parse(await RNFS.readFile(DETACHED_PATH, 'utf8'));
            if (Array.isArray(d)) detached = d.filter((x: any) => typeof x === 'string');
          }
        } catch {}
        if (await RNFS.exists(ENTRIES_PATH)) {
          const data = JSON.parse(await RNFS.readFile(ENTRIES_PATH, 'utf8'));
          if (Array.isArray(data)) list = data;
        }
        // Audio files that no transcript points to (the app was closed mid-transcription) are never deleted:
        // they come back as "not transcribed" entries so the user can transcribe or remove them.
        try {
          await RNFS.mkdir(AUDIO_DIR);
          const known = new Set([...list.map(e => e.audioFile).filter(Boolean), ...detached]);
          const found = (await RNFS.readDir(AUDIO_DIR)).filter(f => f.isFile() && /\.wav$/i.test(f.name) && !known.has(f.name));
          const recovered: Entry[] = found.map((f, i) => ({
            id: `${f.name.replace(/\.wav$/i, '')}-${i}`,
            title: `recovered-${f.name.replace(/^(rec|imp)-/, '').replace(/\.wav$/i, '')}`,
            date: (f.mtime ?? new Date()).toISOString().slice(0, 10),
            duration: fmt(Math.max(0, (Number(f.size) - 44) / BYTES_PER_SEC)),
            words: 0,
            text: '',
            audioFile: f.name,
            source: f.name.startsWith('imp-') ? 'import' : 'mic',
            status: 'untranscribed',
          }));
          list = [...recovered, ...list];
        } catch {}
        setEntries(list);
        // start the name counter above every number already used (covers installs from before the counter existed)
        counterBase = list.reduce((m, e) => {
          const x = /^untitled-(\d+)$/.exec(e.title);
          return x ? Math.max(m, Number(x[1])) : m;
        }, list.length);
      } catch {}
      try {
        if (await RNFS.exists(SETTINGS_PATH)) {
          const st = JSON.parse(await RNFS.readFile(SETTINGS_PATH, 'utf8'));
          if (typeof st.sfx === 'boolean') setSfxOn(st.sfx);
          if (typeof st.appLock === 'boolean') {
            lockOn = st.appLock;
            setAppLock(st.appLock);
          }
          if (typeof st.nameCounter === 'number' && st.nameCounter > counterBase) counterBase = st.nameCounter;
          if (typeof st.deleteAudio === 'boolean') setDeleteAudio(st.deleteAudio);
          if (typeof st.showTimestamps === 'boolean') setShowTimestamps(st.showTimestamps);
          if (typeof st.batchMode === 'boolean') setBatchMode(st.batchMode);
        }
      } catch {}
      nameCounterRef.current = counterBase;
      setNameCounter(counterBase);
      setLocked(lockOn && !!Biometrics);
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
    RNFS.writeFile(SETTINGS_PATH, JSON.stringify({ deleteAudio, showTimestamps, batchMode, nameCounter, appLock, sfx: sfxOn }), 'utf8').catch(() => {});
  }, [deleteAudio, showTimestamps, batchMode, nameCounter, appLock, sfxOn, loaded]);

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
  // Downloads the test clip to a temporary name and only moves it into place once it checks out as a real WAV.
  const downloadSample = async (): Promise<boolean> => {
    setLoading(true);
    setStatus('Downloading test file...');
    const part = `${AUDIO_PATH}.part`;
    try {
      await RNFS.unlink(part).catch(() => {});
      const res = await RNFS.downloadFile({ fromUrl: SAMPLE_URL, toFile: part }).promise;
      if (res.statusCode !== 200) throw new Error(`Server answered ${res.statusCode}`);
      const size = Number((await RNFS.stat(part)).size);
      if (size < SAMPLE_MIN_BYTES) throw new Error('Download was incomplete');
      const info = parseWav(fromBase64(await RNFS.read(part, 32768, 0, 'base64')), size);
      if (!info || info.rate !== SAMPLE_RATE || info.channels !== 1 || info.bits !== 16) {
        throw new Error('Download was incomplete');
      }
      if (await RNFS.exists(AUDIO_PATH)) await RNFS.unlink(AUDIO_PATH);
      await RNFS.moveFile(part, AUDIO_PATH);
      return true;
    } catch (e: any) {
      RNFS.unlink(part).catch(() => {});
      console.warn('Sample download failed:', e);
      setStatus(`Error: ${friendlyNetError(e, 'test file', 'about 350 KB')}`);
      return false;
    } finally {
      setLoading(false);
    }
  };

  const transcribeFile = async () => {
    let haveSample = false;
    try {
      // no point downloading a clip the app cannot transcribe yet
      if (!ctxRef.current && !(await RNFS.exists(MODEL_PATH))) {
        setStatus('Error: Speech model not downloaded. Open Settings > Speech model to get it.');
        return;
      }
      haveSample = await RNFS.exists(AUDIO_PATH);
    } catch {}
    if (!haveSample) {
      Alert.alert(
        'Download test file?',
        'The test file is a short JFK speech clip (about 350 KB). It downloads once; after that the test works offline.',
        [
          { text: 'Cancel', style: 'cancel' },
          {
            text: 'Download',
            onPress: async () => {
              if (await downloadSample()) transcribeFile();
            },
          },
        ],
      );
      return;
    }
    setLoading(true);
    try {
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
    playSfx('record', true);
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
      playSfx('record', true); // the microphone is closed now, so this cannot be recorded

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

  // ---------- import audio ----------

  // Transcribes a file that is already inside AUDIO_DIR and applies the retention policy afterwards.
  // Same rules as a microphone recording: nothing is deleted before transcription succeeds, and a failure keeps the
  // audio and lists it in Recent as "not transcribed" so it can be retried.
  const transcribeAndStore = async (
    id: string,
    audioFile: string,
    seconds: number,
    source: Entry['source'],
    title: string | undefined,
    discardAfter: boolean,
  ) => {
    const path = `${AUDIO_DIR}/${audioFile}`;
    try {
      const ctx = await getContext();
      setStatus(`Transcribing ${seconds.toFixed(0)}s of audio...`);
      const start = Date.now();
      const { promise } = ctx.transcribe(path, { language: 'en' });
      const out: any = await promise;
      if (discardAfter) await RNFS.unlink(path).catch(() => {});
      addEntry(out.result ?? '', seconds, title, out.segments, {
        id,
        source,
        audioFile: discardAfter ? undefined : audioFile,
        status: 'done',
      });
      setStatus(`Done: ${seconds.toFixed(0)}s of audio transcribed in ${((Date.now() - start) / 1000).toFixed(1)}s`);
    } catch (e: any) {
      addEntry('', seconds, title, undefined, { id, source, audioFile, status: 'failed' });
      setStatus(`Error: ${e?.message ?? String(e)} Your audio was saved; tap it in Recent to try again.`);
    }
  };

  const importAudio = async () => {
    if (recording || loading) {
      setStatus('Finish the current recording or transcription first');
      return;
    }

    // 1. Android's system file picker. Cancelling is not an error.
    let picked: Awaited<ReturnType<typeof pick>>[number];
    try {
      [picked] = await withoutLock(() => pick({ type: [pickerTypes.audio], allowMultiSelection: false }));
    } catch (e: any) {
      if (isErrorWithCode(e) && (e.code === errorCodes.OPERATION_CANCELED || e.code === errorCodes.IN_PROGRESS)) return;
      console.warn('Import: file picker failed', e);
      setStatus('Error: Could not open the file picker. Please try again.');
      return;
    }

    const originalName = picked.name ?? 'audio';
    const ext = (originalName.split('.').pop() ?? '').toLowerCase();
    const mime = (picked.type ?? '').toLowerCase();
    const looksSupported = IMPORT_EXTENSIONS.includes(ext) || (IMPORT_EXTENSIONS.includes('wav') && /wav/.test(mime));
    if (!looksSupported) {
      setStatus(
        `Error: "${originalName}" is not a supported format. Viva Voce can import ${IMPORT_EXTENSIONS.map(x => x.toUpperCase()).join(', ')} files for now. Convert other audio to WAV first.`,
      );
      return;
    }
    if (picked.size && picked.size > MAX_IMPORT_BYTES) {
      setStatus(`Error: This file is too large (${(picked.size / 1e6).toFixed(0)} MB). Imports are limited to ${(MAX_IMPORT_BYTES / 1e6).toFixed(0)} MB for now.`);
      return;
    }

    setLoading(true);
    const discardAfter = deleteAudioRef.current; // retention policy, read once at the start
    const id = String(Date.now());
    let tempDir: string | null = null;
    let finalPath: string | null = null;
    let stored = false;
    try {
      // 2. enough free space for the copy?
      if (picked.size) {
        try {
          const info = await RNFS.getFSInfo();
          if (info.freeSpace < picked.size + IMPORT_FREE_SPACE_MARGIN) {
            throw new Error(
              `Not enough free storage to import this file (about ${(picked.size / 1e6).toFixed(0)} MB needed). Free up some space and try again.`,
            );
          }
        } catch (e: any) {
          if (String(e?.message).startsWith('Not enough')) throw e;
        }
      }

      // 3. copy into the app's private storage (the picker copies natively, so big files don't pass through JS)
      setStatus('Importing audio...');
      const [copy] = await keepLocalCopy({
        files: [{ uri: picked.uri, fileName: originalName }],
        destination: 'documentDirectory',
      });
      if (copy.status !== 'success') {
        console.warn('Import: copy failed', copy.copyError);
        const reason = friendlyFsError(copy.copyError);
        throw new Error(reason !== String(copy.copyError) ? reason : 'Could not read that file. It may have been moved or deleted.');
      }
      const tempPath = decodeURIComponent(copy.localUri.replace(/^file:\/\//, ''));
      const dir = tempPath.substring(0, tempPath.lastIndexOf('/'));
      if (dir.startsWith(`${RNFS.DocumentDirectoryPath}/`)) tempDir = dir; // the picker's own temporary folder

      // 4. check it really is a usable WAV file
      setStatus('Checking audio...');
      const size = Number((await RNFS.stat(tempPath)).size);
      if (size > MAX_IMPORT_BYTES) throw new Error(`This file is too large (${(size / 1e6).toFixed(0)} MB).`);
      let info: WavInfo | null = null;
      try {
        info = parseWav(fromBase64(await RNFS.read(tempPath, 32768, 0, 'base64')), size);
      } catch {}
      if (!info) throw new Error(`"${originalName}" is not a readable WAV file.`);
      if (info.format !== 1 || info.bits !== 16) {
        throw new Error('Only standard 16-bit WAV files can be imported for now.');
      }
      if (STRICT_WAV_FORMAT && (info.rate !== 16000 || info.channels !== 1)) {
        throw new Error(
          `This WAV is ${(info.rate / 1000).toString()} kHz ${info.channels === 1 ? 'mono' : info.channels === 2 ? 'stereo' : `${info.channels}-channel`}. For now Viva Voce needs 16 kHz mono WAV, like its own recordings.`,
        );
      }
      if (info.seconds < 0.5) throw new Error('This audio file is too short to transcribe.');

      // 5. move into the audio folder under a unique name
      const audioFile = `imp-${id}.wav`;
      finalPath = `${AUDIO_DIR}/${audioFile}`;
      await RNFS.mkdir(AUDIO_DIR);
      await RNFS.moveFile(tempPath, finalPath);
      stored = true;
      if (tempDir) await RNFS.unlink(tempDir).catch(() => {});
      tempDir = null;

      // 6. same pipeline as a recording
      const title = originalName.replace(/\.[^.]+$/, '').trim().slice(0, 60) || undefined;
      await transcribeAndStore(id, audioFile, info.seconds, 'import', title, discardAfter);
    } catch (e: any) {
      if (!stored && finalPath) RNFS.unlink(finalPath).catch(() => {});
      console.warn('Import failed', e);
      setStatus(`Error: ${friendlyFsError(e)}`);
    } finally {
      // nothing the user imported is left half-copied in the app folder
      if (tempDir) RNFS.unlink(tempDir).catch(() => {});
      setLoading(false);
    }
  };

  // ---------- record button ----------
  const toggleRecording = async () => {
    if (recording) {
      if (activeModeRef.current === 'batch') await stopBatch();
      else await stopLive();
      return;
    }
    if (startingRef.current) return; // a second tap while the start cue is playing
    startingRef.current = true;
    try {
      stopPlayback(); // the microphone and playback should not compete
      if (!(await ensureMic())) return;
      setText('');
      activeModeRef.current = batchMode ? 'batch' : 'live';
      await playSfx('record', true); // the cue finishes BEFORE the microphone opens, so it is not recorded
      if (batchMode) await startBatch();
      else await startLive();
    } finally {
      startingRef.current = false;
    }
  };

  const exportEntry = (e: Entry) =>
    withoutLock(() => Share.share({ message: `${e.title}\n${e.date} · ${e.duration}\n\n${e.text}` }));

  const confirmRename = (keep: boolean) => {
    if (!renaming) return;
    const name = keep ? renaming.title : draftName.trim() || renaming.title;
    const updated = { ...renaming, title: name };
    setEntries(cur => cur.map(e => (e.id === updated.id ? updated : e)));
    setRenaming(null);
    setSelected(updated);
  };

  const toggleSfx = (on: boolean) => {
    setSfxOn(on);
    if (on) {
      sfxEnabled = true; // the effect that syncs this runs after render; the confirmation should play now
      playSfx('toggle');
    }
  };

  // ---------- app lock ----------
  // 'ok' = the user proved who they are; 'unavailable' = this phone has nothing to check against; 'failed' = cancelled / wrong.
  const runAuth = async (subtitle: string): Promise<'ok' | 'failed' | 'unavailable'> => {
    if (!Biometrics) return 'unavailable';
    try {
      const r: any = await withoutLock<any>(() =>
        Biometrics.authenticateWithOptions({
          title: 'Viva Voce',
          subtitle,
          cancelLabel: 'Cancel',
          fallbackLabel: 'Use PIN',
          allowDeviceCredentials: true, // fingerprint/face first, phone PIN/pattern as the fallback
          disableDeviceFallback: false,
        }),
      );
      if (r?.success) return 'ok';
      console.warn('App lock: not authenticated', r?.error, r?.errorCode);
      return NO_SECURITY_RE.test(String(r?.error ?? '')) ? 'unavailable' : 'failed';
    } catch (err: any) {
      console.warn('App lock: prompt error', err);
      return NO_SECURITY_RE.test(String(err?.message ?? err)) ? 'unavailable' : 'failed';
    }
  };

  const lockNow = () => {
    // dialogs and sheets are separate windows on Android, so close them all rather than rely on covering them
    setSelected(null);
    setSettingsOpen(false);
    setLicensesOpen(false);
    setMenuOpen(false);
    setDeleting(null);
    setRenaming(null);
    setSearchOpen(false);
    setQuery('');
    setLockMsg('');
    setLocked(true);
  };

  const tryUnlock = async () => {
    if (authBusyRef.current || !lockedRef.current || AppState.currentState !== 'active') return;
    authBusyRef.current = true;
    setLockMsg('');
    try {
      const res = await runAuth('Unlock to see your transcripts');
      if (res === 'ok') {
        setLocked(false);
      } else if (res === 'unavailable') {
        // The phone no longer has a screen lock / fingerprint to check against. Staying locked would lock the user out
        // of their own transcripts for good, and without a phone lock the app lock protects nothing anyway.
        setAppLock(false);
        setLocked(false);
        setStatus('App lock turned off: this phone has no screen lock set up.');
      } else {
        setLockMsg('Not unlocked yet. Tap Unlock to try again.');
      }
    } finally {
      authBusyRef.current = false;
    }
  };

  const toggleAppLock = async (on: boolean) => {
    if (!on) {
      setAppLock(false);
      return;
    }
    if (!Biometrics) {
      Alert.alert('App lock unavailable', 'This build does not include the app-lock component yet.');
      return;
    }
    // prove it works before turning it on, so a broken setup can never lock the user out
    const res = await runAuth('Confirm to turn on app lock');
    if (res === 'ok') {
      setAppLock(true);
      playSfx('toggle');
    }
    else if (res === 'unavailable')
      Alert.alert('Set up a screen lock first', 'Add a fingerprint, PIN or pattern in your phone settings, then try again.');
  };

  // lock when the app goes to the background; prompt again when it comes back
  useEffect(() => {
    const sub = AppState.addEventListener('change', st => {
      if (st === 'background' && appLockRef.current && !suppressLockRef.current && !authBusyRef.current) {
        lockNow();
      } else if (st === 'active' && lockedRef.current && appLockRef.current) {
        tryUnlock();
      }
    });
    return () => sub.remove();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // cold start: once settings are read, either open straight away or ask for authentication
  useEffect(() => {
    if (!loaded || !locked) return;
    if (!appLock || !Biometrics) setLocked(false);
    else tryUnlock();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loaded, locked]);

  // ---------- delete ----------
  // Removes an audio file that an entry explicitly points to. A file that is already gone counts as deleted.
  const removeAudioFile = async (file: string) => {
    const path = `${AUDIO_DIR}/${file}`;
    if (!(await RNFS.exists(path))) return;
    try {
      await RNFS.unlink(path);
    } catch (err) {
      if (await RNFS.exists(path).catch(() => true)) throw err; // still there: a real failure
    }
  };

  // withAudio=false: remove the transcript, keep the audio file. withAudio=true: remove both.
  // Order matters: playback stops first; then the saved lists are updated (if that fails nothing has been touched yet);
  // only then is the audio file removed. Only the file named by this entry's own audioFile is ever deleted.
  const deleteEntry = async (target: Entry, withAudio: boolean) => {
    if (deleteBusyRef.current) return;
    deleteBusyRef.current = true;
    setDeleteBusy(true);
    const file = isPlainFileName(target.audioFile) ? target.audioFile : undefined;
    let audioProblem = false;
    try {
      if (playerRef.current.id === target.id) stopPlayback(); // also cancels a load that is still in progress

      const remaining = entriesRef.current.filter(x => x.id !== target.id);
      try {
        if (file && !withAudio) {
          // remember that this audio was kept on purpose, so startup recovery does not list it again
          let detached: string[] = [];
          try {
            if (await RNFS.exists(DETACHED_PATH)) {
              const d = JSON.parse(await RNFS.readFile(DETACHED_PATH, 'utf8'));
              if (Array.isArray(d)) detached = d.filter((x: any) => typeof x === 'string');
            }
          } catch {}
          if (!detached.includes(file)) {
            await RNFS.writeFile(DETACHED_PATH, JSON.stringify([...detached, file]), 'utf8');
          }
        }
        await RNFS.writeFile(ENTRIES_PATH, JSON.stringify(remaining), 'utf8');
      } catch (err) {
        console.warn('Delete: could not save the updated list', err);
        setDeleting(null);
        playSfx('error');
        Alert.alert('Could not delete', 'Your transcript was not deleted. Please try again.');
        return;
      }

      if (file && withAudio) {
        try {
          await removeAudioFile(file);
        } catch (err) {
          console.warn('Delete: could not remove audio file', file, err);
          audioProblem = true;
        }
      }

      setEntries(cur => cur.filter(x => x.id !== target.id));
      setSelected(cur => (cur?.id === target.id ? null : cur));
      setRenaming(cur => (cur?.id === target.id ? null : cur));
      if (remaining.length === 0) {
        setSearchOpen(false);
        setQuery('');
      }
      setDeleting(null);
      setStatus(`Deleted "${target.title}"`);
      playSfx('delete');
      if (audioProblem) {
        Alert.alert(
          'Transcript deleted',
          'The transcript was deleted, but its audio file could not be removed. It may show up in Recent as a recording that is not transcribed, where you can try deleting it again.',
        );
      }
    } finally {
      deleteBusyRef.current = false;
      setDeleteBusy(false);
    }
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
                <Pressable onPress={() => { playSfx('tap'); setQuery(''); }} hitSlop={10} accessibilityLabel="Clear search">
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
            {visible.map(e => {
              const active = player.id === e.id && player.status !== 'idle';
              const playing = active && player.status === 'playing';
              return (
                <View key={e.id} style={s.card}>
                  <View style={{ flexDirection: 'row', alignItems: 'center', gap: 12 }}>
                    <Pressable
                      onPress={() => {
                        playSfx('tap');
                        openEntry(e);
                      }}
                      style={({ pressed }) => [{ flex: 1, minWidth: 0 }, pressed && { opacity: 0.7 }]}
                    >
                      <Text style={s.cardTitle} numberOfLines={1}>
                        {e.title}
                      </Text>
                      <Text style={[s.mono9, { marginTop: 6 }]}>
                        {e.status === 'failed' || e.status === 'untranscribed'
                          ? `${e.duration} · NOT TRANSCRIBED`
                          : e.words === 0
                          ? `${e.duration} · NO SPEECH DETECTED`
                          : `${e.duration} · ${e.words.toLocaleString()} WORDS`}
                      </Text>
                      {!!q && !e.title.toLowerCase().includes(q) && (
                        <Text style={[s.mono9, { marginTop: 6, color: C.text }]} numberOfLines={2}>
                          {snippetFor(e.text, q)}
                        </Text>
                      )}
                    </Pressable>
                    <PlayButton
                      playing={playing}
                      loading={active && player.status === 'loading'}
                      disabled={!e.audioFile}
                      onPress={() => togglePlay(e)}
                      label={playing ? `Pause ${e.title}` : `Play ${e.title}`}
                    />
                  </View>
                  {active && player.status !== 'loading' && (
                    <View style={s.cardPlayer}>
                      <SeekBar pos={player.pos} dur={player.dur} onSeek={seekTo} />
                    </View>
                  )}
                </View>
              );
            })}
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
        <Pressable
          onPress={() => {
            playSfx('tap');
            setMenuOpen(true);
          }}
          accessibilityLabel="Add audio"
          style={s.fab}
        >
          <Text style={{ color: C.amber, fontSize: 28, lineHeight: 30 }}>+</Text>
        </Pressable>
        <Pressable
          style={s.navItem}
          onPress={() => {
            playSfx('tap');
            setSettingsOpen(true);
          }}
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
                icon="⇪"
                title="Import audio"
                sub="WAV FILE FROM YOUR PHONE"
                onPress={() => {
                  setMenuOpen(false);
                  importAudio();
                }}
              />
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
                silent
                onPress={() => {
                  if (recording || loading) return;
                  if (batchMode) playSfx('toggle'); // batch -> real-time means the switch is going ON
                  setBatchMode(v => !v);
                }}
                right={
                  <Switch
                    value={!batchMode}
                    onValueChange={v => {
                      if (v) playSfx('toggle');
                      setBatchMode(!v);
                    }}
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
                silent
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
                onPress={() => {
                  playSfx('tap');
                  confirmRename(!nameChanged);
                }}
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

      {/* Delete dialog */}
      <Modal
        visible={!!deleting}
        transparent
        statusBarTranslucent
        animationType="fade"
        onRequestClose={() => !deleteBusy && setDeleting(null)}
      >
        <Pressable
          style={[s.overlay, { justifyContent: 'center', paddingHorizontal: 20 }]}
          onPress={() => !deleteBusy && setDeleting(null)}
        >
          <Pressable style={s.dialog} onPress={noop}>
            <View style={{ padding: 20 }}>
              <Text style={[s.mono9, { color: C.red, letterSpacing: 2 }]}>DELETE</Text>
              <Text style={[s.menuTitle, { fontSize: 18, marginTop: 4 }]}>Delete transcription?</Text>
              <Text style={s.help}>What would you like to remove?</Text>
              <Text style={[s.mono9, { color: C.text, marginTop: 12 }]} numberOfLines={1}>
                {deleting?.title}
              </Text>
              <Text style={[s.help, { marginTop: 12 }]}>
                {deleting?.audioFile
                  ? 'Delete Transcript removes the text only; the audio stays on your device. Delete Transcript + Audio removes both, and cannot be undone.'
                  : 'This entry has no saved audio, so either choice removes the transcript. This cannot be undone.'}
              </Text>
            </View>
            <View style={s.btnCol}>
              <Pressable
                style={[s.btn, s.btnStack, s.btnDangerOutline, deleteBusy && { opacity: 0.5 }]}
                disabled={deleteBusy}
                onPress={() => deleting && deleteEntry(deleting, false)}
                accessibilityLabel="Delete transcript only"
              >
                <Text style={[s.btnText, { color: C.red }]}>DELETE TRANSCRIPT</Text>
              </Pressable>
              <Pressable
                style={[s.btn, s.btnStack, { backgroundColor: C.red }, deleteBusy && { opacity: 0.5 }]}
                disabled={deleteBusy}
                onPress={() => deleting && deleteEntry(deleting, true)}
                accessibilityLabel="Delete transcript and audio"
              >
                <Text style={[s.btnText, { color: C.canvas }]}>DELETE TRANSCRIPT + AUDIO</Text>
              </Pressable>
              <Pressable
                style={[s.btn, s.btnStack, { backgroundColor: C.faint }]}
                disabled={deleteBusy}
                onPress={() => {
                  playSfx('tap');
                  setDeleting(null);
                }}
                accessibilityLabel="Cancel"
              >
                <Text style={[s.btnText, { color: C.muted }]}>CANCEL</Text>
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
                      ['@react-native-documents/picker', 'MIT'],
                      ['react-native-sound', 'MIT'],
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
                      silent
                      onPress={() => {
                        if (!deleteAudio) playSfx('toggle');
                        setDeleteAudio(v => !v);
                      }}
                      right={
                        <Switch
                          value={deleteAudio}
                          onValueChange={v => {
                            if (v) playSfx('toggle');
                            setDeleteAudio(v);
                          }}
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
                  <Text style={[s.mono9, { letterSpacing: 1.5, marginTop: 24, marginBottom: 8, marginLeft: 4 }]}>SOUNDS</Text>
                  <View style={s.group}>
                    <MenuRow
                      icon="♪"
                      title="Sound effects"
                      sub="TAPS, ALERTS AND RECORDING CUES"
                      silent
                      onPress={() => toggleSfx(!sfxOn)}
                      right={
                        <Switch
                          value={sfxOn}
                          onValueChange={toggleSfx}
                          trackColor={{ false: C.faint, true: C.amber }}
                          thumbColor={C.canvas}
                        />
                      }
                    />
                  </View>
                  <Text style={[s.mono9, { letterSpacing: 1.5, marginTop: 24, marginBottom: 8, marginLeft: 4 }]}>APP LOCK</Text>
                  <View style={s.group}>
                    <MenuRow
                      icon="◈"
                      title="Lock the app"
                      sub="FINGERPRINT, FACE OR PHONE PIN"
                      silent
                      onPress={() => toggleAppLock(!appLock)}
                      right={
                        <Switch
                          value={appLock}
                          onValueChange={toggleAppLock}
                          trackColor={{ false: C.faint, true: C.amber }}
                          thumbColor={C.canvas}
                        />
                      }
                    />
                  </View>
                  <Text style={[s.help, { marginLeft: 4, marginTop: 8 }]}>
                    Viva Voce locks whenever you leave it and asks you to unlock when you come back.
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
                <View style={{ flexDirection: 'row', gap: 8 }}>
                  <Pressable onPress={() => { playSfx('tap'); exportEntry(selected); }} style={s.exportBtn}>
                    <Text style={[s.mono9, { color: C.amber }]}>↑ EXPORT</Text>
                  </Pressable>
                  <Pressable
                    onPress={() => {
                      playSfx('tap');
                      setDeleting(selected);
                    }}
                    style={[s.exportBtn, { borderColor: `${C.red}66` }]}
                    accessibilityLabel="Delete transcription"
                  >
                    <Text style={[s.mono9, { color: C.red }]}>⌫ DELETE</Text>
                  </Pressable>
                </View>
              </View>
              {!!selected.audioFile && (
                <View style={s.sheetPlayer}>
                  <PlayButton
                    playing={player.id === selected.id && player.status === 'playing'}
                    loading={player.id === selected.id && player.status === 'loading'}
                    disabled={false}
                    onPress={() => togglePlay(selected)}
                    label={player.id === selected.id && player.status === 'playing' ? 'Pause audio' : 'Play audio'}
                  />
                  {player.id === selected.id && (player.status === 'playing' || player.status === 'paused') ? (
                    <SeekBar pos={player.pos} dur={player.dur} onSeek={seekTo} />
                  ) : (
                    <Text style={[s.mono9, { flex: 1 }]}>PLAY THE ORIGINAL AUDIO</Text>
                  )}
                </View>
              )}
              {!!selected.segs && (
                <View style={s.tsRow}>
                  <Text style={[s.mono9, { letterSpacing: 1.5 }]}>TIMESTAMPS</Text>
                  <Switch
                    value={showTimestamps}
                    onValueChange={v => {
                      if (v) playSfx('toggle');
                      setShowTimestamps(v);
                    }}
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

      {/* Lock screen: a full-screen Modal is drawn above every other dialog and sheet */}
      <Modal visible={locked} animationType="none" statusBarTranslucent onRequestClose={noop}>
        <View style={{ flex: 1, backgroundColor: C.canvas, alignItems: 'center', justifyContent: 'center', padding: 32 }}>
          <View style={[s.menuIcon, { width: 64, height: 64, borderRadius: 20 }]}>
            <Text style={{ color: C.amber, fontSize: 28 }}>◈</Text>
          </View>
          <Text style={[s.mono9, { color: C.amber, letterSpacing: 2, marginTop: 24 }]}>LOCKED</Text>
          <Text style={[s.menuTitle, { fontSize: 18, marginTop: 4 }]}>Viva Voce</Text>
          {loaded && (
            <>
              <Text style={[s.help, { textAlign: 'center', marginTop: 8 }]}>
                {lockMsg || 'Unlock to see your transcripts.'}
              </Text>
              <Pressable
                style={[s.btn, s.btnStack, { backgroundColor: C.amber, marginTop: 24, alignSelf: 'stretch' }]}
                onPress={tryUnlock}
                accessibilityLabel="Unlock"
              >
                <Text style={[s.btnText, { color: C.canvas }]}>UNLOCK</Text>
              </Pressable>
            </>
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
  cardPlayer: { marginTop: 12, paddingTop: 8, borderTopWidth: 1, borderTopColor: C.borderSubtle },
  playBtn: { width: 32, height: 32, borderRadius: 16, borderWidth: 1, alignItems: 'center', justifyContent: 'center' },
  seekHit: { height: 28, justifyContent: 'center' },
  seekTrack: { height: 4, borderRadius: 2, backgroundColor: C.faint, overflow: 'hidden' },
  seekThumb: { position: 'absolute', top: 8, width: 12, height: 12, borderRadius: 6, backgroundColor: C.amber, borderWidth: 2, borderColor: C.canvas },
  seekTimes: { flexDirection: 'row', justifyContent: 'space-between' },
  sheetPlayer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    paddingHorizontal: 20,
    paddingVertical: 10,
    backgroundColor: C.raised,
    borderBottomWidth: 1,
    borderBottomColor: C.border,
  },

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
  btnCol: { gap: 8, padding: 12, borderTopWidth: 1, borderTopColor: C.border },
  btnStack: { flex: 0, alignSelf: 'stretch' }, // s.btn has flex: 1, which collapses to zero height in a column
  btnDangerOutline: { borderWidth: 1, borderColor: C.red },
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