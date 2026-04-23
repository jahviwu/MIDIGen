'use client'

import { useEffect, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import { Midi } from '@tonejs/midi'
import Soundfont from 'soundfont-player'

const GM_INSTRUMENTS: Record<number, string> = {
  0: 'acoustic_grand_piano', 1: 'bright_acoustic_piano', 2: 'electric_grand_piano',
  3: 'honkytonk_piano', 4: 'electric_piano_1', 5: 'electric_piano_2',
  6: 'harpsichord', 7: 'clavinet', 8: 'celesta', 9: 'glockenspiel',
  24: 'acoustic_guitar_nylon', 25: 'acoustic_guitar_steel',
  32: 'acoustic_bass', 33: 'electric_bass_finger',
  40: 'violin', 41: 'viola', 42: 'cello',
  48: 'string_ensemble_1', 49: 'string_ensemble_2',
  56: 'trumpet', 57: 'trombone', 60: 'french_horn',
  65: 'alto_sax', 66: 'tenor_sax', 73: 'flute', 74: 'recorder',
}

function getInstrumentName(program: number): string {
  return GM_INSTRUMENTS[program] ?? 'acoustic_grand_piano'
}

type Status = 'idle' | 'loading-midi' | 'loading-fonts' | 'ready' | 'playing' | 'error'

export default function PreviewPage() {
  const router = useRouter()

  const [status, setStatus]       = useState<Status>('idle')
  const [statusMsg, setStatusMsg] = useState('Initializing…')
  const [error, setError]         = useState<string | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [totalDuration, setTotalDuration] = useState(0)
  const [progress, setProgress]   = useState(0)

  const waveformRef  = useRef<HTMLCanvasElement>(null)
  const progressRef  = useRef<HTMLCanvasElement>(null)
  const animFrameRef = useRef<number | null>(null)
  const startTimeRef = useRef<number>(0)
  const pauseTimeRef = useRef<number>(0)
  const acRef        = useRef<AudioContext | null>(null)
  const midiRef      = useRef<Midi | null>(null)
  const instrumentMapRef = useRef<Map<number, Soundfont.Player>>(new Map())
  const scheduledRef = useRef<boolean>(false)
  const analyserRef  = useRef<AnalyserNode | null>(null)
  const waveDataRef  = useRef<Float32Array | null>(null)

  // Load on mount
  useEffect(() => {
    load()
    return () => {
      animFrameRef.current && cancelAnimationFrame(animFrameRef.current)
      acRef.current?.close()
    }
  }, [])

  async function load() {
    try {
      setStatus('loading-midi')
      setStatusMsg('Reading MIDI data…')

      const raw = sessionStorage.getItem('midiDataBase64')
      if (!raw) throw new Error('No MIDI found. Please generate a MIDI first.')

      const bytes = Uint8Array.from(atob(raw), c => c.charCodeAt(0))
      const midi  = new Midi(bytes)
      midiRef.current = midi

      const dur = Math.max(
        ...midi.tracks.flatMap(t => t.notes.map(n => n.time + n.duration)),
        1
      )
      setTotalDuration(dur)

      setStatus('loading-fonts')
      setStatusMsg('Loading soundfonts…')

      const ac = new AudioContext()
      acRef.current = ac

      // Analyser for waveform
      const analyser = ac.createAnalyser()
      analyser.fftSize = 2048
      analyser.connect(ac.destination)
      analyserRef.current = analyser
      waveDataRef.current = new Float32Array(analyser.fftSize)

      const programs = Array.from(new Set(
        midi.tracks
          .filter(t => !t.instrument.percussion)
          .map(t => t.instrument.number)
      ))

      await Promise.all(programs.map(async program => {
        const name = getInstrumentName(program)
        const inst = await Soundfont.instrument(ac, name as Soundfont.InstrumentName, {
          soundfont: 'MusyngKite',
          format: 'mp3',
          gain: 4,
          destination: analyser, // route through analyser → destination
        })
        instrumentMapRef.current.set(program, inst)
      }))

      setStatus('ready')
      setStatusMsg('Ready to play')

      // Draw static waveform bars from note data
      drawStaticWaveform(midi, dur)

    } catch (e: any) {
      setError(e.message)
      setStatus('error')
    }
  }

  function drawStaticWaveform(midi: Midi, dur: number) {
    const canvas = waveformRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const W = canvas.width
    const H = canvas.height
    ctx.clearRect(0, 0, W, H)

    // Build a density map from notes
    const buckets = new Float32Array(W)
    for (const track of midi.tracks) {
      for (const note of track.notes) {
        const startBucket = Math.floor((note.time / dur) * W)
        const endBucket   = Math.floor(((note.time + note.duration) / dur) * W)
        for (let i = startBucket; i <= Math.min(endBucket, W - 1); i++) {
          buckets[i] += note.velocity
        }
      }
    }

    const maxVal = Math.max(...buckets, 1)

    // Draw bars
    for (let i = 0; i < W; i++) {
      const normalized = buckets[i] / maxVal
      const barH = normalized * (H * 0.9)
      const y    = (H - barH) / 2

      const alpha = 0.3 + normalized * 0.5
      ctx.fillStyle = `rgba(37, 99, 235, ${alpha})`
      ctx.fillRect(i, y, 1, barH)
    }
  }

  function drawProgressOverlay(fraction: number) {
    const wave = waveformRef.current
    const prog = progressRef.current
    if (!wave || !prog) return

    const ctx = prog.getContext('2d')
    if (!ctx) return

    const W = prog.width
    const H = prog.height
    ctx.clearRect(0, 0, W, H)

    // Tint played portion
    ctx.fillStyle = 'rgba(37, 99, 235, 0.18)'
    ctx.fillRect(0, 0, fraction * W, H)

    // Playhead line
    ctx.fillStyle = '#2563eb'
    ctx.fillRect(fraction * W - 1, 0, 2, H)
  }

  function scheduleAllNotes(startOffset: number = 0) {
    const ac   = acRef.current
    const midi = midiRef.current
    if (!ac || !midi) return

    const now = ac.currentTime

    for (const track of midi.tracks) {
      if (track.instrument.percussion) continue
      const inst = instrumentMapRef.current.get(track.instrument.number)
      if (!inst) continue

      const events = track.notes
        .filter(n => n.time + n.duration > startOffset)
        .map(n => ({
          time:     Math.max(0, n.time - startOffset),
          note:     n.midi,
          duration: n.duration,
          gain:     n.velocity,
        }))

      if (events.length > 0) {
        inst.schedule(now, events)
      }
    }
  }

  function stopAllNotes() {
    for (const inst of instrumentMapRef.current.values()) {
      inst.stop()
    }
  }

  function startPlaybackLoop(startOffset: number) {
    const ac = acRef.current
    if (!ac) return

    startTimeRef.current = ac.currentTime - startOffset

    function tick() {
      const ac = acRef.current
      if (!ac) return

      const elapsed  = ac.currentTime - startTimeRef.current
      const fraction = Math.min(elapsed / totalDuration, 1)

      setCurrentTime(elapsed)
      setProgress(fraction)
      drawProgressOverlay(fraction)

      if (fraction >= 1) {
        setIsPlaying(false)
        setStatus('ready')
        setStatusMsg('Ready to play')
        setCurrentTime(0)
        setProgress(0)
        drawProgressOverlay(0)
        pauseTimeRef.current = 0
        return
      }

      animFrameRef.current = requestAnimationFrame(tick)
    }

    animFrameRef.current = requestAnimationFrame(tick)
  }

  function handlePlay() {
    if (status !== 'ready' && status !== 'playing') return
    const ac = acRef.current
    if (!ac) return

    if (isPlaying) {
      // Pause
      pauseTimeRef.current = ac.currentTime - startTimeRef.current
      stopAllNotes()
      animFrameRef.current && cancelAnimationFrame(animFrameRef.current)
      setIsPlaying(false)
      setStatus('ready')
      setStatusMsg('Paused')
    } else {
      // Play / resume
      if (ac.state === 'suspended') ac.resume()
      const offset = pauseTimeRef.current
      scheduleAllNotes(offset)
      startPlaybackLoop(offset)
      setIsPlaying(true)
      setStatus('playing')
      setStatusMsg('Playing…')
    }
  }

  function handleStop() {
    stopAllNotes()
    animFrameRef.current && cancelAnimationFrame(animFrameRef.current)
    pauseTimeRef.current = 0
    setIsPlaying(false)
    setStatus('ready')
    setStatusMsg('Ready to play')
    setCurrentTime(0)
    setProgress(0)
    drawProgressOverlay(0)
  }

  function handleSeek(e: React.MouseEvent<HTMLDivElement>) {
    const rect     = e.currentTarget.getBoundingClientRect()
    const fraction = (e.clientX - rect.left) / rect.width
    const seekTime = fraction * totalDuration

    if (isPlaying) {
      stopAllNotes()
      animFrameRef.current && cancelAnimationFrame(animFrameRef.current)
    }

    pauseTimeRef.current = seekTime
    setCurrentTime(seekTime)
    setProgress(fraction)
    drawProgressOverlay(fraction)

    if (isPlaying) {
      const ac = acRef.current
      if (!ac) return
      if (ac.state === 'suspended') ac.resume()
      scheduleAllNotes(seekTime)
      startPlaybackLoop(seekTime)
    }
  }

  function handleDownloadMidi() {
    const raw = sessionStorage.getItem('midiDataBase64')
    if (!raw) return
    const bytes = Uint8Array.from(atob(raw), c => c.charCodeAt(0))
    const blob  = new Blob([bytes], { type: 'audio/midi' })
    const url   = URL.createObjectURL(blob)
    const a     = document.createElement('a')
    a.href = url; a.download = 'midigen.mid'; a.click()
    URL.revokeObjectURL(url)
  }

  function formatTime(s: number) {
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    return `${m}:${sec.toString().padStart(2, '0')}`
  }

  const isLoading = status === 'loading-midi' || status === 'loading-fonts'
  const canPlay   = status === 'ready' || status === 'playing'

  return (
    <main style={{
      minHeight: '100vh',
      background: 'var(--bg)',
      display: 'flex',
      flexDirection: 'column',
    }}>

      {/* Nav — same as home */}
      <nav style={{
        borderBottom: '1px solid var(--border)',
        padding: '0 32px',
        height: '56px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: 'var(--bg)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '28px', height: '28px', borderRadius: '8px',
            background: 'var(--navy)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <span style={{ color: 'white', fontSize: '14px', fontWeight: 700 }}>M</span>
          </div>
          <span style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: '16px', color: 'var(--navy)' }}>
            MIDIGen
          </span>
        </div>
        <button
          onClick={() => router.push('/editor')}
          style={{
            fontSize: '13px', fontWeight: 600,
            color: 'var(--navy)',
            background: 'transparent',
            border: '1px solid var(--border)',
            borderRadius: '6px',
            padding: '6px 14px',
            cursor: 'pointer',
          }}
        >
          ← Back to Editor
        </button>
      </nav>

      {/* Hero */}
      <div style={{
        background: 'linear-gradient(135deg, var(--navy) 0%, var(--navy-light) 100%)',
        padding: '40px 32px',
        textAlign: 'center',
      }}>
        <h1 style={{
          fontFamily: 'var(--font-display)',
          fontSize: '36px',
          fontWeight: 800,
          color: 'white',
          lineHeight: 1.1,
          marginBottom: '8px',
        }}>
          Preview
        </h1>
        <p style={{ color: 'rgba(255,255,255,0.65)', fontSize: '14px' }}>
          Listen to your generated MIDI
        </p>
      </div>

      {/* Main card */}
      <div style={{
        flex: 1,
        background: 'var(--bg-2)',
        display: 'flex',
        justifyContent: 'center',
        padding: '40px 20px',
      }}>
        <div style={{
          width: '100%',
          maxWidth: '640px',
          background: 'var(--bg)',
          borderRadius: '16px',
          border: '1px solid var(--border)',
          padding: '32px',
          boxShadow: '0 4px 24px rgba(0,0,0,0.06)',
        }}>

          {/* Status */}
          <div style={{
            marginBottom: '24px',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
          }}>
            {isLoading && (
              <div style={{
                width: '10px', height: '10px', borderRadius: '50%',
                background: 'var(--blue)',
                animation: 'pulse 1s infinite',
              }} />
            )}
            {status === 'playing' && (
              <div style={{
                width: '10px', height: '10px', borderRadius: '50%',
                background: '#16a34a',
                animation: 'pulse 0.6s infinite',
              }} />
            )}
            {status === 'ready' && (
              <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#16a34a' }} />
            )}
            {status === 'error' && (
              <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: 'var(--red)' }} />
            )}
            <span style={{ fontSize: '13px', color: 'var(--text-2)', fontWeight: 500 }}>
              {statusMsg}
            </span>
          </div>

          {/* Error */}
          {error && (
            <div style={{
              padding: '10px 14px',
              background: 'var(--red-pale)',
              border: '1px solid #fca5a5',
              borderRadius: '8px',
              color: 'var(--red)',
              fontSize: '13px',
              marginBottom: '16px',
            }}>
              {error}
            </div>
          )}

          {/* Waveform */}
          <div
            onClick={canPlay ? handleSeek : undefined}
            style={{
              position: 'relative',
              borderRadius: '10px',
              overflow: 'hidden',
              background: '#f1f5f9',
              border: '1px solid var(--border)',
              marginBottom: '16px',
              cursor: canPlay ? 'pointer' : 'default',
              height: '100px',
            }}
          >
            <canvas
              ref={waveformRef}
              width={600}
              height={100}
              style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}
            />
            <canvas
              ref={progressRef}
              width={600}
              height={100}
              style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}
            />
            {isLoading && (
              <div style={{
                position: 'absolute', inset: 0,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '13px', color: 'var(--text-3)',
              }}>
                Loading soundfonts…
              </div>
            )}
          </div>

          {/* Time display */}
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: '11px',
            color: 'var(--text-3)',
            marginBottom: '20px',
            fontVariantNumeric: 'tabular-nums',
          }}>
            <span>{formatTime(currentTime)}</span>
            <span>{formatTime(totalDuration)}</span>
          </div>

          {/* Transport controls */}
          <div style={{ display: 'flex', gap: '10px', marginBottom: '24px' }}>
            {/* Play / Pause */}
            <button
              onClick={handlePlay}
              disabled={!canPlay}
              style={{
                flex: 1,
                padding: '12px',
                fontSize: '20px',
                borderRadius: '10px',
                border: 'none',
                background: canPlay
                  ? 'linear-gradient(135deg, var(--navy) 0%, var(--navy-light) 100%)'
                  : 'var(--bg-3)',
                color: canPlay ? 'white' : 'var(--text-3)',
                cursor: canPlay ? 'pointer' : 'not-allowed',
                boxShadow: canPlay ? '0 4px 14px rgba(26,45,90,0.3)' : 'none',
                transition: 'opacity 0.15s',
              }}
            >
              {isPlaying ? '⏸' : '▶'}
            </button>

            {/* Stop */}
            <button
              onClick={handleStop}
              disabled={!canPlay}
              style={{
                padding: '12px 20px',
                fontSize: '18px',
                borderRadius: '10px',
                border: '1px solid var(--border)',
                background: 'var(--bg)',
                color: canPlay ? 'var(--text-2)' : 'var(--text-3)',
                cursor: canPlay ? 'pointer' : 'not-allowed',
              }}
            >
              ⏹
            </button>
          </div>

          {/* Download MIDI */}
          <button
            onClick={handleDownloadMidi}
            style={{
              width: '100%',
              padding: '12px',
              fontSize: '14px',
              fontWeight: 600,
              borderRadius: '10px',
              border: '1px solid var(--border)',
              background: 'var(--bg)',
              color: 'var(--navy)',
              cursor: 'pointer',
            }}
          >
            Download MIDI
          </button>

        </div>
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
      `}</style>
    </main>
  )
}
