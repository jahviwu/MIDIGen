'use client'

import { useEffect, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'

// Types
interface Note {
  id: string
  pitch: number
  startTime: number
  duration: number
  velocity: number
  selected: boolean
}

interface MidiData {
  notes: Note[]
  duration: number
  emotion: string | null
  genre: string | null
  midiBase64: string
}

// Constants
const NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
const BLACK_KEYS = new Set([1,3,6,8,10])

const PITCH_MIN  = 12
const PITCH_MAX  = 120
const PITCH_RANGE = PITCH_MAX - PITCH_MIN

const ROW_H      = 16
const PX_PER_SEC = 140
const KEY_W      = 44

// Quantization grids
const QUANTIZE_GRIDS: Record<string, number> = {
  "1/6 step": (1/6) * (1/4),
  "1/4 step": (1/4) * (1/4),
  "1/3 step": (1/3) * (1/4),
  "1/2 step": (1/2) * (1/4),
  "step":     1/4,

  "1/6 beat": 1/6,
  "1/4 beat": 1/4,
  "1/3 beat": 1/3,
  "1/2 beat": 1/2,
  "beat":     1.0,

  "bar":      4.0
}

function pitchName(p: number) {
  return NOTE_NAMES[p % 12] + (Math.floor(p / 12) - 1)
}

export default function Editor() {
  const router = useRouter()
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const [notes, setNotes]       = useState<Note[]>([])
  const [duration, setDuration] = useState(0)
  const [emotion, setEmotion]   = useState<string | null>(null)
  const [genre, setGenre]       = useState<string | null>(null)
  const [midiBase64, setMidiB64] = useState('')

  const [quantizeGrid, setQuantizeGrid] = useState("1/4 beat")
  const [mode, setMode] = useState<"draw" | "select">("draw")

  const resizingRef = useRef<null | {
    id: string
    startX: number
    startDuration: number
  }>(null)

  const [hoverResizeId, setHoverResizeId] = useState<string | null>(null)

  // Undo stack
  const undoStack = useRef<Note[][]>([])
  function pushUndo() {
    undoStack.current.push(JSON.parse(JSON.stringify(notes)))
  }

  function handleUndo() {
    const prev = undoStack.current.pop()
    if (prev) setNotes(prev)
  }

  // Ctrl+Z Undo + Ctrl+X delete selected
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.ctrlKey && e.key === 'z') {
        e.preventDefault()
        handleUndo()
      }
      if (e.ctrlKey && e.key === 'x') {
        e.preventDefault()
        pushUndo()
        setNotes(prev => prev.filter(n => !n.selected))
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [notes])

  // Load session MIDI
  useEffect(() => {
    const raw = sessionStorage.getItem('midiData')
    if (!raw) {
      router.push('/')
      return
    }

    const data: MidiData = JSON.parse(raw)

    setNotes(
      data.notes.map((n, i) => ({
        ...n,
        id: String(i),
        selected: false
      }))
    )

    setDuration(data.duration)
    setEmotion(data.emotion)
    setGenre(data.genre)
    setMidiB64(data.midiBase64)
  }, [router])

  const totalW = Math.max(duration * PX_PER_SEC + 240, 900)
  const totalH = PITCH_RANGE * ROW_H

  // Draw
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width  = totalW
    canvas.height = totalH

    ctx.fillStyle = '#ffffff'
    ctx.fillRect(0, 0, totalW, totalH)

    // Rows
    for (let p = PITCH_MIN; p < PITCH_MAX; p++) {
      const y = (PITCH_MAX - p - 1) * ROW_H
      const isBlack = BLACK_KEYS.has(p % 12)

      ctx.fillStyle = isBlack ? '#f1f5f9' : '#ffffff'
      ctx.fillRect(0, y, totalW, ROW_H)

      ctx.fillStyle = '#e2e8f0'
      ctx.fillRect(0, y + ROW_H - 1, totalW, 1)
    }

    // Time grid
    for (let t = 0; t <= duration + 2; t += 0.5) {
      const x = t * PX_PER_SEC
      ctx.fillStyle = t % 2 === 0 ? '#e2e8f0' : '#f1f5f9'
      ctx.fillRect(x, 0, 1, totalH)
    }

    // Notes
    for (const note of notes) {
      if (note.pitch < PITCH_MIN || note.pitch >= PITCH_MAX) continue

      const x = note.startTime * PX_PER_SEC
      const y = (PITCH_MAX - note.pitch - 1) * ROW_H + 1
      const w = Math.max(note.duration * PX_PER_SEC - 2, 5)
      const h = ROW_H - 2

      let color = '#2563eb'
      if (note.selected) color = '#ef4444'
      if (hoverResizeId === note.id) color = '#22c55e'

      ctx.fillStyle = color
      ctx.beginPath()
      ctx.roundRect(x + 1, y, w, h, 3)
      ctx.fill()
    }
  }, [notes, duration, totalW, totalH, hoverResizeId])

  useEffect(() => {
  function stopResize() {
    resizingRef.current = null
    setHoverResizeId(null)
  }

  window.addEventListener('mouseup', stopResize)
  return () => window.removeEventListener('mouseup', stopResize)
  }, [])

  // Note hit detection
  function getNoteAt(canvas: HTMLCanvasElement, e: React.MouseEvent) {
    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    const t = x / PX_PER_SEC
    const pitch = PITCH_MAX - Math.floor(y / ROW_H) - 1

    return notes.find(n =>
      n.pitch === pitch &&
      t >= n.startTime &&
      t <= n.startTime + n.duration
    )
  }

  // Mouse handlers
  function handleMouseDown(e: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = e.currentTarget
    const hit = getNoteAt(canvas, e)

    // Select mode
    if (mode === "select") {
      if (hit) {
        pushUndo()
        setNotes(prev =>
          prev.map(n =>
            n.id === hit.id ? { ...n, selected: !n.selected } : n
          )
        )
      }
      return
    }

    // DRAW MODE
    if (hit) {
      const rect = canvas.getBoundingClientRect()
      const x = e.clientX - rect.left
      const noteRight = (hit.startTime + hit.duration) * PX_PER_SEC

      // Resize
      if (Math.abs(x - noteRight) < 6) {
        resizingRef.current = {
          id: hit.id,
          startX: x,
          startDuration: hit.duration
        }
        return
      }

      // Delete
      pushUndo()
      setNotes(prev => prev.filter(n => n.id !== hit.id))
      return
    }

    // Add new note
    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    const t = x / PX_PER_SEC
    const pitch = PITCH_MAX - Math.floor(y / ROW_H) - 1

    const snap = 0.25
    const snappedTime = Math.round(t / snap) * snap

    const newNote: Note = {
      id: crypto.randomUUID(),
      pitch,
      startTime: snappedTime,
      duration: 0.5,
      velocity: 100,
      selected: false,
    }

    pushUndo()
    setNotes(prev => [...prev, newNote])
  }

function handleMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
  const canvas = e.currentTarget
  const rect = canvas.getBoundingClientRect()
  const x = e.clientX - rect.left
  const y = e.clientY - rect.top

  const t = x / PX_PER_SEC
  const pitch = PITCH_MAX - Math.floor(y / ROW_H) - 1

  const hit = notes.find(n =>
    n.pitch === pitch &&
    t >= n.startTime &&
    t <= n.startTime + n.duration
  )

  setHoverResizeId(null)

  if (hit && mode === "draw") {
    const noteRight = (hit.startTime + hit.duration) * PX_PER_SEC
    if (Math.abs(x - noteRight) < 6) {
      setHoverResizeId(hit.id)
    }
  }

  if (resizingRef.current) {
    const { id, startX, startDuration } = resizingRef.current
    const delta = (x - startX) / PX_PER_SEC
    const newDuration = Math.max(0.1, startDuration + delta)

    setNotes(prev =>
      prev.map(n =>
        n.id === id ? { ...n, duration: newDuration } : n
      )
    )
  }
}

function handleMouseUp() {
  // stop resizing when mouse is released
  resizingRef.current = null
  setHoverResizeId(null)
}
  // Quantize
  function handleQuantize() {
    const snap = QUANTIZE_GRIDS[quantizeGrid]

    pushUndo()
    setNotes(prev =>
      prev.map(n => ({
        ...n,
        startTime: Math.round(n.startTime / snap) * snap,
        duration: Math.max(snap, Math.round(n.duration / snap) * snap)
      }))
    )
  }

  // Download
  function handleDownload() {
    const bytes = atob(midiBase64)
    const arr = new Uint8Array(bytes.length)

    for (let i = 0; i < bytes.length; i++) {
      arr[i] = bytes.charCodeAt(i)
    }

    const blob = new Blob([arr], { type: 'audio/midi' })
    const url = URL.createObjectURL(blob)

    const a = document.createElement('a')
    a.href = url
    a.download = 'midigen.mid'
    a.click()

    URL.revokeObjectURL(url)
  }

  // Deselect all
  function handleDeselectAll() {
    pushUndo()
    setNotes(prev => prev.map(n => ({ ...n, selected: false })))
  }

  // Preview navigation
  function handlePreview() {
    // Save MIDI for preview page
    sessionStorage.setItem('midiDataBase64', midiBase64)

    // Navigate to preview page
    router.push('/preview')
  }

  // UI
  return (
    <main style={{ display: 'flex', height: '100vh', flexDirection: 'column' }}>

      {/* Toolbar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 16,
          padding: '12px 20px',
          background: '#f8fafc',
          borderBottom: '1px solid #e2e8f0',
          boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
          position: 'sticky',
          top: 0,
          zIndex: 10
        }}
      >
        {/* Mode Dropdown */}
        <select
          value={mode}
          onChange={e => setMode(e.target.value as any)}
          style={{
            fontSize: 16,
            padding: '8px 12px',
            borderRadius: 6,
            border: '1px solid #cbd5e1',
            background: '#fff',
            cursor: 'pointer'
          }}
        >
          <option value="draw">Draw Mode</option>
          <option value="select">Select Mode</option>
        </select>

        {/* Deselect All */}
        <button
          onClick={handleDeselectAll}
          style={{
            fontSize: 16,
            padding: '8px 16px',
            borderRadius: 6,
            background: '#475569',
            color: 'white',
            border: 'none',
            cursor: 'pointer'
          }}
        >
          Deselect All
        </button>

        {/* Quantize */}
        <select
          value={quantizeGrid}
          onChange={e => setQuantizeGrid(e.target.value)}
          style={{
            fontSize: 16,
            padding: '8px 12px',
            borderRadius: 6,
            border: '1px solid #cbd5e1',
            background: '#fff',
            cursor: 'pointer'
          }}
        >
          {Object.keys(QUANTIZE_GRIDS).map(key => (
            <option key={key} value={key}>{key}</option>
          ))}
        </select>

        <button
          onClick={handleQuantize}
          style={{
            fontSize: 16,
            padding: '8px 16px',
            borderRadius: 6,
            background: '#2563eb',
            color: 'white',
            border: 'none',
            cursor: 'pointer'
          }}
        >
          Quantize
        </button>

        {/* Download */}
        <button
          onClick={handleDownload}
          style={{
            fontSize: 16,
            padding: '8px 16px',
            borderRadius: 6,
            background: '#0f766e',
            color: 'white',
            border: 'none',
            cursor: 'pointer'
          }}
        >
          Download MIDI
        </button>

        {/* Preview */}
        <button
          onClick={handlePreview}
          style={{
            fontSize: 16,
            padding: '8px 16px',
            borderRadius: 6,
            background: '#16a34a',
            color: 'white',
            border: 'none',
            cursor: 'pointer'
          }}
        >
          Preview
        </button>
      </div>

      <div style={{ display: 'flex', flex: 1 }}>

        {/* Keys */}
        <div
          style={{
            width: KEY_W,
            overflowY: 'auto',
            borderRight: '1px solid #ddd'
          }}
        >
          {Array.from({ length: PITCH_RANGE }, (_, i) => {
            const pitch = PITCH_MAX - i - 1
            const isBlack = BLACK_KEYS.has(pitch % 12)

            return (
              <div
                key={pitch}
                style={{
                  height: ROW_H,
                  background: isBlack ? '#f1f5f9' : '#fff',
                  borderBottom: '1px solid #eee',
                  fontSize: 10,
                  textAlign: 'right',
                  paddingRight: 4
                }}
              >
                {pitchName(pitch)}
              </div>
            )
          })}
        </div>

        {/* Canvas */}
        <div style={{ flex: 1, overflow: 'auto' }}>
          <canvas
            ref={canvasRef}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
          />
        </div>

      </div>
    </main>
  )
}
