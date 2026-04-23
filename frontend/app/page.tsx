'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'

const EMOTIONS = ['angry','exciting','fear','funny','happy','lazy','magnificent','quiet','romantic','sad','warm']
const GENRES   = ['classical','country','jazz','pop','rock','traditional']

export default function Home() {
  const router = useRouter()
  const [prompt, setPrompt]           = useState('')
  const [length, setLength]           = useState(1000)
  const [temperature, setTemperature] = useState(0.9)
  const [useGroq, setUseGroq]         = useState(false)
  const [loading, setLoading]         = useState(false)
  const [progress, setProgress]       = useState(0)
  const [progressLabel, setProgressLabel] = useState('')
  const [error, setError]             = useState('')

  async function handleGenerate() {
    if (!prompt.trim()) { setError('Please enter a prompt'); return }
    setError('')
    setLoading(true)
    setProgress(0)

    const stages = [
      { pct: 8,  label: 'Parsing prompt...' },
      { pct: 20, label: 'Loading model...' },
      { pct: 38, label: 'Finding seed file...' },
      { pct: 55, label: 'Generating tokens...' },
      { pct: 75, label: 'Still generating...' },
      { pct: 90, label: 'Reconstructing MIDI...' },
    ]

    let idx = 0
    setProgressLabel(stages[0].label)
    const interval = setInterval(() => {
      idx++
      if (idx < stages.length) {
        setProgress(stages[idx].pct)
        setProgressLabel(stages[idx].label)
      }
    }, 3000)

    try {
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, length, temperature, useGroq }),
      })
      clearInterval(interval)

      if (!res.ok) {
        const d = await res.json()
        throw new Error(d.error || 'Generation failed')
      }

      setProgress(100)
      setProgressLabel('Done!')
      const data = await res.json()
      sessionStorage.setItem('midiData', JSON.stringify(data))
      setTimeout(() => router.push('/editor'), 400)

    } catch (err: any) {
      clearInterval(interval)
      setError(err.message || 'Something went wrong')
      setLoading(false)
      setProgress(0)
    }
  }

  return (
    <main style={{
      minHeight: '100vh',
      background: 'var(--bg)',
      display: 'flex',
      flexDirection: 'column',
    }}>

      {/* Top nav */}
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
        <span style={{ fontSize: '12px', color: 'var(--text-3)' }}>
          AI Music Generation
        </span>
      </nav>

      {/* Hero section */}
      <div style={{
        background: 'linear-gradient(135deg, var(--navy) 0%, var(--navy-light) 100%)',
        padding: '64px 32px',
        textAlign: 'center',
      }}>
        <div className="fade-up">
          <h1 style={{
            fontFamily: 'var(--font-display)',
            fontSize: '52px',
            fontWeight: 800,
            color: 'white',
            lineHeight: 1.1,
            marginBottom: '16px',
          }}>
            Generate MIDI<br />
            <span style={{ color: '#93c5fd' }}>from a prompt</span>
          </h1>
          <p style={{ color: 'rgba(255,255,255,0.65)', fontSize: '16px', maxWidth: '480px', margin: '0 auto' }}>
            Create a MIDI file you can edit and download.
          </p>
        </div>
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

          {/* Prompt */}
          <div className="fade-up-1" style={{ marginBottom: '20px' }}>
            <label style={{
              display: 'block',
              fontSize: '12px',
              fontWeight: 600,
              color: 'var(--navy)',
              marginBottom: '8px',
              letterSpacing: '0.04em',
            }}>
              Describe your music
            </label>
            <textarea
              value={prompt}
              onChange={e => setPrompt(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && e.metaKey) handleGenerate() }}
              placeholder='e.g. "angry country song for my dog"'
              disabled={loading}
              rows={3}
              style={{
                width: '100%',
                padding: '12px 14px',
                fontSize: '15px',
                resize: 'none',
                lineHeight: 1.5,
              }}
            />
            <div style={{ marginTop: '6px', fontSize: '11px', color: 'var(--text-3)' }}>
              Try: emotion + genre + any description
            </div>
          </div>

          {/* Controls */}
          <div className="fade-up-2" style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '16px',
            marginBottom: '20px',
          }}>
            <div>
              <label style={{ display: 'block', fontSize: '11px', fontWeight: 600, color: 'var(--text-2)', marginBottom: '6px' }}>
                Length: {length} tokens
              </label>
              <input
                type="range" min={200} max={3000} step={100}
                value={length}
                onChange={e => setLength(Number(e.target.value))}
                disabled={loading}
                style={{ width: '100%', accentColor: 'var(--blue)', border: 'none', boxShadow: 'none', background: 'none', padding: 0 }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--text-3)' }}>
                <span>200</span><span>3000</span>
              </div>
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '11px', fontWeight: 600, color: 'var(--text-2)', marginBottom: '6px' }}>
                Temperature: {temperature.toFixed(2)}
              </label>
              <input
                type="range" min={0.5} max={1.5} step={0.05}
                value={temperature}
                onChange={e => setTemperature(Number(e.target.value))}
                disabled={loading}
                style={{ width: '100%', accentColor: 'var(--blue)', border: 'none', boxShadow: 'none', background: 'none', padding: 0 }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--text-3)' }}>
                <span>Structured</span><span>Creative</span>
              </div>
            </div>
          </div>

          {/* Parser toggle */}
          <div className="fade-up-2" style={{ marginBottom: '24px' }}>
            <label style={{ display: 'block', fontSize: '11px', fontWeight: 600, color: 'var(--text-2)', marginBottom: '8px' }}>
              PROMPT PARSER
            </label>
            <div style={{ display: 'flex', gap: '8px' }}>
              {[
                { val: false, label: 'Simple (Keywords)', desc: 'No API call' },
                { val: true,  label: 'Groq AI',          desc: 'Groq API call' },
              ].map(opt => (
                <button
                  key={String(opt.val)}
                  onClick={() => setUseGroq(opt.val)}
                  disabled={loading}
                  style={{
                    flex: 1,
                    padding: '10px 12px',
                    borderRadius: '8px',
                    border: `1.5px solid ${useGroq === opt.val ? 'var(--blue)' : 'var(--border)'}`,
                    background: useGroq === opt.val ? 'var(--blue-pale)' : 'var(--bg)',
                    color: useGroq === opt.val ? 'var(--blue)' : 'var(--text-2)',
                    textAlign: 'left',
                  }}
                >
                  <div style={{ fontSize: '12px', fontWeight: 600 }}>{opt.label}</div>
                  <div style={{ fontSize: '10px', opacity: 0.7, marginTop: '2px' }}>{opt.desc}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Error */}
          {error && (
            <div style={{
              marginBottom: '16px',
              padding: '10px 14px',
              background: 'var(--red-pale)',
              border: '1px solid #fca5a5',
              borderRadius: '8px',
              color: 'var(--red)',
              fontSize: '13px',
            }}>
              {error}
            </div>
          )}

          {/* Progress bar */}
          {loading && (
            <div style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                <span style={{ fontSize: '12px', color: 'var(--blue)', fontWeight: 500 }}>{progressLabel}</span>
                <span style={{ fontSize: '12px', color: 'var(--text-3)' }}>{progress}%</span>
              </div>
              <div style={{ height: '6px', background: 'var(--bg-3)', borderRadius: '3px', overflow: 'hidden' }}>
                <div style={{
                  height: '100%',
                  width: `${progress}%`,
                  background: 'linear-gradient(90deg, var(--navy) 0%, var(--blue-light) 100%)',
                  borderRadius: '3px',
                  transition: 'width 0.5s ease',
                }} />
              </div>
            </div>
          )}

          {/* Generate button */}
          <div className="fade-up-3">
            <button
              onClick={handleGenerate}
              disabled={loading}
              style={{
                width: '100%',
                padding: '14px',
                fontSize: '14px',
                fontWeight: 700,
                letterSpacing: '0.04em',
                background: loading
                  ? 'var(--bg-3)'
                  : 'linear-gradient(135deg, var(--navy) 0%, var(--navy-light) 100%)',
                color: loading ? 'var(--text-3)' : 'white',
                border: 'none',
                borderRadius: '10px',
                boxShadow: loading ? 'none' : '0 4px 14px rgba(26,45,90,0.3)',
              }}
            >
              {loading ? 'Generating...' : 'Generate MIDI →'}
            </button>
          </div>

          {/* Tag hints */}
          <div className="fade-up-4" style={{ marginTop: '20px' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-3)', marginBottom: '6px' }}>Available emotions & genres:</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
              {[...EMOTIONS, ...GENRES].map(tag => (
                <span
                  key={tag}
                  onClick={() => !loading && setPrompt(p => p ? p + ' ' + tag : tag)}
                  style={{
                    padding: '3px 8px',
                    borderRadius: '12px',
                    fontSize: '11px',
                    background: 'var(--bg-3)',
                    color: 'var(--text-2)',
                    border: '1px solid var(--border)',
                    cursor: 'pointer',
                  }}
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>

        </div>
      </div>
    </main>
  )
}