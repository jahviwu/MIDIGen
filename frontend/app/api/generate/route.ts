import { NextRequest, NextResponse } from 'next/server'
import { exec } from 'child_process'
import { promisify } from 'util'
import * as fs from 'fs'
import * as path from 'path'
import * as os from 'os'

const execAsync = promisify(exec)

const PROJECT_ROOT = path.join(process.cwd(), '..', 'midi-generation-project')
const PYTHON       = path.join(PROJECT_ROOT, '.venv', 'Scripts', 'python.exe')
const GEN_SCRIPT   = path.join(PROJECT_ROOT, 'generation', 'generate_midi.py')
const OUTPUT_DIR   = path.join(PROJECT_ROOT, 'generation', 'outputs')

export async function POST(req: NextRequest) {
  try {
    const { prompt, length, temperature, useGroq } = await req.json()

    if (!prompt?.trim()) {
      return NextResponse.json({ error: 'Prompt is required' }, { status: 400 })
    }
    const safePrompt = prompt.replace(/\r?\n|\r/g, ' ').trim()

    const timestamp  = Date.now()
    const outputPath = path.join(OUTPUT_DIR, `generated_${timestamp}.mid`)

    const args = [
      `"${GEN_SCRIPT}"`,
      `--prompt "${safePrompt.replace(/"/g, '\\"')}"`,
      `--length ${length || 1000}`,
      `--temperature ${temperature || 0.9}`,
      `--output "${outputPath}"`,
      useGroq ? '--groq' : '',
    ].filter(Boolean).join(' ')

    const { stdout, stderr } = await execAsync(`"${PYTHON}" ${args}`, {
      timeout: 300000,
      cwd: PROJECT_ROOT,
    })

    if (!fs.existsSync(outputPath)) {
      throw new Error('Generation completed but output file not found')
    }

    const emotionMatch  = stdout.match(/emotion=(\w+)/)
    const genreMatch    = stdout.match(/genre=(\w+)/)
    const durationMatch = stdout.match(/Duration:\s*([\d.]+)/)

    const emotion  = emotionMatch?.[1]  || null
    const genre    = genreMatch?.[1]    || null
    const duration = parseFloat(durationMatch?.[1] || '10')

    const midiBuffer = fs.readFileSync(outputPath)
    const midiBase64 = midiBuffer.toString('base64')
    const notes      = await parseMidi(outputPath)

    cleanupOldFiles(OUTPUT_DIR)

    return NextResponse.json({ emotion, genre, duration, midiBase64, notes })

  } catch (err: any) {
    console.error('Generation error:', err)
    return NextResponse.json({ error: err.message || 'Generation failed' }, { status: 500 })
  }
}

async function parseMidi(midiPath: string) {
  const script = `
import json, pretty_midi, sys
pm = pretty_midi.PrettyMIDI(sys.argv[1])
notes = []
for inst in pm.instruments:
    for n in inst.notes:
        notes.append({'pitch': n.pitch, 'startTime': round(n.start,4), 'duration': round(n.end-n.start,4), 'velocity': n.velocity, 'selected': False})
notes.sort(key=lambda x: x['startTime'])
print(json.dumps(notes))
`
  const tmp = path.join(os.tmpdir(), `parse_${Date.now()}.py`)
  fs.writeFileSync(tmp, script)
  try {
    const { stdout } = await execAsync(`"${PYTHON}" "${tmp}" "${midiPath}"`)
    fs.unlinkSync(tmp)
    return JSON.parse(stdout.trim())
  } catch {
    try { fs.unlinkSync(tmp) } catch {}
    return []
  }
}

function cleanupOldFiles(dir: string) {
  try {
    fs.readdirSync(dir)
      .filter(f => f.startsWith('generated_') && f.endsWith('.mid'))
      .map(f => ({ name: f, t: fs.statSync(path.join(dir, f)).mtimeMs }))
      .sort((a, b) => b.t - a.t)
      .slice(5)
      .forEach(f => { try { fs.unlinkSync(path.join(dir, f.name)) } catch {} })
  } catch {}
}