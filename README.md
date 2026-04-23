# MidiGen: Symbolic Music Generation

An end-to-end pipeline that learns to compose original music by training an autoregressive Transformer on a large corpus of MIDI files. Describe a mood or style in plain text and get an original MIDI composition back.

---

## How It Works

1. You type a prompt like `"warm traditional song"` or `"dark cinematic tension"`
2. The prompt is parsed by **Groq LLM** to extract musical parameters (tempo, key, mood, instruments)
3. A trained **Transformer decoder** autoregressively samples a token sequence
4. Generated tokens are reconstructed back into a playable `.mid` file
5. The **Next.js** frontend lets you generate, preview, and download the result

---

## Pipeline

```
Raw MIDI → Tokenizer → Preprocessor → Transformer → Generator → MIDI Output
```

---

## Dataset

Trained on the **[XMIDI Dataset](https://github.com/ngyunkang/XMIDI)** — the largest known symbolic music dataset with emotion and genre labels.

| Stat | Value |
|---|---|
| Total MIDI files | 108,023 |
| Total duration | ~5,278 hours |
| Labels | Emotion + Genre |

> The raw and tokenized dataset files are not included in this repository due to size (~100k files). Download from the XMIDI Dataset source and run the preprocessing pipeline to reproduce.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, React |
| LLM / Prompt Parsing | Groq API |
| Model | Transformer Decoder (PyTorch) |
| Tokenization | Custom event-based tokenizer |
| MIDI I/O | `pretty_midi` |
| Token Events | `NOTE_ON`, `NOTE_OFF`, `VELOCITY`, `TIME_SHIFT` |
| Runtime | Node.js + Python (.venv) |

---

## Getting Started

### Prerequisites

- Node.js 18+
- Python 3.9+
- A [Groq API key](https://console.groq.com)

---

### 1. Clone the repo

```bash
git clone https://github.com/jahviwu/MIDIGen
cd MIDIGen
```

---

### 2. Set up the Python backend

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

### 3. Set up the Next.js frontend

```bash
cd frontend
npm install
```

---

### 4. Configure environment variables

Create a `.env.local` file inside the `midigen/` folder:

```bash
cp .env.example .env.local
```

Then open `.env.local` and fill in your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

---

### 5. Run the app

```bash
cd frontend
npm run dev
```

Visit [http://localhost:3000](http://localhost:3000)

---

## Project Structure

```
4620 Model/
├── frontend/                        # Next.js frontend + API routes
│   ├── src/
│   ├── .env.example                # Template for environment variables
│   └── package.json
│
└── backend/        # Python backend
    ├── generation/
    │   ├── generate_midi.py        # Main generation script
    │   └── prompt_parser.py        # Groq LLM prompt parser
    ├── tokenizer/                  # Custom event-based tokenizer
    ├── model/                      # Transformer decoder
    ├── requirements.txt
    └── data/                       # MIDI dataset (not included in repo)
        ├── raw/                    # 108,023 raw MIDI files
        └── tokenized/              # Preprocessed token sequences
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key from [console.groq.com](https://console.groq.com) |

---

## License

MIT License: Feel free to use, modify, and build on this project.