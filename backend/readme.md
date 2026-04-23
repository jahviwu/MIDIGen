# Symbolic Music Generation

An end-to-end pipeline that learns to compose original music by training 
an autoregressive Transformer on a large corpus of MIDI files.

## What it does
- Ingests and tokenizes 108,023 MIDI files from the XMIDI Dataset
- Trains a Transformer decoder to model musical sequences at the event level
- Generates new compositions by autoregressively sampling token sequences
- Reconstructs generated tokens back into playable .mid files

## Pipeline
Raw MIDI → Tokenizer → Preprocessor → Transformer → Generator → MIDI Output

## Dataset
Trained on the [XMIDI Dataset](https://github.com/xmusic-project/XMIDI_Dataset) — 
the largest known symbolic music dataset with emotion and genre labels, 
comprising 108,023 MIDI files totaling ~5,278 hours of music.

## Stack
- Python, PyTorch
- pretty_midi
- Custom event-based tokenizer (NOTE_ON, NOTE_OFF, VELOCITY, TIME_SHIFT)