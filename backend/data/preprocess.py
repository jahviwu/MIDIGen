import json
import multiprocessing as mp
from pathlib import Path
import traceback
import torch
from tokenizer import midi_to_tokens

BASE = Path(__file__).resolve().parent
RAW_DIR = BASE / "midi_raw" / "XMIDI_Dataset"
OUT_DIR = BASE / "midi_tokenized"
VOCAB_PATH = BASE / "vocab.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_vocab():
    with open(VOCAB_PATH) as f:
        return json.load(f)

vocab = load_vocab()

def process_file(mid_path):
    """Tokenize a single MIDI file in a separate process."""
    try:
        out_path = OUT_DIR / (mid_path.stem + ".pt")

        # Skip if already tokenized
        if out_path.exists():
            return ("skip", mid_path.name)

        tokens = midi_to_tokens(mid_path)
        ids = torch.tensor([vocab.get(tok, vocab["<UNK>"]) for tok in tokens], dtype=torch.long)
        torch.save(ids, out_path)

        return ("ok", mid_path.name)

    except Exception as e:
        return ("error", mid_path.name, str(e), traceback.format_exc())


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw MIDI directory not found:\n  {RAW_DIR}")

    midi_files = sorted(list(RAW_DIR.glob("*.mid")) + list(RAW_DIR.glob("*.midi")))
    total = len(midi_files)

    print(f"Found {total:,} MIDI files")
    print(f"Using {mp.cpu_count()} CPU cores\n")

    with mp.Pool(mp.cpu_count()) as pool:
        for i, result in enumerate(pool.imap_unordered(process_file, midi_files), 1):
            status = result[0]

            if status == "ok":
                print(f"[{i}/{total}] ✔ {result[1]}")
            elif status == "skip":
                print(f"[{i}/{total}] ↷ {result[1]}")
            else:
                print(f"[{i}/{total}] ✖ ERROR in {result[1]}")
                print(result[2])
                print(result[3])

    print("\nDone. Tokenization complete.")


if __name__ == "__main__":
    main()
