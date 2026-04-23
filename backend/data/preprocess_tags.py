import json
import multiprocessing as mp
from pathlib import Path

import torch
from tqdm import tqdm

BASE = Path(__file__).resolve().parent
TOKENIZED_DIR = BASE / "midi_tokenized"
VOCAB_PATH    = BASE / "vocab.json"

with open(VOCAB_PATH) as f:
    vocab = json.load(f)

unk_id = vocab["<UNK>"]

EMOTION_MAP = {
    "angry":       "<EMOTION_ANGRY>",
    "exciting":    "<EMOTION_EXCITING>",
    "fear":        "<EMOTION_FEAR>",
    "funny":       "<EMOTION_FUNNY>",
    "happy":       "<EMOTION_HAPPY>",
    "lazy":        "<EMOTION_LAZY>",
    "magnificent": "<EMOTION_MAGNIFICENT>",
    "quiet":       "<EMOTION_QUIET>",
    "romantic":    "<EMOTION_ROMANTIC>",
    "sad":         "<EMOTION_SAD>",
    "warm":        "<EMOTION_WARM>",
}

GENRE_MAP = {
    "classical":   "<GENRE_CLASSICAL>",
    "country":     "<GENRE_COUNTRY>",
    "jazz":        "<GENRE_JAZZ>",
    "pop":         "<GENRE_POP>",
    "rock":        "<GENRE_ROCK>",
    "traditional": "<GENRE_TRADITIONAL>",
}

# Parse Function
def parse_emotion_genre(filename: str):
    parts = filename.replace(".pt", "").split("_")
    if len(parts) < 4:
        return None, None
    return parts[1].lower(), parts[2].lower()

def process_file(pt_path):
    try:
        emotion_str, genre_str = parse_emotion_genre(pt_path.name)

        if emotion_str is None or genre_str is None:
            return ("skip", pt_path.name)

        emotion_tok = EMOTION_MAP.get(emotion_str)
        genre_tok   = GENRE_MAP.get(genre_str)

        if emotion_tok is None or genre_tok is None:
            return ("skip", pt_path.name)

        emotion_id = vocab.get(emotion_tok, unk_id)
        genre_id   = vocab.get(genre_tok, unk_id)

        ids = torch.load(pt_path, weights_only=True)

        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids, dtype=torch.long)

        # Skip if already tagged
        if len(ids) >= 2 and ids[0].item() == emotion_id and ids[1].item() == genre_id:
            return ("already", pt_path.name)

        prefix = torch.tensor([emotion_id, genre_id], dtype=torch.long)
        new_ids = torch.cat([prefix, ids])

        torch.save(new_ids, pt_path)

        return ("updated", pt_path.name)

    except Exception as e:
        return ("error", pt_path.name, str(e))

def main():
    pt_files = list(TOKENIZED_DIR.glob("XMIDI_*.pt"))
    print(f"Found {len(pt_files):,} files")

    if len(pt_files) == 0:
        print("⚠️ No files found — check your folder or filename pattern")
        return

    updated = skipped = already = errors = 0

    with mp.Pool(mp.cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(process_file, pt_files), total=len(pt_files)):
            status = result[0]

            if status == "updated":
                updated += 1
            elif status == "already":
                already += 1
            elif status == "skip":
                skipped += 1
            else:
                errors += 1
                print("Error:", result)

    print("\nDone.")
    print(f"Updated: {updated}")
    print(f"Already: {already}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()