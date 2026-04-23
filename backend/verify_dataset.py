import json
import random
from pathlib import Path
from collections import Counter

import torch
import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parent
TOKENIZED_DIR = BASE / "data" / "midi_tokenized"
VOCAB_PATH    = BASE / "data" / "vocab.json"

SAMPLE_SIZE = 100        # how many files to analyze
MIN_SEQ_LEN = 10         # flag very short sequences
MAX_TIME_SHIFT_WARN = 1000  # arbitrary large shift warning


def load_vocab():
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    id_to_token = {v: k for k, v in vocab.items()}
    return vocab, id_to_token


def analyze_file(path, id_to_token):
    try:
        ids = torch.load(path, weights_only=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load file: {e}")

    if not isinstance(ids, torch.Tensor):
        ids = torch.tensor(ids, dtype=torch.long)

    tokens = []
    for t in ids.tolist():
        if int(t) not in id_to_token:
            raise ValueError(f"Unknown token id: {t}")
        tokens.append(id_to_token[int(t)])

    return tokens


def compute_statistics(tokens):
    counter = Counter(tokens)

    total = sum(counter.values())
    note_on  = sum(v for k, v in counter.items() if k.startswith("NOTE_ON"))
    note_off = sum(v for k, v in counter.items() if k.startswith("NOTE_OFF"))
    velocity = sum(v for k, v in counter.items() if k.startswith("VELOCITY"))
    timeshift = sum(v for k, v in counter.items() if k.startswith("TIME_SHIFT"))

    return {
        "total_tokens": total,
        "note_on": note_on,
        "note_off": note_off,
        "velocity": velocity,
        "timeshift": timeshift,
        "note_on_pct": note_on / total * 100 if total else 0,
        "timeshift_pct": timeshift / total * 100 if total else 0,
        "unique_tokens": len(counter),
        "counter": counter,
    }


def checks(path, tokens, stats):
    issues = []

    # 1. Sequence too short
    if len(tokens) < MIN_SEQ_LEN:
        issues.append("VERY_SHORT_SEQUENCE")

    # 2. NOTE_ON vs NOTE_OFF imbalance
    if abs(stats["note_on"] - stats["note_off"]) > 0.1 * max(1, stats["note_on"]):
        issues.append("NOTE_ON_OFF_IMBALANCE")

    # 3. Missing velocities
    if stats["velocity"] == 0:
        issues.append("NO_VELOCITY_TOKENS")

    # 4. Suspicious TIME_SHIFT
    large_shifts = [t for t in tokens if t.startswith("TIME_SHIFT_")]
    for t in large_shifts:
        try:
            val = int(t.split("_")[-1])
            if val > MAX_TIME_SHIFT_WARN:
                issues.append(f"LARGE_TIME_SHIFT_{val}")
                break
        except:
            continue

    return issues


def plot_histogram(counter, title="Token Frequency Histogram"):
    items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    labels = [k for k, v in items[:50]]
    values = [v for k, v in items[:50]]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def main():
    vocab, id_to_token = load_vocab()

    pt_files = list(TOKENIZED_DIR.glob("XMIDI_*.pt"))
    if not pt_files:
        print("No tokenized files found.")
        return

    print(f"Found {len(pt_files):,} tokenized files.")

    sample_files = random.sample(pt_files, min(SAMPLE_SIZE, len(pt_files)))
    print(f"Analyzing {len(sample_files)} random files...\n")

    global_counter = Counter()
    all_lengths = []

    corrupted_files = []
    issue_counts = Counter()

    for path in sample_files:
        print(f"→ {path.name}")

        try:
            tokens = analyze_file(path, id_to_token)
        except Exception as e:
            print(f"  Corrupted: {e}")
            corrupted_files.append(path)
            continue

        stats = compute_statistics(tokens)
        issues = checks(path, tokens, stats)

        all_lengths.append(len(tokens))
        global_counter.update(stats["counter"])

        print(f"  Tokens:          {len(tokens):,}")
        print(f"  NOTE_ON %:       {stats['note_on_pct']:.2f}%")
        print(f"  TIME_SHIFT %:    {stats['timeshift_pct']:.2f}%")
        print(f"  Unique tokens:   {stats['unique_tokens']}")

        if issues:
            print(f"  ⚠ Issues: {', '.join(issues)}")
            for i in issues:
                issue_counts[i] += 1

        print()

    # ===== SUMMARY =====
    print("\nDATASET SUMMARY: ")

    if all_lengths:
        print(f"Avg length:   {sum(all_lengths)/len(all_lengths):.2f}")
        print(f"Max length:   {max(all_lengths)}")
        print(f"Min length:   {min(all_lengths)}")

    total = sum(global_counter.values())
    note_on = sum(v for k, v in global_counter.items() if k.startswith("NOTE_ON"))
    timeshift = sum(v for k, v in global_counter.items() if k.startswith("TIME_SHIFT"))

    if total > 0:
        print(f"\nTotal tokens:     {total:,}")
        print(f"NOTE_ON %:        {note_on / total * 100:.2f}%")
        print(f"TIME_SHIFT %:     {timeshift / total * 100:.2f}%")

    print(f"Unique tokens:    {len(global_counter)}")

    print(f"\nCorrupted files:  {len(corrupted_files)}")

    if issue_counts:
        print("\nIssue breakdown:")
        for k, v in issue_counts.items():
            print(f"  {k}: {v}")

    print("\nPlotting histogram...")
    plot_histogram(global_counter, title="Token Frequency (Top 50)")


if __name__ == "__main__":
    main()