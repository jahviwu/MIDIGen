import sys
import json
from pathlib import Path
import argparse

import torch
import pretty_midi

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from models.transformer_model import load_model
from tokenizer import tokens_to_midi, save_midi
from generation.prompt_parser import parse_prompt  # ← new import

CHECKPOINT = BASE / "training" / "checkpoints" / "best_model.pt"
VOCAB_PATH = BASE / "data" / "vocab.json"
OUTPUT_DIR = BASE / "generation" / "outputs"

DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_K = 40
DEFAULT_LENGTH = 1000

# Sampling
def top_k_sample(logits, top_k, temperature):
    logits = logits / temperature

    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        cutoff = values[-1]
        logits[logits < cutoff] = float("-inf")

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()

# Prompt to token conversion
def map_prompt_to_tokens(emotion, genre, vocab):
    tokens = []

    if emotion:
        emo_tok = f"<EMOTION_{emotion.upper()}>"
        if emo_tok in vocab:
            tokens.append(vocab[emo_tok])
        else:
            print(f"Warning: emotion token '{emo_tok}' not found in vocab")

    if genre:
        gen_tok = f"<GENRE_{genre.upper()}>"
        if gen_tok in vocab:
            tokens.append(vocab[gen_tok])
        else:
            print(f"Warning: genre token '{gen_tok}' not found in vocab")

    return tokens

# Generation
def generate(prompt, length, temperature, top_k, output_path, use_groq=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load vocab
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)

    id_to_token = {v: k for k, v in vocab.items()}
    pad_id = vocab.get("<PAD>", 0)
    unk_id = vocab.get("<UNK>", 1)

    print(f"Vocabulary loaded: {len(vocab)} tokens")

    # Load model
    model = load_model(CHECKPOINT, device=device)
    model.eval()
    print(f"Loaded model: {CHECKPOINT}")

    # Debug: Inspect checkpoint metadata
    try:
        ckpt = torch.load(CHECKPOINT, map_location="cpu")
        epoch = ckpt.get("epoch", "unknown")
        loss  = ckpt.get("loss", ckpt.get("val_loss", "unknown"))
        print(f"[DEBUG] Checkpoint epoch: {epoch}, loss: {loss}")
    except Exception as e:
        print(f"[DEBUG] Could not read checkpoint metadata: {e}")

    # Parse natural language prompt → emotion + genre
    print(f"\nParsing prompt: \"{prompt}\"")
    emotion, genre = parse_prompt(prompt, use_groq=use_groq)
    print(f"  Emotion: {emotion or '(none)'}")
    print(f"  Genre:   {genre   or '(none)'}")

    # Convert emotion/genre → conditioning token IDs
    prompt_token_ids = map_prompt_to_tokens(emotion, genre, vocab)

    if not prompt_token_ids:
        print("No valid emotion/genre tags found — using fallback <EMOTION_QUIET>")
        prompt_token_ids = [vocab["<EMOTION_QUIET>"]]
    else:
        id_to_token = {v: k for k, v in vocab.items()}
        print("Conditioning tokens:", [id_to_token[t] for t in prompt_token_ids])

    # Start sequence with conditioning tokens only
    sequence = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)

    print(f"\nGenerating {length} tokens (temp={temperature}, top_k={top_k})")

    max_ctx = model.cfg["max_seq_len"]

    with torch.no_grad():
        for step in range(length):
            context = sequence[:, -max_ctx:]

            logits = model(context)
            next_logits = logits[0, -1, :]

            # Never generate PAD or UNK
            next_logits[pad_id] = float("-inf")
            next_logits[unk_id] = float("-inf")

            next_id = top_k_sample(next_logits, top_k, temperature)

            sequence = torch.cat(
                [sequence, torch.tensor([[next_id]], device=device)], dim=1
            )

            if (step + 1) % 100 == 0:
                print(f"  Step {step+1}/{length} - last token: {id_to_token.get(next_id, '?')}")

    # Convert IDs → token strings
    generated_ids = sequence[0].tolist()
    generated_tokens = [id_to_token.get(i, "<UNK>") for i in generated_ids]

    print("\nReconstructing MIDI...")
    pm = tokens_to_midi(generated_tokens)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_midi(pm, output_path)

    print("\nDone.")
    print(f"  Output:   {output_path}")
    print(f"  Duration: {pm.get_end_time():.1f} seconds")
    print(f"  Notes:    {sum(len(inst.notes) for inst in pm.instruments)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt",      type=str,   default="")
    parser.add_argument("--length",      type=int,   default=DEFAULT_LENGTH)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_k",       type=int,   default=DEFAULT_TOP_K)
    parser.add_argument("--output",      type=str,   default="generation/outputs/output.mid")
    parser.add_argument("--groq",        action="store_true", help="Use Groq for smarter prompt parsing")
    args = parser.parse_args()

    generate(
        prompt=args.prompt,
        length=args.length,
        temperature=args.temperature,
        top_k=args.top_k,
        output_path=Path(args.output),
        use_groq=args.groq,
    )