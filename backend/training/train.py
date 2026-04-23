import sys
import time
from pathlib import Path
import os
import shutil
import pathlib
 
import torch
import torch.nn as nn
import torch.serialization
from torch.utils.data import Dataset, DataLoader
 
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
 
from models.transformer_model import MusicTransformer, save_model, load_model
 
TOKENIZED_DIR  = BASE / "data" / "midi_tokenized"
CHECKPOINT_DIR = BASE / "training" / "checkpoints"
LOSS_LOG       = BASE / "training" / "loss_log.txt"
 
# Model config for RTX 5080 (16GB VRAM)
MODEL_CONFIG = {
    "vocab_size":   410,
    "d_model":      512,
    "n_heads":      8,
    "n_layers":     8,
    "d_ff":         2048,
    "max_seq_len":  1024,
    "dropout":      0.1,
    "pad_token_id": 0,
}
 
# Training hyperparameters
SEQ_LEN          = 512
BATCH_SIZE       = 128
GRAD_ACCUM_STEPS = 1
EPOCHS           = 12
LEARNING_RATE    = 3e-4
WARMUP_STEPS     = 1000
MAX_GRAD_NORM    = 1.0
SAVE_EVERY       = 1
SAVE_STEPS       = 5000
PAD_TOKEN_ID     = 0
 
 
class MidiDataset(Dataset):
 
    def __init__(self, tokenized_dir: Path, seq_len: int):
        self.seq_len = seq_len
        self.chunks  = []
 
        torch.serialization.add_safe_globals([pathlib.WindowsPath])
 
        cache_path = tokenized_dir / f"chunk_cache_seqlen{seq_len}.pt"
 
        if cache_path.exists():
            print(f"Loading cached chunk index from {cache_path}")
            self.chunks = torch.load(cache_path, weights_only=False)
            print(f"Loaded {len(self.chunks):,} chunks from cache.")
        else:
            pt_files = list(tokenized_dir.glob("*.pt"))
            if not pt_files:
                raise FileNotFoundError(f"No .pt files found in {tokenized_dir}\nRun data/preprocess.py first.")
 
            print(f"Scanning {len(pt_files):,} tokenized files...")
 
            for path in pt_files:
                try:
                    ids = torch.load(path, weights_only=True)
                    length = len(ids)
                    if length >= seq_len + 1:
                        for start in range(0, length - seq_len, seq_len):
                            self.chunks.append((path, start))
                except Exception:
                    pass
 
            if len(self.chunks) == 0:
                raise RuntimeError(
                    f"No chunks found with seq_len={seq_len}. "
                    f"Try reducing SEQ_LEN (e.g. back to 256)."
                )
 
            print(f"Total training chunks: {len(self.chunks):,}")
            torch.save(self.chunks, cache_path)
            print(f"Index cached to: {cache_path}")
 
        # ── Preload all tokenized files into RAM ──
        # With 64GB RAM this fits easily and eliminates all disk I/O during training
        print("Preloading tokenized files into RAM (one-time, takes a few minutes)...")
        preload_start = time.time()
        unique_paths = set(path for path, _ in self.chunks)
        self.ram_cache = {}
        for i, path in enumerate(unique_paths):
            try:
                ids = torch.load(path, weights_only=True)
                if isinstance(ids, list):
                    ids = torch.tensor(ids, dtype=torch.long)
                self.ram_cache[path] = ids
            except Exception:
                pass
            if (i + 1) % 10000 == 0:
                print(f"  Preloaded {i+1:,}/{len(unique_paths):,} files...")
        print(f"Preload complete in {time.time()-preload_start:.1f}s — {len(self.ram_cache):,} files in RAM.")
 
    def __len__(self):
        return len(self.chunks)
 
    def __getitem__(self, idx):
        path, start = self.chunks[idx]
        ids = self.ram_cache.get(path)
        if ids is None:
            raise KeyError(f"File not in RAM cache: {path}")
        chunk = ids[start : start + self.seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
 
 
def collate_fn(batch):
    xs, ys = zip(*batch)
    xs_padded = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=PAD_TOKEN_ID)
    ys_padded = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=PAD_TOKEN_ID)
    pad_mask  = (xs_padded == PAD_TOKEN_ID)
    return xs_padded, ys_padded, pad_mask
 
 
def get_lr(step: int, d_model: int, warmup_steps: int) -> float:
    if step == 0:
        return 1e-9
    scale = d_model ** -0.5
    return scale * min(step ** -0.5, step * warmup_steps ** -1.5)
 
 
def save_ckpt(path, model, optimizer, epoch, step, global_step):
    tmp = Path(str(path) + ".tmp")
    torch.save({
        "config":      model.cfg,
        "model_state": model.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "epoch":       epoch,
        "step":        step,
        "global_step": global_step,
    }, tmp)
    shutil.move(str(tmp), str(path))
 
 
def load_checkpoint(model, optimizer, path):
    if not os.path.exists(path):
        print("No checkpoint found, starting fresh.")
        return 0, 0, 0
    checkpoint = torch.load(path, map_location="cuda")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch       = checkpoint.get("epoch", 0)
    step        = checkpoint.get("step", 0)
    global_step = checkpoint.get("global_step", 0)
    print(f"Resuming from epoch {epoch + 1}, step {step}, global_step {global_step}")
    return epoch, step, global_step
 
 
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")
 
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
 
    dataset = MidiDataset(TOKENIZED_DIR, SEQ_LEN)
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
 
    model     = MusicTransformer(config=MODEL_CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scaler    = torch.amp.GradScaler('cuda')
 
    latest_path = CHECKPOINT_DIR / "latest.pt"
    start_epoch, start_step, global_step = load_checkpoint(model, optimizer, path=latest_path)
 
    print(f"Parameters: {model.num_parameters():,}")
 
    log_file  = open(LOSS_LOG, "a")
    best_loss = float("inf")
 
    epoch = start_epoch
    step  = 0
 
    import atexit, signal
 
    def emergency_save():
        try:
            save_ckpt(CHECKPOINT_DIR / "latest.pt", model, optimizer, epoch, step, global_step)
            print(f"\nEmergency checkpoint saved (epoch {epoch + 1}, step {step}).")
        except Exception as e:
            print(f"\nEmergency save failed: {e}")
 
    atexit.register(emergency_save)
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
 
    try:
        for epoch in range(start_epoch, EPOCHS):
            model.train()
 
            epoch_loss    = 0.0
            steps_trained = 0
            epoch_start   = time.time()
            optimizer.zero_grad()
 
            for step, (x, y, pad_mask) in enumerate(loader):
 
                if epoch == start_epoch and step < start_step:
                    continue
 
                x        = x.to(device)
                y        = y.to(device)
                pad_mask = pad_mask.to(device)

                # Check first batch tokens
                if global_step == 0 and epoch == start_epoch and step == start_step:
                    print("\nDEBUG sample tokens:", x[0][:10])
 
                lr = get_lr(global_step + 1, MODEL_CONFIG["d_model"], WARMUP_STEPS)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
 
                with torch.amp.autocast('cuda'):
                    logits = model(x, key_padding_mask=pad_mask)
                    loss   = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, MODEL_CONFIG["vocab_size"]),
                        y.reshape(-1),
                        ignore_index=PAD_TOKEN_ID,
                    )
 
                scaler.scale(loss / GRAD_ACCUM_STEPS).backward()
 
                if (step + 1) % GRAD_ACCUM_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
 
                epoch_loss    += loss.item()
                steps_trained += 1
                global_step   += 1
 
                if (step + 1) % 100 == 0:
                    avg = epoch_loss / steps_trained
                    print(f"  Epoch {epoch+1} | Step {step+1}/{len(loader)} | Loss {avg:.4f} | LR {lr:.2e}")
 
                if (step + 1) % SAVE_STEPS == 0:
                    save_ckpt(CHECKPOINT_DIR / "latest.pt", model, optimizer, epoch, step + 1, global_step)
                    print(f"  Checkpoint saved at step {step + 1}")
 
            avg_loss = epoch_loss / steps_trained if steps_trained > 0 else float("nan")
            elapsed  = time.time() - epoch_start
            log_line = f"Epoch {epoch+1:03d} | Loss {avg_loss:.4f} | Time {elapsed:.1f}s"
            print(f"\n{log_line}\n")
            log_file.write(log_line + "\n")
            log_file.flush()
 
            start_step = 0
 
            if (epoch + 1) % SAVE_EVERY == 0:
                save_ckpt(CHECKPOINT_DIR / "latest.pt", model, optimizer, epoch, 0, global_step)
                print(f"  Epoch checkpoint saved: checkpoints/latest.pt")
 
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_ckpt(CHECKPOINT_DIR / "best_model.pt", model, optimizer, epoch, 0, global_step)
                print(f"  Best model updated: checkpoints/best_model.pt (loss {best_loss:.4f})")
 
    except KeyboardInterrupt:
        print("\nCTRL-C detected — emergency_save will run on exit.")
 
    log_file.close()
    print("Training complete.")
 
 
if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    train()