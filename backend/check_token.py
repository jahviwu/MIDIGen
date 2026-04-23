from collections import Counter
import torch
from pathlib import Path
import torch.serialization
import pathlib

torch.serialization.add_safe_globals([pathlib.WindowsPath])

files = list(Path("data/midi_tokenized").glob("*.pt"))

counter = Counter()

for i in range(100):
    ids = torch.load(files[i], weights_only=True)

    if isinstance(ids, list):
        counter.update(ids)
    else:
        counter.update(ids.tolist())

print("Token 19 frequency:", counter[19])