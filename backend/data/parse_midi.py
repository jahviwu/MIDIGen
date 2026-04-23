import pretty_midi
import json
from pathlib import Path
from tokenizer import midi_to_tokens

raw_dir = Path("project/data/midi_raw")
tok_dir = Path("project/data/midi_tokenized")
tok_dir.mkdir(exist_ok=True)

for midi_path in raw_dir.glob("*.mid"):
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        tokens = midi_to_tokens(pm)
        out = tok_dir / (midi_path.stem + ".json")
        out.write_text(json.dumps(tokens))
        print("OK:", midi_path.name)
    except Exception as e:
        print("Failed:", midi_path.name, e)
