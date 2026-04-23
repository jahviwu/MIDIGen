import pretty_midi

# Variable-length time shift buckets (in seconds)
TIME_SHIFT_BUCKETS = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]  # 10–640 ms
TIME_SHIFT_MS = [int(t * 1000) for t in TIME_SHIFT_BUCKETS]


def quantize_time_shift(dt: float):

    #Decompose a time delta (seconds) into a sequence of TIME_SHIFT_* tokens using the defined buckets (greedy, largest-first).

    tokens = []
    for bucket, ms in reversed(list(zip(TIME_SHIFT_BUCKETS, TIME_SHIFT_MS))):
        while dt >= bucket - 1e-6:
            tokens.append(f"TIME_SHIFT_{ms}")
            dt -= bucket
    return tokens


def midi_to_tokens(pm):

    # Accepts PrettyMIDI object, Path object, and a string path
    # Returns a flat token sequence: TIME_SHIFT_*, VELOCITY_*, NOTE_ON_*, NOTE_OFF_*

    # ✅ FIX: ensure pm is a PrettyMIDI object
    if isinstance(pm, (str, bytes)):
        pm = pretty_midi.PrettyMIDI(pm)
    elif hasattr(pm, "as_posix"):  # handles Path objects
        pm = pretty_midi.PrettyMIDI(str(pm))

    tokens = []

    # Collect note events
    events = []
    for inst in pm.instruments:
        for note in inst.notes:
            events.append(("on", note.start, note.pitch, note.velocity))
            events.append(("off", note.end, note.pitch, note.velocity))

    # Sort by time
    events.sort(key=lambda x: x[1])

    last_time = 0.0

    for event_type, t, pitch, velocity in events:
        dt = t - last_time
        if dt > 0:
            tokens.extend(quantize_time_shift(dt))

        # ✅ Only include velocity for NOTE_ON (more efficient)
        if event_type == "on":
            tokens.append(f"VELOCITY_{velocity}")
            tokens.append(f"NOTE_ON_{pitch}")
        else:
            tokens.append(f"NOTE_OFF_{pitch}")

        last_time = t

    return tokens


def tokens_to_midi(tokens):
    
    # Convert token sequence back into a PrettyMIDI object.
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)  # piano

    current_time = 0.0
    current_velocity = 64
    active_notes = {}  # pitch -> start_time

    for tok in tokens:
        if tok.startswith("TIME_SHIFT_"):
            ms = int(tok.split("_")[2])
            current_time += ms / 1000.0

        elif tok.startswith("VELOCITY_"):
            current_velocity = int(tok.split("_")[1])

        elif tok.startswith("NOTE_ON_"):
            pitch = int(tok.split("_")[2])
            active_notes[pitch] = current_time

        elif tok.startswith("NOTE_OFF_"):
            pitch = int(tok.split("_")[2])
            if pitch in active_notes:
                start = active_notes[pitch]
                end = current_time
                if end > start:
                    note = pretty_midi.Note(
                        velocity=current_velocity,
                        pitch=pitch,
                        start=start,
                        end=end,
                    )
                    inst.notes.append(note)
                del active_notes[pitch]

    pm.instruments.append(inst)
    return pm


def save_midi(pm: pretty_midi.PrettyMIDI, path):
    pm.write(str(path))