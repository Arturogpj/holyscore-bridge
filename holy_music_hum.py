"""Hum / sing -> notes with pYIN (the same pitch tracker WanGP's Hum-to-Song uses) plus onset splitting.

transcribe(path, ...) returns
  notes: [{start, length, pitch}] in quarter notes from the song start
  trace: [[seconds, midi_or_None], ...] the raw pitch curve, for drawing what was heard
"""
import numpy as np

_NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _scale(key):
    key = (key or "C").strip()
    minor = key.endswith("m")
    root = key[:-1] if minor else key
    pc = _NOTE_PC.get(root[:1].upper(), 0) + root[1:].count("#") - root[1:].count("b")
    steps = (0, 2, 3, 5, 7, 8, 10) if minor else (0, 2, 4, 5, 7, 9, 11)
    return {(pc + s) % 12 for s in steps}


def _snap(pitch, scale):
    for d in range(7):
        for p in (pitch + d, pitch - d):
            if p % 12 in scale:
                return p
    return pitch


def transcribe(path, bpm=90.0, offset=0.0, latency=0.0, key="C", snap=True, tight=True, grid=0.25, transpose=0):
    import librosa

    y, sr = librosa.load(path, sr=22050, mono=True)
    if len(y) < sr * 0.2:
        raise ValueError("The recording is too short.")
    hop = 256
    f0, voiced, prob = librosa.pyin(y, fmin=65, fmax=1050, sr=sr, frame_length=1024, hop_length=hop)
    times = librosa.times_like(f0, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0][: len(f0)]
    gate = max(float(np.percentile(rms, 95)) * 0.12, 1e-4)
    midi = librosa.hz_to_midi(np.where(np.isfinite(f0), f0, np.nan))
    ok = voiced & (prob > 0.35) & (rms > gate) & np.isfinite(midi)
    # light median smoothing of the pitch curve inside voiced runs
    smooth = midi.copy()
    for i in range(len(midi)):
        window = midi[max(0, i - 2): i + 3]
        window = window[np.isfinite(window)]
        if ok[i] and len(window):
            smooth[i] = float(np.median(window))
    onset_frames = set(librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, backtrack=True, units="frames").tolist())

    segments = []
    current = None
    for i in range(len(smooth)):
        if not ok[i]:
            if current:
                segments.append(current)
            current = None
            continue
        if current is not None:
            center = float(np.median(current["pitches"][-8:]))
            # a new note: an attack (onset) after at least ~60 ms, or a stable pitch move
            new_attack = i in onset_frames and (i - current["i0"]) * hop / sr > 0.06
            moved = abs(smooth[i] - center) > 0.75 and all(abs(smooth[j] - center) > 0.6 for j in range(i, min(i + 3, len(smooth))) if ok[j])
            if new_attack or moved:
                segments.append(current)
                current = None
        if current is None:
            current = {"i0": i, "i1": i, "pitches": []}
        current["i1"] = i
        current["pitches"].append(float(smooth[i]))
    if current:
        segments.append(current)

    scale = _scale(key)
    spq = 60.0 / float(bpm)
    step = float(grid) if tight else 0.125
    notes = []
    for seg in segments:
        t0 = times[seg["i0"]]
        t1 = times[seg["i1"]] + hop / sr
        if t1 - t0 < 0.07:
            continue
        pitch = int(round(float(np.median(seg["pitches"])))) + int(transpose)
        if snap:
            pitch = _snap(pitch, scale)
        q0 = (t0 - latency) / spq + offset
        q1 = (t1 - latency) / spq + offset
        start = round(q0 / step) * step
        end = max(start + step, round(q1 / step) * step)
        if start < offset - 1e-6:
            continue
        if notes and start < notes[-1]["start"] + notes[-1]["length"]:
            last = notes[-1]
            if last["pitch"] == pitch and start - last["start"] < step * 1.01:
                last["length"] = max(last["length"], end - last["start"])
                continue
            last["length"] = max(step, start - last["start"])
            if start <= last["start"]:
                continue
        notes.append({"start": float(start), "length": float(end - start), "pitch": pitch})

    trace = [[round(float(times[i]), 3), round(float(smooth[i]), 2) if ok[i] else None] for i in range(0, len(times), 2)]
    return {"notes": notes, "trace": trace, "duration": round(len(y) / sr, 2)}
