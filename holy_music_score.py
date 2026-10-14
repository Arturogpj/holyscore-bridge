"""Holy Score Song document <-> YuE2 ABC score.

compile_song() feeds WanGP's own SheetSage2 serializer
(models/TTS/yue2/sheetsage2/notation_sheetsage2.generate_abc_from_data), so the
ABC is byte-for-byte the dialect YuE2 was trained on and is validated by it.
parse_abc() reads that dialect back (e.g. the score a render exported) into a
Song document the editor can show and change.

Song document (times in quarter notes from the start of the song):
{
  "title": "", "mode": "vocal" | "instrumental",
  "key": "Am", "meter": "4/4", "bpm": 90,
  "sections": [{"label": "verse", "bars": 8}, ...],          # sequential
  "chords": [{"start": 0, "symbol": "Am"}, ...],              # held until next
  "vocal": [{"start": 0, "length": 1, "pitch": 69}, ...],     # sung line
  "lead":  [{"start": 0, "length": 1, "pitch": 69}, ...],     # Ins voice
  "keyChanges": [{"bar": 16, "key": "C"}],                    # optional
  "meterChanges": [{"bar": 8, "meter": "3/4"}]                # optional
}
"""
from __future__ import annotations

import re
from fractions import Fraction

SECTION_LABELS = ("intro", "verse", "pre-chorus", "chorus", "post-chorus", "bridge",
                  "interlude", "solo", "instrumental", "outro", "theme", "development",
                  "variation", "pre-outro", "fade-out", "loop", "rap")

# Chord suffix (as written in the app / ABC) -> mir_eval quality used by SheetSage2.
_SUFFIX_TO_QUALITY = {
    "": "maj", "maj": "maj", "M": "maj",
    "m": "min", "min": "min", "-": "min",
    "7": "7", "maj7": "maj7", "M7": "maj7", "m7": "min7", "min7": "min7",
    "dim": "dim", "dim7": "dim7", "aug": "aug", "+": "aug",
    "m7b5": "hdim7", "ø": "hdim7",
    "sus2": "sus2", "sus4": "sus4", "sus": "sus4", "7sus4": "sus4(b7)",
    "6": "maj6", "m6": "min6", "m(maj7)": "minmaj7", "mmaj7": "minmaj7",
}
_QUALITY_TO_SUFFIX = {"maj": "", "min": "m", "7": "7", "maj7": "maj7", "min7": "m7", "dim": "dim",
                      "dim7": "dim7", "aug": "aug", "hdim7": "m7b5", "sus2": "sus2", "sus4": "sus4",
                      "sus4(b7)": "7sus4", "maj6": "6", "min6": "m6", "minmaj7": "m(maj7)"}
_CHORD_RE = re.compile(r"^([A-G](?:#{1,2}|b{1,2})?)(.*?)(?:/([A-G](?:#{1,2}|b{1,2})?))?$")
_NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


class SongError(ValueError):
    pass


def _meter(text):
    num, den = str(text).split("/")
    num, den = int(num), int(den)
    if num < 1 or den < 1 or den & (den - 1):
        raise SongError(f"Invalid meter {text!r}")
    return num, den


def chord_to_mir(symbol: str) -> str:
    """'Am7/G' -> 'A:min7/G' ; 'N' stays 'N'."""
    symbol = str(symbol or "").strip()
    if symbol in ("", "N", "NC", "N.C."):
        return "N"
    match = _CHORD_RE.fullmatch(symbol)
    if not match:
        raise SongError(f"Unrecognised chord {symbol!r}")
    root, suffix, bass = match.groups()
    quality = _SUFFIX_TO_QUALITY.get(suffix)
    if quality is None:
        raise SongError(f"Unsupported chord quality in {symbol!r} (try m, 7, maj7, m7, dim, aug, sus2, sus4, 6, m6, m7b5)")
    return f"{root}:{quality}" + (f"/{bass}" if bass else "")


def _bar_layout(song):
    """List of (start_quarter, length_quarters, beats, den) for every bar."""
    total_bars = sum(int(section.get("bars", 0)) for section in song.get("sections", []))
    if total_bars < 1:
        raise SongError("The song needs at least one section with bars.")
    changes = {int(change["bar"]): change["meter"] for change in song.get("meterChanges", []) or []}
    num, den = _meter(song.get("meter", "4/4"))
    bars, position = [], Fraction(0)
    for index in range(total_bars):
        if index in changes:
            num, den = _meter(changes[index])
        length = Fraction(4 * num, den)
        bars.append((position, length, num, den))
        position += length
    return bars


def score_seconds(song):
    """Length of the written score in seconds."""
    bars = _bar_layout(song)
    return float(bars[-1][0] + bars[-1][1]) * 60.0 / float(song.get("bpm", 90))


def song_rows(song):
    """Build the (midi, beats, chords, keys, structures) inputs for SheetSage2."""
    import pretty_midi

    bpm = float(song.get("bpm", 90))
    if not 20 <= bpm <= 300:
        raise SongError("Tempo must be between 20 and 300 BPM.")
    seconds_per_quarter = 60.0 / bpm
    sec = lambda quarters: float(quarters) * seconds_per_quarter
    bars = _bar_layout(song)
    end = bars[-1][0] + bars[-1][1]

    beats = []
    for start, length, num, den in bars:
        step = Fraction(4, den)
        for beat in range(num):
            beats.append([sec(start + step * beat), beat + 1, num, den])
    beats.append([sec(end), 1, bars[-1][2], bars[-1][3]])  # closing downbeat

    keys = [[0.0, sec(end), str(song.get("key", "C"))]]
    key_changes = sorted(song.get("keyChanges", []) or [], key=lambda change: int(change["bar"]))
    for change in key_changes:
        at = bars[int(change["bar"])][0]
        keys[-1][1] = sec(at)
        keys.append([sec(at), sec(end), str(change["key"])])
    keys = [row for row in keys if row[1] > row[0]]

    structures, bar_index = [], 0
    for section in song.get("sections", []):
        count = int(section.get("bars", 0))
        if count < 1:
            continue
        label = str(section.get("label", "verse")).strip().lower() or "verse"
        start = bars[bar_index][0]
        stop = bars[bar_index + count - 1][0] + bars[bar_index + count - 1][1]
        structures.append([sec(start), sec(stop), label])
        bar_index += count

    chords = []
    ordered = sorted(song.get("chords", []) or [], key=lambda chord: float(chord["start"]))
    for index, chord in enumerate(ordered):
        start = Fraction(chord["start"]).limit_denominator(48)
        stop = Fraction(ordered[index + 1]["start"]).limit_denominator(48) if index + 1 < len(ordered) else end
        stop = min(stop, end)
        if stop <= start or start >= end:
            continue
        symbol = chord_to_mir(chord.get("symbol"))
        if symbol != "N":
            chords.append([sec(start), sec(stop), symbol])

    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    for voice, key in (("Vocal", "vocal"), ("Ins", "lead")):
        instrument = pretty_midi.Instrument(program=0, name=voice)
        notes = sorted(song.get(key, []) or [], key=lambda note: (float(note["start"]), -float(note["length"])))
        cursor = Fraction(0)
        for note in notes:
            start = max(Fraction(note["start"]).limit_denominator(48), cursor)  # keep it monophonic
            stop = min(Fraction(note["start"]).limit_denominator(48) + Fraction(note["length"]).limit_denominator(48), end)
            if stop <= start:
                continue
            pitch = int(note["pitch"])
            if not 0 <= pitch <= 127:
                raise SongError(f"Pitch {pitch} out of MIDI range")
            instrument.notes.append(pretty_midi.Note(velocity=96, pitch=pitch, start=sec(start), end=sec(stop)))
            cursor = stop
        midi.instruments.append(instrument)
    return midi, beats, chords, keys, structures


def _notation():
    """WanGP's SheetSage2 notation module, without importing all TTS models."""
    import importlib.util
    import os
    import sys
    import types

    loaded = sys.modules.get("models.TTS.yue2.sheetsage2.notation_sheetsage2")
    if loaded is not None:
        return loaded
    name = "_holy_music_sheetsage2"
    if name + ".notation_sheetsage2" in sys.modules:
        return sys.modules[name + ".notation_sheetsage2"]
    app_dir = os.environ.get("WANGP_APP_DIR") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    folder = os.path.join(app_dir, "models", "TTS", "yue2", "sheetsage2")
    package = types.ModuleType(name)
    package.__path__ = [folder]
    sys.modules[name] = package
    for module in ("io_sheetsage2", "notation_sheetsage2"):
        spec = importlib.util.spec_from_file_location(f"{name}.{module}", os.path.join(folder, module + ".py"))
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = loaded
        spec.loader.exec_module(loaded)
    return loaded


def compile_song(song, melody_only=False):
    """Song document -> (abc_text, warnings). Raises SongError / AbcRebuildError."""
    generate_abc_from_data = _notation().generate_abc_from_data

    if song.get("mode") == "instrumental" and song.get("vocal"):
        raise SongError("Instrumental songs keep the melody in the lead voice; clear the vocal line first.")
    midi, beats, chords, keys, structures = song_rows(song)
    abc, score = generate_abc_from_data(midi, beats, chords, keys, structures, melody_only=melody_only)
    return abc, list(score.diagnostics)


# ---------------------------------------------------------------- parsing ---

_TOKEN_RE = re.compile(r'"(?P<chord>[^"]*)"|\[K:(?P<key>[^\]]+)\]|(?P<acc>[_=^]*)(?P<letter>[A-Ga-gz])(?P<oct>[,\']*)(?P<dur>\d*)(?P<tie>-?)')
_KEY_SIG = {"C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7, "F": -1, "Bb": -2, "Eb": -3,
            "Ab": -4, "Db": -5, "Gb": -6, "Cb": -7, "Am": 0, "Em": 1, "Bm": 2, "F#m": 3, "C#m": 4, "G#m": 5,
            "D#m": 6, "A#m": 7, "Dm": -1, "Gm": -2, "Cm": -3, "Fm": -4, "Bbm": -5, "Ebm": -6, "Abm": -7}


def _key_accidentals(key):
    count = _KEY_SIG.get(key.strip(), 0)
    order = "FCGDAEB" if count > 0 else "BEADGCF"
    return {letter: (1 if count > 0 else -1) for letter in order[:abs(count)]}


def _pitch(letter, octave_marks, accidental, key_acc, bar_acc):
    upper = letter.upper()
    octave = 5 if letter.islower() else 4  # MIDI octave index: C4 = 60 -> 'C'
    octave += octave_marks.count("'") - octave_marks.count(",")
    slot = upper  # like YuE2's writer: a bar accidental applies to the letter in every octave
    if accidental:
        shift = accidental.count("^") - accidental.count("_")
        bar_acc[slot] = shift
    else:
        shift = bar_acc.get(slot, key_acc.get(upper, 0))
    return 12 * (octave + 1) + _NOTE_PC[upper] + shift


def _chord_from_abc(text):
    """'Am7/G' as written in ABC -> app symbol (same spelling, validated)."""
    text = text.strip()
    match = _CHORD_RE.fullmatch(text)
    if not match:
        return None
    root, suffix, bass = match.groups()
    quality = _SUFFIX_TO_QUALITY.get(suffix)
    if quality is None:
        return None
    return root + _QUALITY_TO_SUFFIX[quality] + (f"/{bass}" if bass else "")


def parse_abc(abc: str) -> dict:
    """YuE2/SheetSage2 ABC -> Song document (best effort for hand-edited text)."""
    lines = [line.rstrip() for line in abc.replace("\r\n", "\n").split("\n")]
    header = {}
    body_start = 0
    for index, line in enumerate(lines):
        if re.match(r"^[A-Za-z]:", line) and not line.startswith("V:") or line.startswith("V: ") and "clef=" in line:
            header.setdefault(line[0], line[2:].strip())
            body_start = index + 1
            if line.startswith("K:"):
                break
    meter = header.get("M", "4/4")
    if meter in ("C", "C|"):
        meter = "4/4"
    unit = Fraction(header.get("L", "1/8"))
    tempo = re.search(r"=\s*(\d+(?:\.\d+)?)", header.get("Q", "")) if header.get("Q") else None
    tempo_beat = Fraction(re.match(r"\s*(\d+/\d+)", header.get("Q", "1/4=90")).group(1)) if header.get("Q") and "=" in header.get("Q", "") else Fraction(1, 4)
    bpm = float(tempo.group(1)) * float(tempo_beat / Fraction(1, 4)) if tempo else 90.0
    key = header.get("K", "C").split()[0]
    quarter_per_unit = unit * 4

    song = {"title": header.get("T", ""), "mode": "vocal", "key": key, "meter": meter, "bpm": round(bpm, 2),
            "sections": [], "chords": [], "vocal": [], "lead": [], "keyChanges": [], "meterChanges": []}
    voices = {"Vocal": {"pos": Fraction(0), "bar": 0, "notes": song["vocal"]},
              "Ins": {"pos": Fraction(0), "bar": 0, "notes": song["lead"]}}
    current_key = {"Vocal": key, "Ins": key}
    current_meter = meter
    pending_labels = []
    group_start_bar = 0
    voice = None
    last_chord = None
    open_ties = {"Vocal": None, "Ins": None}

    def bar_quarters():
        num, den = _meter(current_meter)
        return Fraction(4 * num, den)

    for line in lines[body_start:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("%"):
            label = stripped.lstrip("%").strip().lower()
            if label:
                pending_labels.append(label)
            continue
        if stripped.startswith("V:"):
            voice = "Vocal" if "Vocal" in stripped else "Ins" if "Ins" in stripped else None
            if voice == "Vocal":
                group_start_bar = voices["Vocal"]["bar"]
                if pending_labels:
                    for label in pending_labels:
                        song["sections"].append({"label": label, "startBar": group_start_bar})
                    pending_labels = []
            continue
        if stripped.startswith("M:"):
            new_meter = stripped[2:].strip()
            if voice == "Vocal" and new_meter != current_meter:
                song["meterChanges"].append({"bar": voices["Vocal"]["bar"], "meter": new_meter})
            current_meter = new_meter
            continue
        if stripped.startswith("K:"):
            new_key = stripped[2:].strip().split()[0]
            if voice == "Vocal" and new_key != current_key["Vocal"]:
                song["keyChanges"].append({"bar": voices["Vocal"]["bar"], "key": new_key})
            if voice:
                current_key[voice] = new_key
            continue
        if voice is None or re.match(r"^[A-Za-z]:", stripped):
            continue
        state = voices[voice]
        for bar_text in stripped.split("|")[:-1] if stripped.endswith("|") else stripped.split("|"):
            bar_text = bar_text.strip()
            if not bar_text:
                continue
            full_rest = re.fullmatch(r"Z(\d*)", bar_text)
            if full_rest:
                count = int(full_rest.group(1) or 1)
                state["pos"] += bar_quarters() * count
                state["bar"] += count
                open_ties[voice] = None
                continue
            bar_start = state["pos"]
            bar_acc = {}
            key_acc = _key_accidentals(current_key[voice])
            cursor = bar_start
            for match in _TOKEN_RE.finditer(bar_text):
                if match.group("chord") is not None:
                    symbol = _chord_from_abc(match.group("chord"))
                    if voice == "Vocal" and symbol and symbol != last_chord:
                        song["chords"].append({"start": float(cursor), "symbol": symbol})
                        last_chord = symbol
                    continue
                if match.group("key") is not None:
                    current_key[voice] = match.group("key").strip()
                    key_acc = _key_accidentals(current_key[voice])
                    if voice == "Vocal":
                        song["keyChanges"].append({"bar": state["bar"], "key": current_key[voice]})
                    continue
                length = int(match.group("dur") or 1) * quarter_per_unit
                if match.group("letter") == "z":
                    open_ties[voice] = None
                    cursor += length
                    continue
                pitch = _pitch(match.group("letter"), match.group("oct"), match.group("acc"), key_acc, bar_acc)
                tied = open_ties[voice]
                if tied is not None and tied["pitch"] == pitch and abs(Fraction(tied["start"]).limit_denominator(96) + Fraction(tied["length"]).limit_denominator(96) - cursor) < Fraction(1, 96):
                    tied["length"] = float(Fraction(tied["length"]).limit_denominator(96) + length)
                    note = tied
                else:
                    note = {"start": float(cursor), "length": float(length), "pitch": pitch}
                    state["notes"].append(note)
                open_ties[voice] = note if match.group("tie") else None
                cursor += length
            state["pos"] = bar_start + bar_quarters()
            state["bar"] += 1

    total_bars = max(voices["Vocal"]["bar"], voices["Ins"]["bar"])
    starts = song["sections"]
    if not starts or starts[0]["startBar"] > 0:
        starts.insert(0, {"label": "intro" if starts else "verse", "startBar": 0})
    merged = []
    for section in starts:  # several labels on one bar: keep the last ("intro and verse" style)
        if merged and merged[-1]["startBar"] == section["startBar"]:
            merged[-1] = section
        else:
            merged.append(section)
    song["sections"] = [
        {"label": section["label"], "bars": (merged[index + 1]["startBar"] if index + 1 < len(merged) else total_bars) - section["startBar"]}
        for index, section in enumerate(merged)
    ]
    song["sections"] = [section for section in song["sections"] if section["bars"] > 0]
    if not song["vocal"] and song["lead"]:
        song["mode"] = "instrumental"
    if not song["keyChanges"]:
        song.pop("keyChanges")
    if not song["meterChanges"]:
        song.pop("meterChanges")
    return song
