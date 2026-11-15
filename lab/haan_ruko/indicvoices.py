"""Real people for the haan vs ruko model: Hindi phone conversations from AI4Bharat's
IndicVoices (CC BY 4.0, https://huggingface.co/datasets/ai4bharat/IndicVoices).

Why: the first model learned from 5 machine voices and failed on short clips. IndicVoices'
conversations are two people on a real phone line (8 kHz, upsampled to 16 kHz in the files),
cut into pieces at pauses, with what was said written down. A piece that is only "हाँ",
"जी", "अच्छा"... is a real listening sound; any other piece is real speech. One file of 82
had 372 such listening pieces from 128 speakers (11 Nov 2026).

Speakers, not pieces, are split: a test speaker is never heard in training, so the test says
how the model does on real people it never met.

Files 0-9 were the first import: one wav per piece, under IV_SOUNDS. Files 10-81 (14 Nov 2026)
are cut one at a time and their raw download deleted at once, so the disk never holds more
than a file or two; what they keep is one packed FLAC per file, outside OneDrive (MORE).
"""

from __future__ import annotations

import io
import json
import os
import time
import zlib
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile

from common.audio.resample import PHONE_RATE, resample
from lab.haan_ruko.clips import SOUNDS
from lab.haan_ruko.phrases import KEEP, STOP
from lab.haan_ruko.speech import NO_GAP_OPENING_S, Timing

REPO = "ai4bharat/IndicVoices"
IV_SOUNDS = SOUNDS.parent / "haan_ruko_indicvoices"
# Listening sounds as IndicVoices writes them (Hindi letters). Only whole pieces that are
# exactly one of these count; "हम" is left out, because it also means "we".
LISTENING = {
    "हाँ", "हां", "हा", "जी", "जी हाँ", "जी हां", "हाँ जी", "हां जी", "जी जी", "जी जी जी", "हाँ हाँ", "हां हां",
    "हाँ हाँ हाँ", "अच्छा", "अच्छा अच्छा", "अच्छा जी", "अच्छा हाँ", "ठीक है", "ठीक", "ठीक है जी", "ठीक है ठीक है",
    "ओके", "ओके ओके", "ओके जी", "हम्म", "ह्म्म", "हूँ", "हुँ", "बिल्कुल", "हाँ बिल्कुल", "जी बिल्कुल", "सही है", "सही",
    "हाँ सही", "राइट", "यस", "यस यस", "हाँ ठीक है", "अच्छा ठीक है", "हाँ बोलिए", "जी बोलिए", "हाँ बोलो", "ओह अच्छा",
    "ओके सर", "जी सर", "जी मैडम", "हाँ सर",
    # Added 11 Nov 2026: forms the first list missed, found among the short pieces labelled
    # speech that the model was surest were listening sounds. Labelled speech, they set the
    # strict line to almost zero. Greetings, questions ("आँय" is "huh?"), sentence openers
    # ("हाँ तो") and thanks stay speech.
    "ठीक ठीक", "ठीक ठीक ठीक", "ठीक है ठीक", "ठीक हे", "हाँ ठीक", "हाँ ठीक हे", "हाँ हाँ ठीक है", "जी मैम", "जी जी सर",
    "ठीक है सर", "ठीक है मैम", "अच्छा सर", "अच्छा मैम", "हाँ मैम", "हाँ हाँ सर", "यस सर", "यस मैम", "ओके मैम",
    "अच्छा अच्छा अच्छा", "हाँ जी हाँ", "हांजी", "हँ", "हं", "हूं", "हूँ हूँ", "आँ", "ह", "बोलिए", "बोलिए सर", "हाँ जी बोलिए",
}
# Added 14 Nov 2026: a word rule on top of the list. Listening sounds come in endless
# combinations ("ठीक है सर ठीक है", "जी जी बिल्कुल", "ओके ओके ठीक है"): about 150 such forms
# were still labelled speech after the first list was extended. A piece is a listening sound
# when every word is a listening word (polite words allowed alongside, not alone).
LISTENING_WORDS = {
    "हाँ", "हां", "हा", "हान", "हांजी", "जी", "अच्छा", "ठीक", "ओके", "ओक्के", "यस", "राइट", "हम्म", "ह्म्म", "हूँ",
    "हूं", "हुँ", "हँ", "हं", "आँ", "बिल्कुल", "सही", "बोलिए", "बोलो",
}
POLITE_WORDS = {"सर", "मैम", "मैडम", "भैया", "भइया", "भईया"}
# "है" only right after these: "ठीक है" listens, "हाँ है" ("yes, there is") answers a question.
IS_AFTER = {"ठीक", "सही"}
IS_WORDS = {"है", "हे"}
# "No" and "stop" words, for practising them more (train.py), never for deciding live: the
# model hears sound, not words. A short "nahi" is as short as a "haan", and only 58 of 13,122
# pieces were just such a word (14 Nov 2026): it missed "nahi" on the user's own voice.
NO_WORDS = {"नहीं", "नही", "नई", "ना", "नो", "मत", "रुको", "रुकिए", "रुकिये", "वेट", "स्टॉप"}
SHORTEST_S = 0.25  # shorter pieces were cut so tight the word itself is clipped
LONGEST_LISTENING_S = 2.0
KEEP_UP_TO_S = 2.5  # the model never hears more than 2 s after the first sound
# Speech is plentiful and listening is rare, so speech is sampled: short speech first, as
# short clips are exactly where the first model failed.
SHORT_SPEECH_S = 1.5
SPEECH_PER_LISTENING = 2.0
TEST_SPEAKERS, CHECK_SPEAKERS = 15, 10  # out of 100, by a fixed hash of the speaker
HINDI_FILES = 82
FIRST_IMPORT = 10  # files 0-9, in IV_SOUNDS
# Files 50-60 hold no phone conversations at all, only reading and talking alone (their
# "scenario" column, read without downloading them, 14 Nov 2026): skip them.
NO_CONVERSATIONS = range(50, 61)
# Files 10-81 live outside OneDrive: OneDrive would upload every clip to the user's cloud
# space (14 Nov 2026). HEARKEN_DATA moves the folder.
MORE = Path(os.environ.get("HEARKEN_DATA") or Path.home() / "hearken_data") / "indicvoices"
# English listening words. The model took the user's "ok ok" and "right" for speech (14 Nov
# 2026) and they are rare in Hindi calls (176 of 14,154 pieces), so every one is used.
ENGLISH_LISTENING_WORDS = {"ओके", "ओक्के", "यस", "राइट"}
# From each newer file, this many pieces go into training, beside every English listening
# piece and every short "no". 72 files x 195 is about the first import's 14,154: kept as
# 16-bit sound (half of float), the two together take the memory the first import took.
NEW_PER_FILE = 195


@dataclass(frozen=True)
class Piece:
    id: str
    speaker: str
    answer: str
    text: str
    duration: float
    role: str  # train, check or test
    opens_with_listening: bool
    file: str


def role_of(speaker: str) -> str:
    bucket = zlib.crc32(speaker.encode()) % 100
    return "test" if bucket < TEST_SPEAKERS else "check" if bucket < TEST_SPEAKERS + CHECK_SPEAKERS else "train"


def clean(text: str) -> str:
    return " ".join(text.replace("।", " ").replace(",", " ").replace(".", " ").replace("?", " ").split())


def is_listening(words: list[str]) -> bool:
    """Only listening words, by the list or by the word rule above."""
    if not words:
        return False
    if " ".join(words) in LISTENING:
        return True
    for at, word in enumerate(words):
        if word in IS_WORDS:
            if at == 0 or words[at - 1] not in IS_AFTER:
                return False
        elif word not in LISTENING_WORDS and word not in POLITE_WORDS:
            return False
    return any(word in LISTENING_WORDS for word in words)


def answer_for(text: str, duration: float) -> str | None:
    """keep, stop, or None for a piece we can't label fairly."""
    words = clean(text)
    if not words or "<" in words or "[" in words or duration < SHORTEST_S:
        return None
    if is_listening(words.split()):
        return KEEP if duration <= LONGEST_LISTENING_S else None
    return STOP


def opens_with_listening(text: str) -> bool:
    """Speech that starts like a listening sound ("हाँ जी, मेरा ऑर्डर..."). A piece that is
    only a listening phrase, however many words ("हाँ जी"), is not speech at all."""
    words = clean(text).split()
    if is_listening(words):
        return False
    return any(is_listening(words[:n]) for n in (1, 2, 3) if n < len(words))


def is_short_no(clip: dict) -> bool:
    """A short real piece of speech that starts with a "no" or "stop" word: "नहीं", "नहीं
    नहीं", "रुको...". The first word, not any word: on 14 Nov 2026 "any word" also caught tags
    and reassurance, "है ना" ("isn't it?"), "ठीक है ना", "कोई बात नहीं" ("no problem"),
    which are not a caller asking the robot to stop."""
    words = clip["text"].split()
    return (clip.get("role") is not None and clip["answer"] == STOP and clip.get("duration", 0.0) <= SHORT_SPEECH_S
            and bool(words) and words[0] in NO_WORDS)


def pick(rows: list[dict], rng: np.random.Generator) -> list[dict]:
    """All listening pieces, then about twice as much speech, short speech first."""
    keep = [r for r in rows if r["answer"] == KEEP]
    speech = [r for r in rows if r["answer"] == STOP]
    short = [r for r in speech if r["duration"] <= SHORT_SPEECH_S]
    long = [r for r in speech if r["duration"] > SHORT_SPEECH_S]
    want = int(len(keep) * SPEECH_PER_LISTENING)
    take_short = min(len(short), want // 2)
    take_long = min(len(long), want - take_short)
    chosen = [short[i] for i in rng.permutation(len(short))[:take_short]] + [long[i] for i in rng.permutation(len(long))[:take_long]]
    return keep + chosen


def import_files(count: int, token: str, sounds: Path = IV_SOUNDS, log: Callable[[str], None] = print) -> list[Piece]:
    """The first `count` of IndicVoices' Hindi files: phone conversations only, labelled, at
    the phone's 8 kHz, saved as wav files with one list. Downloads are kept by Hugging Face's
    own cache, so a rerun downloads nothing twice."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    (sounds / "clips").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026_11_11)
    pieces: list[Piece] = []
    for n in range(count):
        name = f"hindi/train-{n:05d}-of-00082.parquet"
        path = hf_hub_download(REPO, name, repo_type="dataset", token=token)
        table = pq.read_table(path, columns=["audio_filepath", "verbatim", "duration", "scenario", "speaker_id"])
        rows = []
        for i, row in enumerate(table.to_pylist()):
            if row["scenario"] != "Conversation":
                continue
            answer = answer_for(row["verbatim"], row["duration"])
            if answer is not None:
                rows.append({**row, "answer": answer, "index": i})
        for row in pick(rows, rng):
            samples, rate = soundfile.read(io.BytesIO(row["audio_filepath"]["bytes"]), dtype="float32", always_2d=True)
            phone = resample(samples.mean(axis=1), rate, PHONE_RATE)[: int(KEEP_UP_TO_S * PHONE_RATE)]
            piece_id = f"iv-{n:02d}-{row['index']:05d}"
            soundfile.write(sounds / "clips" / f"{piece_id}.wav", phone, PHONE_RATE, subtype="PCM_16")
            pieces.append(Piece(piece_id, row["speaker_id"], row["answer"], clean(row["verbatim"]), round(row["duration"], 3),
                                role_of(row["speaker_id"]), opens_with_listening(row["verbatim"]), f"clips/{piece_id}.wav"))
        log(f"{name}: {Counter(p.answer for p in pieces)} so far, {len({p.speaker for p in pieces})} speakers")
    (sounds / "pieces.json").write_text(json.dumps([asdict(p) for p in pieces], ensure_ascii=False, indent=1), encoding="utf-8")
    return pieces


def _item(piece_id: str, speaker: str, answer: str, text: str, duration: float, opens: bool,
          audio: np.ndarray) -> tuple[dict, Timing, np.ndarray]:
    """A piece in the same form as the machine clips: (clip, timing, 8 kHz audio). The dataset
    already cut each piece at its pauses, so its speech starts at 0 and ends at its end; a piece
    opening with a listening word teaches nothing in its first moments, as for the machine
    sentences."""
    role = role_of(speaker)
    timing = Timing(piece_id, 0.0, min(duration, len(audio) / PHONE_RATE), ignore_until=NO_GAP_OPENING_S if opens else None)
    clip = {"id": piece_id, "group": f"real {answer}", "answer": answer, "text": text, "voice": speaker,
            "held_back": role == "test", "role": role, "duration": duration}
    return clip, timing, audio


def load_pieces(sounds: Path = IV_SOUNDS) -> list[tuple[dict, Timing, np.ndarray]]:
    """The first import's pieces. Sound stays 16-bit, as saved: half the memory of float, and
    sound.on_a_rough_line turns it into exactly the float a float read gives."""
    out = []
    for piece in json.loads((sounds / "pieces.json").read_text(encoding="utf-8")):
        audio, _ = soundfile.read(sounds / piece["file"], dtype="int16")
        out.append(_item(piece["id"], piece["speaker"], piece["answer"], piece["text"], piece["duration"],
                         piece["opens_with_listening"], audio))
    return out


# --- files 10-81: cut one at a time, raw file deleted at once ------------------------------


def file_name(n: int) -> str:
    return f"hindi/train-{n:05d}-of-{HINDI_FILES:05d}.parquet"


def store_rows(rows: list[dict], rng: np.random.Generator) -> list[dict]:
    """What a newer file keeps for good: every labelled conversation piece up to 2 s, as a later
    relabel can only move those between listening and speech, so the raw file is never needed
    again; and about as many longer speech pieces as listening ones, as the first import kept."""
    short = [r for r in rows if r["duration"] <= LONGEST_LISTENING_S]
    long = [r for r in rows if r["duration"] > LONGEST_LISTENING_S]
    listening = sum(r["answer"] == KEEP for r in short)
    return short + [long[i] for i in sorted(rng.permutation(len(long))[:listening])]


def cut_file(path: Path, n: int, packs: Path) -> dict[str, int]:
    """One raw file's kept pieces, as one 8 kHz FLAC (16-bit, as the first import's wavs) and
    one list of where each piece starts. The list is written last, so a file whose list exists
    is finished."""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["audio_filepath", "verbatim", "duration", "scenario", "speaker_id"])
    rows = []
    for i, row in enumerate(table.select(["verbatim", "duration", "scenario", "speaker_id"]).to_pylist()):
        if row["scenario"] != "Conversation":
            continue
        answer = answer_for(row["verbatim"], row["duration"])
        if answer is not None:
            rows.append({**row, "answer": answer, "index": i})
    kept = store_rows(rows, np.random.default_rng(2026_11_14 + n))
    audio_column = table.column("audio_filepath")
    sounds, entries, at = [], [], 0
    for row in kept:
        samples, rate = soundfile.read(io.BytesIO(audio_column[row["index"]].as_py()["bytes"]), dtype="float32", always_2d=True)
        phone = resample(samples.mean(axis=1), rate, PHONE_RATE)[: int(KEEP_UP_TO_S * PHONE_RATE)].astype(np.float32)
        sounds.append(phone)
        entries.append({"id": f"iv-{n:02d}-{row['index']:05d}", "speaker": row["speaker_id"], "verbatim": row["verbatim"],
                        "duration": round(row["duration"], 3), "start": at, "length": len(phone)})
        at += len(phone)
    # soundfile turns float into 16-bit the same way the first import's wavs were written.
    soundfile.write(packs / f"iv-{n:02d}.flac", np.concatenate(sounds) if sounds else np.zeros(1, np.float32),
                    PHONE_RATE, subtype="PCM_16", format="FLAC")
    temporary = packs / f"iv-{n:02d}.json.part"
    temporary.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, packs / f"iv-{n:02d}.json")
    answers = Counter(r["answer"] for r in kept)
    return {"kept": len(kept), "keep": answers[KEEP], "stop": answers[STOP], "conversation": len(rows)}


def hf_download(token: str, folder: Path) -> Callable[[str], Path]:
    """Downloads one raw file straight into `folder`, outside Hugging Face's shared cache, so
    deleting it there deletes it for good."""
    from huggingface_hub import hf_hub_download

    def download(name: str) -> Path:
        return Path(hf_hub_download(REPO, name, repo_type="dataset", token=token, local_dir=folder))

    return download


def forget_download(folder: Path, name: str) -> list[Path]:
    """Deletes one raw file and the download tool's own note about it: nothing else. Returns
    what it deleted."""
    note = folder / ".cache" / "huggingface" / "download"
    gone = []
    for path in (folder / name, note / f"{name}.metadata", note / f"{name}.lock"):
        if path.is_file():
            path.unlink()
            gone.append(path)
    return gone


def import_more(first: int, last: int, download: Callable[[str], Path], forget: Callable[[str], object],
                store: Path = MORE, ahead: int = 0, log: Callable[[str], None] = print) -> list[dict]:
    """Files first..last-1, one at a time: download, cut, delete the raw file. A finished file is
    skipped, so a stopped run carries on where it stopped. With `ahead`, the next file
    downloads while this one is cut (one more raw file on disk at a time)."""
    packs = store / "packs"
    packs.mkdir(parents=True, exist_ok=True)
    todo = [n for n in range(first, last) if n not in NO_CONVERSATIONS and not (packs / f"iv-{n:02d}.json").exists()]
    log(f"{len(todo)} files to do, {last - first - len(todo)} already done")
    done: list[dict] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(1) as fetcher:
        fetching = {}
        for i, n in enumerate(todo):
            for upcoming in todo[i:i + 1 + ahead]:
                if upcoming not in fetching:
                    fetching[upcoming] = fetcher.submit(download, file_name(upcoming))
            waited = time.perf_counter()
            path = fetching.pop(n).result()
            waited = time.perf_counter() - waited
            size = path.stat().st_size
            cutting = time.perf_counter()
            stats = cut_file(path, n, packs)
            cutting = time.perf_counter() - cutting
            forget(file_name(n))
            stats.update({"file": n, "raw_mb": round(size / 1e6), "pack_mb": round((packs / f"iv-{n:02d}.flac").stat().st_size / 1e6, 1),
                          "waited_s": round(waited, 1), "cut_s": round(cutting, 1)})
            done.append(stats)
            log(f"file {n}: {stats['kept']} pieces kept ({stats['keep']} listening, {stats['stop']} speech), "
                f"pack {stats['pack_mb']} MB, raw {stats['raw_mb']} MB deleted; waited {stats['waited_s']} s for the "
                f"download, cut in {stats['cut_s']} s")
            if (i + 1) % 5 == 0 or i + 1 == len(todo):
                spent = time.perf_counter() - started
                log(f"progress: {i + 1} of {len(todo)} files, {sum(d['kept'] for d in done)} pieces kept, "
                    f"{sum(d['pack_mb'] for d in done):.0f} MB of packs, about {spent / (i + 1) * (len(todo) - i - 1) / 60:.0f} min left")
    return done


def training_share(clips: list[dict], rng: np.random.Generator, per_file: int = NEW_PER_FILE) -> list[int]:
    """Which of one newer file's pieces are used: every English listening piece and every short
    "no" (the two weak spots), then other pieces up to `per_file`, chosen as the first import
    chose: listening, and twice as much speech, short speech first."""
    weak = {i for i, c in enumerate(clips)
            if is_short_no(c) or (c["answer"] == KEEP and ENGLISH_LISTENING_WORDS & set(c["text"].split()))}
    rest = [i for i in range(len(clips)) if i not in weak]
    keep = [i for i in rest if clips[i]["answer"] == KEEP]
    short = [i for i in rest if clips[i]["answer"] == STOP and clips[i]["duration"] <= SHORT_SPEECH_S]
    long = [i for i in rest if clips[i]["answer"] == STOP and clips[i]["duration"] > SHORT_SPEECH_S]
    room = max(0, per_file - len(weak))
    take_keep = min(len(keep), room // 3)
    speech = room - take_keep
    take_short = min(len(short), speech // 2)
    take_long = min(len(long), speech - take_short)
    chosen = (list(weak) + [keep[i] for i in rng.permutation(len(keep))[:take_keep]]
              + [short[i] for i in rng.permutation(len(short))[:take_short]] + [long[i] for i in rng.permutation(len(long))[:take_long]])
    return sorted(chosen)


def load_more(store: Path = MORE, per_file: int = NEW_PER_FILE, seed: int = 2026_11_14) -> list[tuple[dict, Timing, np.ndarray]]:
    """The newer files' pieces used for training and testing, labelled now from their words, so
    a changed rule needs no new cutting. The same choice for every role, so check and test
    hold the same mix as training."""
    rng = np.random.default_rng(seed)
    out = []
    for listing in sorted((store / "packs").glob("iv-*.json")):
        clips, entries = [], []
        for entry in json.loads(listing.read_text(encoding="utf-8")):
            answer = answer_for(entry["verbatim"], entry["duration"])
            if answer is not None:
                clips.append({"answer": answer, "text": clean(entry["verbatim"]), "duration": entry["duration"],
                              "role": role_of(entry["speaker"])})
                entries.append(entry)
        chosen = training_share(clips, rng, per_file)
        if not chosen:
            continue
        sound, _ = soundfile.read(listing.with_suffix(".flac"), dtype="int16")
        for i in chosen:
            entry = entries[i]
            audio = sound[entry["start"]:entry["start"] + entry["length"]].copy()
            out.append(_item(entry["id"], entry["speaker"], clips[i]["answer"], clips[i]["text"], entry["duration"],
                             opens_with_listening(entry["verbatim"]), audio))
    return out
