#!/usr/bin/env python3
"""
EMBRACE transcript anonymization agent.

Given a .mov video and a list of names to redact, this script:
  1. Extracts the audio track to 16 kHz mono WAV.
  2. Transcribes with local faster-whisper, producing word-level timestamps.
  3. Uses Azure OpenAI (GPT-4.1) to expand the protected-name list with
     common Whisper mis-transcriptions (e.g. "Naomi" -> "Nomi", "Naomy").
     Falls back to a regex-only match if the API is unreachable.
  4. Redacts every matching word in both segment text and word-level data.
  5. Writes:
       - out/transcription.txt              (full text, names replaced with [REDACTED])
       - out/transcription.srt              (redacted SRT subtitles)
       - work/audio.wav                     (raw extracted audio)
       - work/audio_muted.wav               (audio with name regions silenced)
       - out/EMBRACE_Child_AI_Sample_blur.anonymized.mov
            (original video + muted audio + burned-in redacted subtitles)

Usage:
    python anonymize.py --input EMBRACE_Child_AI_Sample_blur.mov \
        --names Naomi Marion Simeon

If --no-audio is passed, the output video has no audio track (fastest path
that still gives a properly subtitled redacted video, per user request).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
REDACTION_TOKEN = "[REDACTED]"
# Pad name timestamps so the audio mute fully covers the spoken name
# (Whisper word-level timestamps can be off by ~100-200 ms).
MUTE_PAD_BEFORE_S = 0.15
MUTE_PAD_AFTER_S = 0.20
# SRT lines wrap at this many chars per line.
SRT_LINE_WIDTH = 42
# Local Whisper model. "medium" provides much better proper-noun accuracy than
# "small", which is critical when redacting specific names. Override via env.
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "medium")
# Discard expanded name variants shorter than this many characters to avoid
# false-positive matches on common short tokens. 3 is permissive enough to
# catch real-world short names like "Leo" / "Sam" / "Max" while still excluding
# very-short fragments. The default applies only to AI-expanded variants;
# user-given names are always honored regardless of length.
MIN_VARIANT_LEN = 3

# -----------------------------------------------------------------------------
# Data types
# -----------------------------------------------------------------------------
@dataclass
class Word:
    start: float
    end: float
    word: str            # original token as transcribed (may include leading space)
    redacted: bool = False

@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word]
    redacted_text: str = ""

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def need_tool(name: str) -> None:
    if shutil.which(name) is None:
        sys.exit(f"ERROR: required tool '{name}' not found on PATH.")


def find_ffmpeg(prefer_libass: bool = False) -> str:
    """Return path to a usable ffmpeg, preferring one with libass when needed."""
    # Bundled static ffmpeg first (we know it has libass). Search the script's
    # own dir AND a few likely parents so the script works whether it lives at
    # the project root or in a subdirectory.
    here = Path(__file__).resolve().parent
    search_dirs = [here, here.parent, here.parent.parent, Path.cwd()]
    candidates: list[str] = []
    for d in search_dirs:
        static = d / "tools" / "ffmpeg"
        if static.exists() and str(static) not in candidates:
            candidates.append(str(static))
    if shutil.which("ffmpeg"):
        candidates.append(shutil.which("ffmpeg"))  # type: ignore[arg-type]
    if not candidates:
        sys.exit("ERROR: ffmpeg not found on PATH and tools/ffmpeg missing.")
    if not prefer_libass:
        return candidates[0]
    for c in candidates:
        try:
            res = subprocess.run([c, "-hide_banner", "-filters"],
                                 capture_output=True, text=True, check=False)
            if " ass " in res.stdout or " subtitles " in res.stdout:
                return c
        except Exception:
            continue
    return candidates[0]


def extract_audio(input_video: Path, out_wav: Path, ffmpeg_bin: str) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    run([
        ffmpeg_bin, "-y", "-i", str(input_video),
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(out_wav),
    ])


def transcribe(wav_path: Path, hint_names: list[str] | None = None) -> list[Segment]:
    from faster_whisper import WhisperModel

    print(f"[transcribe] loading faster-whisper model: {WHISPER_MODEL}")
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    # initial_prompt biases the decoder toward the spelling of unusual proper
    # nouns so the children's names come out correctly. We deliberately include
    # the names twice to reinforce them.
    prompt_pieces: list[str] = []
    if hint_names:
        names_csv = ", ".join(hint_names)
        prompt_pieces.append(
            f"Conversation between a robot called Reachy and children named "
            f"{names_csv}. Names mentioned include {names_csv}."
        )
    initial_prompt = " ".join(prompt_pieces) if prompt_pieces else None

    print(f"[transcribe] running on {wav_path}")
    if initial_prompt:
        print(f"[transcribe] initial_prompt = {initial_prompt!r}")
    seg_iter, info = model.transcribe(
        str(wav_path),
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
        language=None,  # auto-detect
        condition_on_previous_text=True,
        initial_prompt=initial_prompt,
    )
    print(f"[transcribe] detected language: {info.language} (p={info.language_probability:.2f})")

    segments: list[Segment] = []
    for s in seg_iter:
        words: list[Word] = []
        if s.words:
            for w in s.words:
                # faster-whisper sometimes emits a leading space we want to keep
                words.append(Word(start=float(w.start), end=float(w.end), word=w.word))
        segments.append(Segment(
            start=float(s.start),
            end=float(s.end),
            text=s.text.strip(),
            words=words,
        ))
        print(f"  [{s.start:7.2f} - {s.end:7.2f}] {s.text.strip()}")
    return segments


# -----------------------------------------------------------------------------
# Name expansion via Azure OpenAI (with fallback)
# -----------------------------------------------------------------------------
def expand_names_with_azure(protected_names: list[str]) -> list[str]:
    """Ask gpt-4.1 for likely Whisper mis-transcription variants for each name.

    Returns a deduplicated list including the originals. On any failure, returns
    just the originals.
    """
    base = {n.strip() for n in protected_names if n.strip()}
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
    deployment = "gpt-4.1"  # known good deployment on this resource

    if not (endpoint and key):
        print("[name-expand] Azure env vars missing; using raw names only.")
        return sorted(base)

    import requests

    sys_prompt = (
        "You assist with anonymizing audio transcripts. Given a list of personal "
        "first names that must be redacted, return a JSON object with a single "
        "key 'variants' whose value is an array of strings that an automatic "
        "speech recognizer (Whisper) might output for any of these names. "
        "Include the originals, common phonetic mis-spellings, and obvious "
        "diminutives. Use only English alphabet, no surrounding punctuation."
    )
    user_prompt = f"Names: {sorted(base)}\nReturn ONLY the JSON object."

    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    payload = {
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 400,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    try:
        r = requests.post(url, headers={"api-key": key, "Content-Type": "application/json"},
                          json=payload, timeout=30)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        # Strip common markdown code fences if present.
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        try:
            data = json.loads(content)
            variants = data.get("variants", [])
        except json.JSONDecodeError:
            # Fall back to grabbing every quoted alphabetic token from the body.
            variants = re.findall(r'"([A-Za-z][A-Za-z\'-]{1,20})"', content)
        for v in variants:
            if isinstance(v, str) and v.strip():
                base.add(v.strip())
        print(f"[name-expand] expanded {len(protected_names)} names -> {len(base)} variants via Azure")
    except Exception as e:
        print(f"[name-expand] Azure call failed ({e}); falling back to raw names.")
    return sorted(base)


# -----------------------------------------------------------------------------
# Redaction
# -----------------------------------------------------------------------------
def normalize_token(tok: str) -> str:
    """Lower-case and strip ALL non-letter characters for matching.

    Dropping apostrophes means "Devin's" normalizes to "devins", which then
    matches the possessive variant we add in `build_name_set`.
    """
    return re.sub(r"[^a-z]", "", tok.lower())


def build_name_set(name_variants: Iterable[str], min_len: int = MIN_VARIANT_LEN,
                   *, explicit: Iterable[str] = ()) -> set[str]:
    """Return a set of lowercase keys to match against normalized tokens.

    Args:
        name_variants: AI-expanded / general variants. Items shorter than
            `min_len` characters (after normalization) are dropped to avoid
            false-positive matches.
        explicit: User-given names that must ALWAYS be included regardless
            of their length (so short legitimate names like "Leo" or "Sam"
            are honored even when the variant filter is conservative).

    For each kept name we add both the bare form and the possessive form
    (Name + "s"), so we catch "Devin", "Devins" (plural-style), and
    "Devin's" -> "devins".
    """
    keys: set[str] = set()
    for n in explicit:
        norm = normalize_token(n)
        if norm:
            keys.add(norm)
            keys.add(norm + "s")
    for n in name_variants:
        norm = normalize_token(n)
        if norm and len(norm) >= min_len:
            keys.add(norm)
            keys.add(norm + "s")
    return keys


def redact_segments(segments: list[Segment], name_keys: set[str]) -> tuple[list[Segment], list[tuple[float, float]]]:
    """Mark matching words as redacted. Return updated segments + list of (start,end) mute intervals."""
    intervals: list[tuple[float, float]] = []
    for seg in segments:
        redacted_words: list[str] = []
        for w in seg.words:
            key = normalize_token(w.word)
            if key and key in name_keys:
                w.redacted = True
                intervals.append((max(0.0, w.start - MUTE_PAD_BEFORE_S),
                                  w.end + MUTE_PAD_AFTER_S))
                # Preserve any leading space in the original token.
                leading_space = " " if w.word.startswith(" ") else ""
                redacted_words.append(f"{leading_space}{REDACTION_TOKEN}")
            else:
                redacted_words.append(w.word)
        if seg.words:
            seg.redacted_text = "".join(redacted_words).strip()
        else:
            # No word-level data -> fall back to regex on segment text
            seg.redacted_text = redact_text_with_regex(seg.text, name_keys)
        # Also rebuild seg.text as a fallback original text we keep around
        if seg.words and not seg.text:
            seg.text = "".join(w.word for w in seg.words).strip()
    return segments, merge_intervals(intervals)


def redact_text_with_regex(text: str, name_keys: set[str]) -> str:
    """Word-boundary regex fallback (case-insensitive).

    Also matches possessive forms (Name's) by allowing an optional 's tail
    that is consumed along with the name.
    """
    if not name_keys:
        return text
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(k) for k in name_keys) + r")(?:'s)?\b",
        flags=re.IGNORECASE,
    )
    return pattern.sub(REDACTION_TOKEN, text)


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1] + 0.05:  # merge if nearly touching
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


# -----------------------------------------------------------------------------
# SRT writing
# -----------------------------------------------------------------------------
def fmt_srt_time(t: float) -> str:
    if t < 0:
        t = 0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms == 1000:
        ms = 0
        s += 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def wrap(text: str, width: int = SRT_LINE_WIDTH) -> str:
    """Simple greedy wrap for subtitle lines (max 2 lines, soft limit)."""
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= width:
            cur = f"{cur} {w}"
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    # Cap at 2 lines for readability; if longer, just join.
    if len(lines) > 2:
        return "\n".join([" ".join(lines[: len(lines) // 2]), " ".join(lines[len(lines) // 2:])])
    return "\n".join(lines)


def write_srt(segments: list[Segment], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, start=1):
            text = seg.redacted_text if seg.redacted_text else seg.text
            if not text:
                continue
            f.write(f"{idx}\n")
            f.write(f"{fmt_srt_time(seg.start)} --> {fmt_srt_time(seg.end)}\n")
            f.write(f"{wrap(text)}\n\n")


def fmt_ass_time(t: float) -> str:
    """ASS time format: H:MM:SS.cs (centiseconds)."""
    if t < 0:
        t = 0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int(round((t - int(t)) * 100))
    if cs == 100:
        cs = 0
        s += 1
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def write_ass(segments: list[Segment], out_path: Path) -> None:
    """Write an ASS subtitle file with styling embedded.

    Using ASS instead of SRT + force_style avoids the comma-escaping headache
    inside the ffmpeg subtitles filter, and gives identical visual results.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1088
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Helvetica,44,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,3,2,0,2,60,60,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    with out_path.open("w", encoding="utf-8") as f:
        f.write(header)
        for seg in segments:
            text = seg.redacted_text if seg.redacted_text else seg.text
            if not text:
                continue
            wrapped = wrap(text).replace("\n", "\\N")
            f.write(
                f"Dialogue: 0,{fmt_ass_time(seg.start)},{fmt_ass_time(seg.end)},"
                f"Default,,0,0,0,,{wrapped}\n"
            )


def write_transcript_txt(segments: list[Segment], out_path: Path) -> None:
    """Plain-text transcript with timestamps, names redacted."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# EMBRACE redacted transcript\n")
        f.write("# Names matching the protected list have been replaced with [REDACTED].\n\n")
        for seg in segments:
            text = seg.redacted_text if seg.redacted_text else seg.text
            f.write(f"[{fmt_srt_time(seg.start)} --> {fmt_srt_time(seg.end)}] {text}\n")


# -----------------------------------------------------------------------------
# Audio muting + final mux
# -----------------------------------------------------------------------------
def build_mute_filter(intervals: list[tuple[float, float]]) -> str:
    """Build an ffmpeg -af 'volume' expression that silences each interval."""
    if not intervals:
        return ""
    conds = "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in intervals)
    # ffmpeg evaluates the enable expression; nonzero -> apply volume=0.
    return f"volume=enable='{conds}':volume=0:eval=frame"


def mute_audio(in_wav: Path, intervals: list[tuple[float, float]], out_wav: Path,
               ffmpeg_bin: str) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if not intervals:
        shutil.copyfile(in_wav, out_wav)
        return
    af = build_mute_filter(intervals)
    run([
        ffmpeg_bin, "-y", "-i", str(in_wav),
        "-af", af,
        "-c:a", "pcm_s16le",
        str(out_wav),
    ])


def burn_and_mute(input_video: Path, ass_path: Path,
                  mute_intervals: list[tuple[float, float]],
                  out_video: Path, ffmpeg_bin: str,
                  drop_audio: bool = False,
                  preset: str = "medium",
                  crf: int = 20,
                  threads: int = 0,
                  copy_video: bool = False) -> None:
    """Burn subtitles into the video AND mute name regions in the original
    audio in a single ffmpeg invocation, preserving the source audio fidelity.

    When `copy_video` is True, the original video stream is copied as-is
    (no re-encode, no burned subtitles). This is a much faster path that
    still mutes the named regions in the high-fidelity source audio.
    """
    out_video.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [ffmpeg_bin, "-y", "-i", str(input_video)]

    if not copy_video:
        cwd = Path.cwd().resolve()
        try:
            rel = ass_path.resolve().relative_to(cwd)
            sub_arg = rel.as_posix()
        except ValueError:
            sub_arg = ass_path.resolve().as_posix()
        cmd += ["-vf", f"ass={sub_arg}"]

    if drop_audio:
        cmd += ["-an"]
    else:
        af = build_mute_filter(mute_intervals)
        if af:
            cmd += ["-af", af]
        cmd += ["-c:a", "aac", "-b:a", "192k"]

    if copy_video:
        cmd += ["-c:v", "copy"]
    else:
        cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf)]
        if threads and threads > 0:
            cmd += ["-threads", str(threads)]
    cmd += [str(out_video)]
    run(cmd)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", "-i", default="EMBRACE_Child_AI_Sample_blur.mov",
                   help="Input .mov path")
    p.add_argument("--names", "-n", nargs="+", default=["Naomi", "Marion", "Simeon"],
                   help="Names to redact (space separated)")
    p.add_argument("--work-dir", default="work", help="Scratch directory")
    p.add_argument("--out-dir", default="out", help="Output directory")
    p.add_argument("--no-audio", action="store_true",
                   help="Drop the audio track entirely (fastest path; still burns subtitles)")
    p.add_argument("--skip-transcribe", action="store_true",
                   help="Re-use a cached work/segments.json if present (dev convenience)")
    p.add_argument("--no-burn", action="store_true",
                   help="Stop after writing transcription.txt/.srt/.ass; skip the final video re-encode")
    p.add_argument("--preset", default="medium",
                   help="x264 preset for the final encode (ultrafast|superfast|veryfast|faster|fast|medium|slow). Default: medium")
    p.add_argument("--crf", type=int, default=20, help="x264 CRF for the final encode (default 20)")
    p.add_argument("--threads", type=int, default=0,
                   help="Limit ffmpeg/libx264 threads (0 = auto). Useful for running two burns in parallel.")
    p.add_argument("--copy-video", action="store_true",
                   help="Skip subtitle burn and copy the original video stream (no re-encode). "
                        "Still mutes named regions in the source audio. Much faster; "
                        "subtitles are still written separately as .srt/.ass.")
    args = p.parse_args()

    load_dotenv()
    need_tool("ffprobe")

    # Expand multi-word names (e.g. "Jenna Eastman") into individual tokens so
    # both the first name and the last name get matched independently. We keep
    # the order and de-duplicate. This lets the user pass full names without
    # having to split them by hand.
    expanded_names: list[str] = []
    seen: set[str] = set()
    for raw in args.names:
        for tok in str(raw).split():
            key = tok.lower()
            if key and key not in seen:
                seen.add(key)
                expanded_names.append(tok)
    if expanded_names != list(args.names):
        print(f"[names] expanded {args.names} -> {expanded_names}")
    args.names = expanded_names
    # Prefer an ffmpeg with libass for the final subtitle burn; fall back to
    # the system ffmpeg for the simpler extract/mute stages.
    ffmpeg_extract = find_ffmpeg(prefer_libass=False)
    ffmpeg_burn = find_ffmpeg(prefer_libass=True)
    print(f"[ffmpeg] extract: {ffmpeg_extract}")
    print(f"[ffmpeg] burn   : {ffmpeg_burn}")

    in_video = Path(args.input).resolve()
    if not in_video.exists():
        sys.exit(f"ERROR: input video not found: {in_video}")

    work = Path(args.work_dir).resolve()
    out = Path(args.out_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    audio_wav = work / "audio.wav"
    seg_json = work / "segments.json"
    srt_path = out / "transcription.srt"
    ass_path = work / "subtitles.ass"   # styled file used for burn-in
    txt_path = out / "transcription.txt"
    muted_wav = work / "audio_muted.wav"
    out_video = out / f"{in_video.stem}.anonymized.mov"

    # ---- Stage 1: extract audio ----
    if not audio_wav.exists():
        print("\n=== Stage 1: extract audio ===")
        extract_audio(in_video, audio_wav, ffmpeg_extract)
    else:
        print(f"[stage1] reusing {audio_wav}")

    # ---- Stage 2: transcribe ----
    print("\n=== Stage 2: transcribe ===")
    if args.skip_transcribe and seg_json.exists():
        print(f"[stage2] loading cached {seg_json}")
        raw = json.loads(seg_json.read_text())
        segments = [Segment(start=s["start"], end=s["end"], text=s["text"],
                            words=[Word(**w) for w in s["words"]])
                    for s in raw]
    else:
        segments = transcribe(audio_wav, hint_names=args.names)
        seg_json.write_text(json.dumps([asdict(s) for s in segments], indent=2))

    # ---- Stage 3: expand & redact names ----
    print("\n=== Stage 3: name expansion + redaction ===")
    variants = expand_names_with_azure(args.names)
    print(f"[redact] effective name variants: {variants}")
    name_keys = build_name_set(variants, explicit=args.names)
    segments, mute_intervals = redact_segments(segments, name_keys)

    redacted_count = sum(1 for s in segments for w in s.words if w.redacted)
    print(f"[redact] {redacted_count} word(s) flagged; {len(mute_intervals)} mute interval(s)")
    for s, e in mute_intervals:
        print(f"   mute [{s:7.2f} -> {e:7.2f}]  ({e - s:.2f}s)")

    # ---- Stage 4: write text + SRT + ASS ----
    print("\n=== Stage 4: write transcription.txt, .srt, and .ass ===")
    write_transcript_txt(segments, txt_path)
    write_srt(segments, srt_path)
    write_ass(segments, ass_path)
    print(f"  wrote {txt_path}")
    print(f"  wrote {srt_path}")
    print(f"  wrote {ass_path}")

    # ---- Stage 5: mute audio ----
    drop_audio = args.no_audio
    if drop_audio:
        print("\n=== Stage 5: skipping audio mute (--no-audio) ===")
    else:
        # Also write a stand-alone muted WAV for diagnostic/debug use.
        print("\n=== Stage 5: mute name regions in standalone audio (diagnostic) ===")
        mute_audio(audio_wav, mute_intervals, muted_wav, ffmpeg_extract)
        print(f"  wrote {muted_wav}")

    if args.no_burn:
        print("\n=== Stage 6 skipped (--no-burn) ===")
        print(f"Transcription artifacts ready: {txt_path}, {srt_path}")
        return 0

    # ---- Stage 6: burn subtitles + mute original audio in one pass ----
    if args.copy_video:
        print("\n=== Stage 6: copy video stream + mute original-fidelity audio (fast) ===")
    else:
        print("\n=== Stage 6: burn subtitles + mute original-fidelity audio ===")
    burn_and_mute(in_video, ass_path,
                  mute_intervals=mute_intervals,
                  out_video=out_video,
                  ffmpeg_bin=ffmpeg_burn,
                  drop_audio=drop_audio,
                  preset=args.preset,
                  crf=args.crf,
                  threads=args.threads,
                  copy_video=args.copy_video)
    print(f"\nDONE. Final anonymized video: {out_video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
