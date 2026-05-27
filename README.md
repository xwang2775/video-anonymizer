# EMBRACE Anonymizer

A small pipeline that takes a video of children interacting with the Reachy robot and produces an anonymized copy with all personal names removed from the audio, the transcript, and the burned-in subtitles.

## What it does

Given a video file and a list of names to protect, the pipeline:

1. Extracts the audio track (16 kHz mono WAV).
2. Transcribes locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (word-level timestamps).
3. Asks Azure OpenAI (GPT-4.1) to expand the protected-name list with likely Whisper mis-spellings (e.g. `<name1>` → `<name2>`, `<name3>`). Falls back to the raw list if the API is unreachable.
4. Mutes the audio at every spoken-name region and replaces matching tokens with `[REDACTED]` in the transcript.
5. Burns the redacted subtitles into the video and remuxes the muted audio back in.

Outputs:

- `out/transcription.txt` — plain transcript with names replaced.
- `out/transcription.srt` — redacted subtitles.
- `out/<stem>.anonymized.mov` — final video with subtitles burned in and names muted.

## Setup

Requires Python 3.10+ and `ffmpeg` / `ffprobe` on `PATH` (a bundled static `ffmpeg` is also detected from `Anonymizing_data/tools/ffmpeg` if present).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env
```

`ffmpeg` install on macOS: `brew install ffmpeg`. On Ubuntu: `sudo apt install ffmpeg`.

## Usage

```bash
cd Anonymizing_data
python anonymize.py \
    --input path/to/video.mov \
    --names <name1> <name2> <name3>
```

Common flags:

| Flag | Meaning |
|---|---|
| `--names A B C` | Names to redact (space-separated; full names like `"Jane Doe"` split into tokens). |
| `--no-audio` | Drop the audio track entirely (fastest; subtitles still burned). |
| `--skip-transcribe` | Reuse a cached `work/segments.json` (dev convenience). |
| `--no-burn` | Stop after writing the transcript/SRT; skip the final video re-encode. |
| `--copy-video` | Copy the original video stream (no re-encode); still mutes named audio regions. Much faster. |
| `--preset / --crf` | x264 encode quality knobs (defaults: `medium` / `20`). |

## Helping with data processing

If you're collaborating to process more videos:

1. **Never commit raw or output data.** `.gitignore` excludes all `*.mov`, `*.m4v`, `*.wav`, `work*/`, `out*/`, and `*.log`. Keep it that way — transcripts and muted audio can still contain identifying content even after redaction.
2. **Never commit `.env`.** Use `.env.example` as the template; share Azure keys out-of-band.
3. **Work on your own copy of the video.** Put it anywhere outside the repo (or inside, ignored). Pass its path with `--input`.
4. **Review the output before sharing it.** Whisper sometimes mis-hears a name into a spelling that isn't in the variant list; if you spot one in `out/transcription.txt`, re-run with the additional spelling appended to `--names`.

A refine mode that streamlines this review-and-rerun loop (and lets you specify exact time intervals to silence) is designed in [docs/superpowers/specs/2026-05-26-refine-redaction-design.md](docs/superpowers/specs/2026-05-26-refine-redaction-design.md) — not yet implemented.

## Layout

```
EMBRACE/
├── Anonymizing_data/
│   ├── anonymize.py            # the pipeline
│   ├── agent_permissions.json  # self-declared permissions for the agent
│   └── tools/                  # optional: drop a static `ffmpeg` here
├── docs/superpowers/specs/     # design docs
├── .env.example                # template for your local .env
├── requirements.txt
└── README.md
```
