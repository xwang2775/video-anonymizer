# Refine-Redaction Mode for `anonymize.py`

**Date:** 2026-05-26
**Status:** Approved by user, pending implementation plan

## Problem

`anonymize.py` redacts names from a video in a single pass: extract audio → transcribe with Whisper → expand names via GPT-4.1 → mute audio + burn redacted subtitles. Whisper and the name-variant expander are not perfect: a reviewer watching the output will sometimes notice content that should have been redacted but wasn't (a missed name variant, a place name, a phone number, a short segment Whisper missed entirely). Today the only recovery is to re-run the whole pipeline from scratch with more names, which is slow and loses the auditable history of what was redacted on each pass.

## Goal

Add a refine mode that lets a reviewer feed back missed content after the initial run and regenerate the video, while:

- Reusing cached transcription (the slow step) by default.
- Remembering everything previously asked to redact, so the reviewer only specifies the new additions.
- Producing a new versioned output file (`*.anonymized.v2.mov`, `v3.mov`, ...) without overwriting prior versions.
- Supporting three orthogonal input methods that cover different kinds of misses.

## Non-Goals

- Selectively redacting *one occurrence* of a word while leaving other occurrences untouched. (Use `--add-intervals` for surgical single-occurrence muting.)
- An interactive UI. Refine mode is CLI only.
- Undoing previously requested redactions. State is append-only; if the reviewer wants to undo, they edit `work/redaction_state.json` by hand.

## Inputs

Three input methods, all combinable in a single refine invocation:

### 1. `--add-names <name>...`
Additional names/words to redact. Behaviorally identical to extending the initial `--names` list:

- Each gets passed through the existing Azure GPT-4.1 variant expansion (then fallback to raw).
- Every occurrence in the transcript is replaced with `[REDACTED]` (text + audio + burned subtitle).
- Multi-token entries (`"Madison Elementary"`) are split into individual tokens, same as today's `--names` behavior.

### 2. `--add-intervals <range>[,<range>...]`
Explicit time ranges to silence even if Whisper produced no recognizable word there. Formats accepted:

- `MM:SS-MM:SS` (e.g. `1:23-1:27`)
- `MM:SS.s-MM:SS.s` (e.g. `2:34.5-2:37.2`)
- `HH:MM:SS-HH:MM:SS` and `HH:MM:SS.ms-HH:MM:SS.ms`

Behavior:
- The range is added to the mute-interval list as-is (no `MUTE_PAD_*` padding — the reviewer specified exact bounds).
- Any transcript word whose `[start, end]` overlaps the range is marked redacted (so the subtitle in that range also shows `[REDACTED]`).

### 3. `--review-file <path>`
Path to a manually annotated copy of `out/transcription.txt`. Reviewer wraps words to redact in `<<...>>`:

```
[00:00:05,000 --> 00:00:08,000] <<Wisconsin>> is great
[00:00:08,000 --> 00:00:12,000] My phone is <<608-555-1234>>
```

Behavior:
- Every token inside `<< >>` is collected and merged into `--add-names`. Multi-token markers (`<<Madison Elementary>>`) split into individual tokens.
- Tokens that contain non-letter characters (`608-555-1234`) get normalized the same way as names; if normalization yields an empty string the token is reported and skipped (reviewer should use `--add-intervals` for those).
- The file is read-only; we do not write back to it.

## Workflow

### Initial run (unchanged behavior, plus state write)

```bash
python anonymize.py -i video.mov --names Naomi Marion Simeon
```

In addition to today's outputs, write `work/redaction_state.json`:

```json
{
  "schema_version": 1,
  "input_video": "/abs/path/video.mov",
  "original_names": ["Naomi", "Marion", "Simeon"],
  "additional_names": [],
  "manual_intervals": [],
  "review_markers": [],
  "current_version": 1
}
```

### Refine run

```bash
python anonymize.py --refine -i video.mov \
    --add-names Wisconsin Madison \
    --add-intervals 1:23-1:27,2:34.5-2:37 \
    --review-file out/transcription.txt
```

Steps:

1. Require `work/redaction_state.json` to exist; error with a clear message if not ("run without --refine first").
2. Load state; `--names` is ignored in refine mode (warn if passed). `-i / --input` must match `state.input_video` (warn-and-continue if it doesn't, to allow renamed paths).
3. Default: load cached `work/segments.json`. With `--force-transcribe`: re-extract audio and re-transcribe before continuing.
4. Apply `--add-names`: append to `state.additional_names` (dedupe).
5. Apply `--add-intervals`: parse each range; append parsed `(start, end)` tuples to `state.manual_intervals` (dedupe).
6. Apply `--review-file`: parse `<<token>>` matches; append unique tokens to `state.review_markers`; tokens that survive normalization also flow into the name set.
7. Build effective name list = `original_names ∪ additional_names ∪ review_markers`. Run name expansion + redaction over the segments (same code path as initial run).
8. Build effective mute intervals = (intervals from redacted words, with existing `MUTE_PAD_*`) ∪ (manual intervals, no padding). Merge overlaps.
9. For each manual interval, also mark overlapping words as redacted so subtitles in that range show `[REDACTED]`.
10. Bump `state.current_version` to `N+1`. Write outputs:
    - `out/<stem>.anonymized.v{N+1}.mov` (final video)
    - `out/transcription.v{N+1}.txt`, `out/transcription.v{N+1}.srt` (review artifacts for next pass)
    - Overwrite `work/audio_muted.wav`, `work/subtitles.ass`, `work/segments.json` is unchanged (transcription cache)
11. Persist updated state.

### Subsequent refines

Each `--refine` call accumulates onto the state, so version 3 includes everything from versions 1+2 plus the new additions. The reviewer only types the new misses, not the full history.

## Output Versioning

| Run | Output video | Transcript txt/srt |
|-----|---|---|
| Initial | `out/<stem>.anonymized.mov` | `out/transcription.txt`, `.srt` |
| Refine 1 | `out/<stem>.anonymized.v2.mov` | `out/transcription.v2.txt`, `.v2.srt` |
| Refine 2 | `out/<stem>.anonymized.v3.mov` | `out/transcription.v3.txt`, `.v3.srt` |

The unsuffixed `transcription.txt` from the initial run stays put — that's what the reviewer typically annotates first. Subsequent versioned transcripts let the reviewer annotate again on the latest output if more misses surface.

## State File Schema

`work/redaction_state.json`:

```json
{
  "schema_version": 1,
  "input_video": "/abs/path/video.mov",
  "original_names": ["Naomi", "Marion", "Simeon"],
  "additional_names": ["Wisconsin", "Madison"],
  "manual_intervals": [[83.0, 87.0], [154.5, 157.2]],
  "review_markers": ["Wisconsin", "Madison"],
  "current_version": 3
}
```

Notes:
- `additional_names` and `review_markers` may overlap; both feed the final name set; tracking them separately preserves an audit trail of *how* a name was added.
- `current_version` is the last successfully written version. Refine writes `v{current_version + 1}`.
- Stored in JSON for human-readability and easy hand-editing if a reviewer wants to undo a misclick.

## Error Handling

| Situation | Behavior |
|---|---|
| `--refine` with no state file | Exit 1 with message: "No prior run found at work/redaction_state.json. Run anonymize.py without --refine first." |
| `--refine` with no `--add-*` flags and no `--review-file` | Exit 1 with message: "Refine mode needs at least one of --add-names / --add-intervals / --review-file." |
| `--review-file` path missing | Exit 1, clear message. |
| `<<...>>` marker with empty interior | Warn and skip. |
| Interval format unparseable | Exit 1 with the offending string and the accepted formats. |
| Interval `end <= start` | Exit 1, "invalid interval: end must be after start". |
| `--input` doesn't match `state.input_video` | Warn ("state was created from X, but --input is Y; continuing with Y") and continue. |
| Cached segments missing but `--refine` set | Exit 1 with message: "work/segments.json not found; re-run with --force-transcribe or do a full initial run." |

## Components Touched

This stays as a single-file refactor of `anonymize.py`. New internal pieces:

- **State module** (`load_state`, `save_state`, `init_state`): JSON I/O + schema check.
- **Interval parser** (`parse_intervals`): accepts the time formats listed above; produces sorted, validated `(start, end)` tuples.
- **Review-file parser** (`parse_review_markers`): scans `<<...>>` and returns the list of inner tokens.
- **Manual-interval redaction** (`apply_manual_intervals`): given segments + manual intervals, marks overlapping words as redacted and rebuilds `redacted_text` for affected segments.
- **Output path versioning** (`versioned_paths`): given `out_dir`, `stem`, `n`, returns the v{n} output paths.
- **`--refine` entry point in `main`**: orchestrates the above; reuses existing `redact_segments`, `expand_names_with_azure`, `write_*`, `mute_audio`, `burn_and_mute`.

## Testing

- Initial run still produces identical artifacts to today (plus the new state file).
- Refine with each input type in isolation produces the expected additional redactions.
- Refine combining all three types accumulates without duplicating intervals.
- Refine without prior state errors cleanly.
- Interval parser accepts all four time formats and rejects garbage.
- `<<...>>` parser handles markers split across multiple subtitle lines, empty markers, and adjacent markers.
- Re-running the same refine with identical inputs still produces a new version (always bump on a successful refine, even if inputs duplicate existing state); this keeps the version-stamped artifacts as a faithful run log.

## Open Decisions (deferred to implementation plan)

- Exact CLI help text wording.
- Whether to keep a `work/refine.log` audit trail (for now, stdout is enough).
- Whether to expose `--no-burn` and `--copy-video` in refine mode (assumption: yes, they pass through unchanged).
