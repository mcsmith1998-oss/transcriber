#!/usr/bin/env python3
"""
transcribe.py — command-line interface for the Interview Transcriber.

Usage:
    python transcribe.py <audio_file>
    python transcribe.py <audio_file> --speakers 3
    python transcribe.py *.m4a --model large-v3 --output-dir ./transcripts

For a graphical interface, run:
    python app.py
"""

import argparse
import sys
from pathlib import Path

import core


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe interview audio with speaker labels."
    )
    parser.add_argument("files", nargs="+", type=Path, help="Audio file(s) to transcribe")
    parser.add_argument(
        "--speakers", type=int, default=0, metavar="N",
        help="Number of speakers (0 = auto-detect)",
    )
    parser.add_argument(
        "--model", default="medium",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size (default: medium)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to save transcripts (default: same folder as input)",
    )
    args = parser.parse_args()

    for p in args.files:
        if not p.exists():
            print(f"ERROR: File not found: {p}", file=sys.stderr)
            sys.exit(1)

    for audio_path in args.files:
        print(f"\nProcessing: {audio_path.name}")

        def on_progress(msg):
            print(f"  {msg}")

        try:
            transcript = core.process(
                audio_path=str(audio_path),
                model_size=args.model,
                num_speakers=args.speakers,
                on_progress=on_progress,
            )
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            continue

        output_dir = args.output_dir or audio_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / (audio_path.stem + "_transcript.txt")
        output_path.write_text(transcript, encoding="utf-8")
        print(f"  Saved: {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
