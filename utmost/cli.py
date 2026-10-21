"""Command line interface: `utmost <audio files or directories> [--emotion Happy]`."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .api import DEFAULT_REPO, load
from .model import EMOTIONS

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3"}


def collect(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in map(Path, paths):
        if item.is_dir():
            files.extend(sorted(p for p in item.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES))
        else:
            files.append(item)
    return files


def read_emotion_table(path: Path) -> dict[str, str]:
    """Rows of `<file name>|<emotion>` or `<file name>,<emotion>`; extra columns are ignored."""
    table = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        sep = "|" if "|" in line else ","
        name, emotion = [x.strip() for x in line.split(sep)[:2]]
        table[name] = emotion
    return table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="utmost", description=__doc__)
    parser.add_argument("inputs", nargs="+", help="audio files or directories")
    parser.add_argument("--emotion", choices=[e.lower() for e in EMOTIONS] + list(EMOTIONS), help="requested emotion for every input")
    parser.add_argument("--emotion-table", type=Path, help="per-file requested emotion: `name|emotion` per line")
    parser.add_argument("--output", "-o", type=Path, help="CSV output path (default: stdout)")
    parser.add_argument("--model", default=DEFAULT_REPO, help="Hugging Face repo id or local directory")
    parser.add_argument("--device", default=None)
    parser.add_argument("--fp32", action="store_true", help="download the float32 weights and run in float32")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)

    files = collect(args.inputs)
    if not files:
        parser.error("no audio files found")
    if args.emotion_table is not None:
        table = read_emotion_table(args.emotion_table)
        missing = [f.name for f in files if f.name not in table]
        if missing:
            parser.error(f"{len(missing)} files have no emotion in {args.emotion_table}, e.g. {missing[0]}")
        emotions = [table[f.name] for f in files]
    else:
        emotions = [args.emotion] * len(files)

    model = load(args.model, device=args.device, fp16=not args.fp32)
    print(f"utmost: {len(files)} file(s) on {model.device} ({str(model.dtype).replace('torch.', '')})", file=sys.stderr)
    results = model.predict(files, emotion=emotions, batch_size=args.batch_size)

    header = ["file", "emotion", "emotion_source", "qMOS", "eMOS", "valence", "arousal", "dominance"]
    header += [f"P({e})" for e in EMOTIONS]
    handle = args.output.open("w", newline="") if args.output else sys.stdout
    writer = csv.writer(handle)
    writer.writerow(header)
    for path, r in zip(files, results):
        row = [str(path), r["emotion"], r["emotion_source"], f"{r['qmos']:.4f}", f"{r['emos']:.4f}"]
        row += [f"{r[k]:.4f}" for k in ("valence", "arousal", "dominance")]
        row += [f"{r['category'][e]:.4f}" for e in EMOTIONS]
        writer.writerow(row)
    if args.output:
        handle.close()


if __name__ == "__main__":
    main()
