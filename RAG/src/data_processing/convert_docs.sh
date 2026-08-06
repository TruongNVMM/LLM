#!/usr/bin/env bash
set -euo pipefail

# Usage: ./convert_docs.sh [input_dir] [output_dir]
# Defaults: input_dir=data_fetch, output_dir=data_fetch

INPUT_DIR=${1:-data_fetch}
OUTPUT_DIR=${2:-$INPUT_DIR}

cmd=""
if command -v soffice >/dev/null 2>&1; then
  cmd=soffice
elif command -v libreoffice >/dev/null 2>&1; then
  cmd=libreoffice
else
  echo "LibreOffice not found. Install it, e.g. 'sudo apt update && sudo apt install -y libreoffice'"
  exit 2
fi

if [ ! -d "$INPUT_DIR" ]; then
  echo "Input directory not found: $INPUT_DIR"
  exit 3
fi

mkdir -p "$OUTPUT_DIR"

find "$INPUT_DIR" -maxdepth 1 -type f -name "*.doc" -print0 | while IFS= read -r -d '' file; do
  base=$(basename "$file")
  # skip temp Word files
  if [[ "$base" == ~\$* ]]; then
    continue
  fi
  outname="${base%.*}.docx"
  outpath="$OUTPUT_DIR/$outname"
  if [ -f "$outpath" ]; then
    echo "Skipped existing: $outname"
    continue
  fi
  echo "Converting: $base -> $outname"
  "$cmd" --headless --convert-to docx --outdir "$OUTPUT_DIR" "$file"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "Conversion failed for $base (exit $rc)" >&2
  fi
done

echo "Done. Converted files are in: $OUTPUT_DIR"
