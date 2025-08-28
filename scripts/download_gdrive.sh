#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/download_gdrive.sh OUTPUT_PATH
#
# Downloads the video from the given Google Drive file ID to OUTPUT_PATH.
# Requires: python3 + pip, installs gdown if missing.

FILE_ID="1spl5lsRrz4hIo-UVr10lgesum6-kUHir"

if [ $# -lt 1 ]; then
  echo "Usage: bash $0 OUTPUT_PATH" >&2
  exit 1
fi

OUT="$1"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found" >&2
  exit 1
fi

if ! python3 -m pip show gdown >/dev/null 2>&1; then
  python3 -m pip install --user gdown >/dev/null
fi

python3 - <<'PY'
import sys, subprocess
file_id = "1spl5lsRrz4hIo-UVr10lgesum6-kUHir"
out = sys.argv[1]
subprocess.run([sys.executable, "-m", "gdown", f"https://drive.google.com/uc?id={file_id}", "-O", out], check=True)
print(f"[OK] downloaded -> {out}")
PY
"$OUT"


