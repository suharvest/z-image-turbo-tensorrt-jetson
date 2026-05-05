#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <artifact_dir> <manifest_path>" >&2
  exit 2
fi

artifact_dir="$1"
manifest_path="$2"

if [ ! -d "$artifact_dir" ]; then
  echo "Artifact directory not found: $artifact_dir" >&2
  exit 1
fi

mkdir -p "$(dirname "$manifest_path")"

(
  cd "$artifact_dir"
  find . -type f \
    ! -name '.DS_Store' \
    ! -path './.git/*' \
    | LC_ALL=C sort \
    | xargs shasum -a 256
) > "$manifest_path"

echo "Wrote $manifest_path"
