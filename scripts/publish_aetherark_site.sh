#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
source_dir="$project_root/site/"

: "${AETHERARK_REMOTE_HOST:?set AETHERARK_REMOTE_HOST (for example, user@host)}"
: "${AETHERARK_REMOTE_DIR:?set AETHERARK_REMOTE_DIR to the destination directory}"

test -f "${source_dir}index.html"
test -f "${source_dir}data.html"
test -f "${source_dir}modeling.html"
test -f "${source_dir}training-results.html"
test -f "${source_dir}assets/training-history.json"

rsync -az --delete \
  -e 'ssh -o BatchMode=yes -o ConnectTimeout=15' \
  "$source_dir" "$AETHERARK_REMOTE_HOST:$AETHERARK_REMOTE_DIR"

ssh -o BatchMode=yes -o ConnectTimeout=15 "$AETHERARK_REMOTE_HOST" \
  "test -f '$AETHERARK_REMOTE_DIR/index.html' && test -f '$AETHERARK_REMOTE_DIR/data.html' && test -f '$AETHERARK_REMOTE_DIR/modeling.html' && test -f '$AETHERARK_REMOTE_DIR/training-results.html'"

printf 'Published site files to %s:%s\n' \
  "$AETHERARK_REMOTE_HOST" "$AETHERARK_REMOTE_DIR"
