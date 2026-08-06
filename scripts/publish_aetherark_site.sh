#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dan/projects/nesso-finetune/PXR
source_dir="$project_root/site/"
remote_host=dan@178.156.236.86
remote_dir=/var/www/aetherark.com/sites/nesso-pxr-finetune/

test -f "${source_dir}index.html"
test -f "${source_dir}data.html"
test -f "${source_dir}modeling.html"
test -f "${source_dir}training-results.html"
test -f "${source_dir}assets/training-history.json"

rsync -az --delete \
  -e 'ssh -o BatchMode=yes -o ConnectTimeout=15' \
  "$source_dir" "$remote_host:$remote_dir"

ssh -o BatchMode=yes -o ConnectTimeout=15 "$remote_host" \
  'test -f /var/www/aetherark.com/sites/nesso-pxr-finetune/index.html && test -f /var/www/aetherark.com/sites/nesso-pxr-finetune/data.html && test -f /var/www/aetherark.com/sites/nesso-pxr-finetune/modeling.html && test -f /var/www/aetherark.com/sites/nesso-pxr-finetune/training-results.html'

printf 'Published https://aetherark.com/sites/nesso-pxr-finetune/\n'
