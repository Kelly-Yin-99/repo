#!/usr/bin/env bash

set -e

cd /workspace/repo

tmpdir=$(mktemp -d)
today=$(date +%F)

wget -O "$tmpdir/data.zip" "https://raw.githubusercontent.com/joachimvandekerckhove/cogs205b-s26/main/modules/02-version-control/files/data.zip"

unzip "$tmpdir/data.zip" -d "$tmpdir"

mkdir -p "data/$today"

find "$tmpdir" -maxdepth 1 -type f -name "*.csv" -exec cp {} "data/$today/" \;

git add scripts/fetch-csvs.sh "data/$today"
git commit -m "Add CSV files"
git push
