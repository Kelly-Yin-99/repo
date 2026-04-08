#!/usr/bin/env bash

set -e

# make temp dir
tmpdir=$(mktemp -d)

# download zip 
wget -O "$tmpdir/data.zip" https://raw.githubusercontent.com/joachimvandekerckhove/cogs205b-s26/main/modules/02-version-control/files/data.zip

# unzip
unzip "$tmpdir/data.zip" -d "$tmpdir"

# get today's date
today=$(date +%F)

mkdir -p "/workspace/repo/data/$today"
find "$tmpdir" -maxdepth 1 -type f -name "*.csv" -exec cp {} "/workspace/repo/data/$today/" \;
cd /workspace/repo
git add data scripts
git commit -m "Add CSV files"
git push
