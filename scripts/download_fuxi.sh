#!/bin/bash
# Download and extract the raw FuXi-S2S forecasts (FudanFuXi/FuXi-S2S on Hugging Face).
# Requires huggingface_hub and 7z. Split the download across jobs with PARTITION/TOTAL_PARTITIONS.
# Usage: bash scripts/download_fuxi.sh <OUTPUT_DIR> [PARTITION] [TOTAL_PARTITIONS]
# Example: bash scripts/download_fuxi.sh /path/to/FuXi-S2S 0 2
#          bash scripts/download_fuxi.sh /path/to/FuXi-S2S 1 2

DATA_DIR=${1:?usage: download_fuxi.sh <OUTPUT_DIR> [PARTITION] [TOTAL_PARTITIONS]}
PARTITION=${2:-0}
TOTAL_PARTITIONS=${3:-1}
PARALLEL=16

TMP_DIR="${DATA_DIR}/.download_tmp_${PARTITION}"
HF_CLI="huggingface-cli"

mkdir -p "$TMP_DIR"

echo "Partition ${PARTITION}/${TOTAL_PARTITIONS}: Fetching file list..."
FILE_LIST=$(python -c "
from huggingface_hub import list_repo_files
files = sorted([f for f in list_repo_files('FudanFuXi/FuXi-S2S', repo_type='dataset') if f.endswith('.7z')])
n = len(files)
chunk = n // $TOTAL_PARTITIONS
start = $PARTITION * chunk
end = n if $PARTITION == $TOTAL_PARTITIONS - 1 else ($PARTITION + 1) * chunk
for f in files[start:end]:
    print(f)
")

TOTAL=$(echo "$FILE_LIST" | wc -l)
echo "Partition ${PARTITION}: $TOTAL files, $PARALLEL parallel workers"

export DATA_DIR TMP_DIR HF_CLI

download_and_extract() {
    fname="$1"
    basename="${fname%.7z}"
    dest="${DATA_DIR}/${basename}"
    archive="${TMP_DIR}/${fname}"

    rm -rf "$dest"

    $HF_CLI download FudanFuXi/FuXi-S2S "$fname" \
        --repo-type dataset \
        --local-dir "$TMP_DIR" \
        --quiet 2>/dev/null || {
        echo "FAILED: ${fname}"
        return 0
    }

    mkdir -p "$dest"
    7z x "$archive" -o"$dest" -y > /dev/null
    rm -f "$archive"
    echo "DONE: ${basename}"
}
export -f download_and_extract

echo "$FILE_LIST" | xargs -P $PARALLEL -I {} bash -c 'download_and_extract "$@"' _ {}

rm -rf "$TMP_DIR"
echo "Partition ${PARTITION} complete."
