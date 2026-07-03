#!/usr/bin/env bash
set -u

infile="pdb_ids.txt"
outdir="dssp"
failed="failed_dssp.txt"

mkdir -p "$outdir"
: > "$failed"

while IFS= read -r pdb || [[ -n "$pdb" ]]; do
    pdb="$(printf '%s' "$pdb" | tr -d '\r' | awk '{$1=$1; print tolower($0)}')"

    [[ -z "$pdb" ]] && continue
    [[ "$pdb" == \#* ]] && continue

    out="${outdir}/${pdb}.dssp"
    tmp="${out}.part"

    if [[ -s "$out" ]]; then
        echo "skip $pdb"
        continue
    fi

    echo "download $pdb"

    url1="https://pdb-redo.eu/dssp/db/${pdb}/legacy"
    url2="https://pdb-redo.eu/db/${pdb}/${pdb}_final.dssp"

    if curl -fL \
        --retry 1 \
        --retry-delay 10 \
        --retry-all-errors \
        --connect-timeout 20 \
        --max-time 60 \
        -A "academic-dssp-batch-download/1.0" \
        -o "$tmp" \
        "$url1"; then
        mv "$tmp" "$out"
    else
        rm -f "$tmp"
        echo "legacy failed, try final: $pdb"

        if curl -fL \
            --retry 1 \
            --retry-delay 10 \
            --retry-all-errors \
            --connect-timeout 20 \
            --max-time 60 \
            -A "academic-dssp-batch-download/1.0" \
            -o "$tmp" \
            "$url2"; then
            mv "$tmp" "$out"
        else
            rm -f "$tmp"
            echo "$pdb" >> "$failed"
        fi
    fi

    sleep "$((RANDOM % 4)).3"
done < "$infile"

echo "done"
echo "failed list: $failed"