#!/usr/bin/env bash
# Run the .par files of one or more jobs lists in order (one search per tile); resumable.
# A missing image is waited for only while `decolace_prep.py fetch` is still running; tiles skipped
# in the first pass are tried again at the end. GPU out-of-memory: one retry at half the batch size.
# A finished tile gets <outputDir>/.done.
# usage: BATCH=4 bash examples/decolace_tric/run_queue.sh JOBS1.txt [JOBS2.txt ...]
set -uo pipefail
B=${BATCH:-4}
JOBS=(); for j in "$@"; do JOBS+=("$(realpath "$j")"); done
cd "$(dirname "$(realpath "$0")")/../.."          # repository root, for python -m smap.*
for pass in 1 2; do
  for jobs in "${JOBS[@]}"; do
    n=$(grep -c . "$jobs"); i=0
    while read -r par; do
      [ -z "$par" ] && continue
      i=$((i + 1))
      out=$(awk '$1 == "outputDir" {print $2; exit}' "$par")
      img=$(awk '$1 == "imageFile" {print $2; exit}' "$par")
      [ -f "$out/.done" ] && continue
      while [ ! -f "$img" ] && pgrep -f "decolace_prep.py fetch" > /dev/null; do sleep 30; done
      if [ ! -f "$img" ]; then
        [ "$pass" = 2 ] && echo "[$i/$n] no image (see fetch_failed.txt): $(basename "$img")"
        continue
      fi
      mkdir -p "$out"
      echo "[$i/$n] $(date '+%a %H:%M') pass $pass $(basename "$jobs" .txt) $(basename "$par" .par)"
      if python -m smap.search_global "$par" --batch-size "$B" --log "$out.log"; then
        touch "$out/.done"
      elif grep -qi "out of memory" "$out.log"; then
        echo "  out of GPU memory: retry with --batch-size $((B / 2))"
        python -m smap.search_global "$par" --batch-size $((B / 2)) --log "$out.retry.log" && touch "$out/.done" \
          || echo "  FAILED: $out.retry.log"
      else
        echo "  FAILED: $out.log"
      fi
    done < "$jobs"
  done
done
echo "queue finished $(date)"
