#!/usr/bin/env bash
# Run every .par listed in a jobs.txt (one search per micrograph), sequentially; resumable:
# a micrograph whose outputDir already has particles.csv is skipped.
# usage: bash examples/ms2_capsid/run_jobs.sh JOBS.txt [extra search_global arguments, e.g. --batch-size 8 --path-map /ms2=$MS2_DATA]
set -uo pipefail
jobs="$(realpath "$1")"; shift
cd "$(dirname "$(realpath "$0")")/../.."          # repository root, for python -m smap.*
n=$(grep -c . "$jobs"); i=0
while read -r par; do
  [ -z "$par" ] && continue
  i=$((i + 1))
  out=$(awk '$1 == "outputDir" {print $2; exit}' "$par")
  if [ -f "$out/particles.csv" ]; then echo "[$i/$n] done: $out"; continue; fi
  mkdir -p "$out"
  echo "[$i/$n] $(date +%H:%M) $(basename "$par")"
  python -m smap.search_global "$par" "$@" --log "$out.log" || echo "  FAILED (exit $?): see $out.log"
done < "$jobs"
