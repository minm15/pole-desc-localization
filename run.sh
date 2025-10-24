#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$ROOT_DIR/output.txt"

# Overwrite previous logs
: > "$OUT"

# Values to sweep
NLISTS=(16 256 1024 2048)
NPROBES=(1 4 8 16)

# Stream filter: drop progress-bar lines and remove carriage returns
filter_stream() {
  sed -u -e 's/\r//g' -e '/ETA:/d' -e '/Elapsed Time:/d'
}

for nprobe in "${NPROBES[@]}"; do
  for nlist in "${NLISTS[@]}"; do
    {
      echo "============================================================"
      echo "[$(date '+%F %T')] Starting run: nlist=${nlist}, nprobe=${nprobe}"
      echo "============================================================"
      echo "[clear] Running nclt/clear.sh ..."
    } >> "$OUT"

    ( cd "$ROOT_DIR/nclt" && bash clear.sh ) 2>&1 | filter_stream >> "$OUT"

    {
      echo "[run] python src/ncltpoles.py --nlist ${nlist} --nprobe ${nprobe}"
    } >> "$OUT"

    # stdbuf makes the stream line-buffered so the filter can work nicely
    ( cd "$ROOT_DIR" && stdbuf -oL -eL python src/ncltpoles.py --nlist "${nlist}" --nprobe "${nprobe}" ) \
      2>&1 | filter_stream >> "$OUT"

    {
      echo "[$(date '+%F %T')] Finished run: nlist=${nlist}, nprobe=${nprobe}"
      echo
    } >> "$OUT"
  done
done

echo "Done. Full log saved to: $OUT"