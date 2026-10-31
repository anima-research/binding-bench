#!/bin/bash
# Mirror this repo's TRACKED files to remote copies on machines without git: exactly `git ls-files`, extras deleted,
# except caches and __pycache__, which are protected from --delete. Each target is a directory that holds only the
# mirror; the project's other contents (lanes, results) live beside it, never inside it.
#
# Usage: tools/sync.sh [-n] [host:/path ...]   (-n: dry run; default targets: $BINDING_BENCH_SYNC_TARGETS from site.env)
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$ROOT/site.env" ] && source "$ROOT/site.env"
DRY=(); [ "${1:-}" = "-n" ] && { DRY=(--dry-run --itemize-changes); shift; }
TARGETS=("$@"); [ ${#TARGETS[@]} -gt 0 ] || read -r -a TARGETS <<< "${BINDING_BENCH_SYNC_TARGETS:-}"
[ ${#TARGETS[@]} -gt 0 ] || { echo "usage: tools/sync.sh [-n] host:/path ... (or set BINDING_BENCH_SYNC_TARGETS)" >&2; exit 2; }
cd "$ROOT"
[ -z "$(git status --porcelain --untracked-files=no)" ] || echo "WARN: uncommitted changes are synced as-is" >&2
LIST="$(mktemp)"; trap 'rm -f "$LIST"' EXIT
git ls-files > "$LIST"
for TARGET in "${TARGETS[@]}"; do
  rsync -a --delete --delete-excluded "${DRY[@]}" \
    --filter='P .cache/' --filter='P __pycache__/' \
    --include-from=<(sed 's|^|/|' "$LIST"; awk -F/ '{p=""; for(i=1;i<NF;i++){p=p"/"$i; print p"/"}}' "$LIST" | sort -u) \
    --exclude='*' ./ "$TARGET/"
  echo "synced $(wc -l < "$LIST") tracked files at $(git rev-parse --short HEAD) -> $TARGET"
done
