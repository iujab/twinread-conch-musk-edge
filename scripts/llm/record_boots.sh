#!/usr/bin/env bash
# Save the board's boot list, plus how each recent boot ended, to results/llm/boots_<label>.txt. Read-only on the board.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$REPO/results/llm/boots_$1.txt"
{
  date -u +"recorded on PC %FT%TZ"
  ssh -F "$HOME/.config/twinread/ssh_config" -o BatchMode=yes uno-q '
    uptime
    journalctl --list-boots --no-pager
    for b in $(journalctl --list-boots --no-pager | awk "NR>1 && \$1<0 {print \$1}"); do
      echo "--- how boot $b ended (last sudo/logind reboot lines, then final line)"
      journalctl -b "$b" --no-pager -o short-precise 2>/dev/null | grep -E "COMMAND=.*(reboot|shutdown|poweroff)|The system will reboot|The system will power off" | tail -3 | cut -c1-160
      journalctl -b "$b" --no-pager -o short-precise -n 1 2>/dev/null | tail -1 | cut -c1-160
    done'
} > "$OUT"
cat "$OUT"
