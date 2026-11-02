#!/usr/bin/env bash
# Save evidence of system services that can disturb an explanation. Read-only on the board. Writes results/llm/background_services.txt.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
{
  date -u +"recorded on PC %FT%TZ"
  ssh -F "$HOME/.config/twinread/ssh_config" -o BatchMode=yes uno-q '
    echo "--- fwupd-refresh start lines this boot"
    journalctl -b 0 --no-pager -o short-precise -u fwupd-refresh.service 2>/dev/null | grep -E "Starting|Finished|Failed" | tail -6 | cut -c1-150
    echo "--- the 11:11:50 event that slowed the third request of run ql_q25_q4km_t4"
    journalctl -b 0 --no-pager -o short-precise --since "2026-11-02 11:11:48" --until "2026-11-02 11:12:06" 2>/dev/null | grep -E "fwupd|FuEngine" | head -8 | cut -c1-150
    echo "--- timers"
    systemctl list-timers --all --no-pager 2>/dev/null | grep -E "NEXT|fwupd|apt|man-db|fstrim|e2scrub|dpkg|logrotate|update" | cut -c1-150
    echo "--- arduino-router-serial restart counter (last line) and this boot count"
    journalctl -b 0 --no-pager -u arduino-router-serial.service -n 200 2>/dev/null | grep -E "restart counter" | tail -1 | cut -c1-150
    journalctl -b 0 --no-pager -u arduino-router-serial.service 2>/dev/null | grep -c "Started arduino-router-serial"
    journalctl -b 0 --no-pager -o short-precise -u arduino-router-serial.service -n 3 2>/dev/null | cut -c1-150
    echo "--- thermal trip points, cpuss0 zone"
    z=/sys/class/thermal/thermal_zone3; cat $z/type; for t in $z/trip_point_*_temp; do echo "$(basename $t) $(cat $t) $(cat ${t%_temp}_type)"; done
    uptime'
} > "$REPO/results/llm/background_services.txt"
cat "$REPO/results/llm/background_services.txt"
