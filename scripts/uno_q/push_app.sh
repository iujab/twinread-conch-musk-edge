#!/usr/bin/env bash
# Copy the device app and dashboard to the board and restart it there, from this PC. The page is back within seconds, in standby.
# The explainer's model and llama-server are not touched: device/README.md says how they get to a board.
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=(ssh -F "$HOME/.config/twinread/ssh_config" -o BatchMode=yes uno-q)
FILES=(device/explainer.py device/explain_prompt.py device/explain_check.py device/explain_vocab.py device/uno_q_app.py device/camera.py
       device/demo.py device/bridge.py device/twinread_runtime.py device/README.md device/ui/app.js device/ui/index.html device/ui/style.css device/deploy)
tar cf - "${FILES[@]}" | "${SSH[@]}" 'cd ~/twinread-conch-musk-edge && tar xf - && rm -rf device/ui/fonts device/ui/hammertone.png'
pid=$("${SSH[@]}" 'systemctl show twinread -p MainPID --value')
# Restart=always brings it back; no sudo needed for its own process. MainPID 0 means it is between runs, and kill 0 would hit this shell.
[ "${pid:-0}" -gt 0 ] && "${SSH[@]}" "kill $pid"
for _ in $(seq 60); do
  new=$("${SSH[@]}" 'systemctl show twinread -p MainPID --value; curl -fsS -m 2 -o /dev/null http://127.0.0.1:8080/state && echo up; true' | tr '\n' ' ')
  case "$new" in *up*) [ "${new%% *}" != "$pid" ] && [ "${new%% *}" != 0 ] && { echo "app back: pid ${new%% *}"; exit 0; } ;; esac
  sleep 1
done
echo "the app did not come back within a minute" >&2; exit 1
