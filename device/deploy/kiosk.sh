#!/bin/bash
# X session for the TwinRead kiosk: Chromium full screen on the local app, relaunched if it exits or the display changes
# (a dock or monitor can appear after boot). Started by LightDM's auto-login. With no display attached it starts no browser:
# nobody could see it, and Chromium costs CPU and heat and keeps the video feed running at full rate.
URL=${TWINREAD_URL:-http://127.0.0.1:8080/}
PROFILE="$HOME/.cache/twinread-kiosk"

xset s off s noblank -dpms 2>/dev/null

outputs() { cat /sys/class/drm/card*-*/status 2>/dev/null | tr '\n' ' '; }
screen_size() { xrandr --current 2>/dev/null | awk '/\*/ {print $1; exit}'; }
has_display() { outputs | grep -qw connected; }

while :; do
  until has_display; do sleep 5; done
  xrandr --auto 2>/dev/null
  until curl -fsS -m 1 -o /dev/null "${URL}state"; do sleep 1; done
  size=$(screen_size)
  size=${size:-1280x720}
  # A kiosk has nobody to dismiss a "restore pages" prompt after a power cut.
  sed -i 's/"exit_type":"[^"]*"/"exit_type":"Normal"/; s/"exited_cleanly":false/"exited_cleanly":true/' \
    "$PROFILE/Default/Preferences" 2>/dev/null
  echo "twinread-kiosk: starting chromium at $size, outputs: $(outputs)"
  chromium --kiosk --window-position=0,0 --window-size="${size/x/,}" --user-data-dir="$PROFILE" \
    --no-first-run --no-default-browser-check --noerrdialogs --disable-infobars --disable-session-crashed-bubble \
    --password-store=basic --disable-background-networking --disable-component-update --disable-sync \
    --disable-features=Translate,MediaRouter --overscroll-history-navigation=0 "$URL" &
  pid=$!
  seen=$(outputs)
  while kill -0 "$pid" 2>/dev/null; do
    sleep 2
    if [ "$(outputs)" != "$seen" ]; then
      echo "twinread-kiosk: display changed, restarting chromium"
      kill "$pid" 2>/dev/null
      break
    fi
  done
  wait "$pid" 2>/dev/null
  sleep 1
done
