#!/usr/bin/env bash
# Make the board run TwinRead on power-up: the dashboard as a service with the camera and model off, then auto-login into a
# full-screen browser on it once a display is attached.
#   sudo device/deploy/install.sh            install and enable (takes effect now for the app, at next boot for the display)
#   sudo device/deploy/install.sh uninstall  back to the stock XFCE login screen
# Needs no network.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }
HERE=$(cd "$(dirname "$0")" && pwd)
DEVICE_DIR=$(dirname "$HERE")
KIOSK_USER=${SUDO_USER:-arduino}
REPO_DIR=$(dirname "$DEVICE_DIR")
UNIT=/etc/systemd/system/twinread.service
DEFAULTS=/etc/default/twinread
LIGHTDM=/etc/lightdm/lightdm.conf.d/50-twinread-kiosk.conf

# The CM bundle is private, so a public clone has only the open one, perhaps without its weights. The app then stays up and says so.
BUNDLE_DIR=$REPO_DIR/bundles/open
[ -f "$REPO_DIR/bundles/cm/model.onnx" ] && BUNDLE_DIR=$REPO_DIR/bundles/cm

fill() { sed "s|@DEVICE_DIR@|$DEVICE_DIR|g; s|@USER@|$KIOSK_USER|g; s|@BUNDLE_DIR@|$BUNDLE_DIR|g" "$1"; }

case "${1:-install}" in
  install)
    [ -x "$DEVICE_DIR/.venv/bin/python" ] || { echo "$DEVICE_DIR/.venv is missing: run install_offline.sh first" >&2; exit 1; }
    [ -f "$BUNDLE_DIR/model.onnx" ] || echo "no model weights in $BUNDLE_DIR: the dashboard will come up and say the model cannot load"
    install -m 755 "$HERE/kiosk.sh" /usr/local/bin/twinread-kiosk
    install -m 644 "$HERE/twinread-kiosk.desktop" /usr/share/xsessions/twinread-kiosk.desktop
    install -d /etc/lightdm/lightdm.conf.d
    fill "$HERE/50-twinread-kiosk.conf" > "$LIGHTDM"
    fill "$HERE/twinread.service" > "$UNIT"
    # Only if missing, so a local choice survives a reinstall. The unit works without this file too.
    [ -e "$DEFAULTS" ] || printf '%s\n' '# Extra arguments for uno_q_app.py. --standby: dashboard up, camera closed, model unloaded.' \
      'TWINREAD_ARGS=--standby' > "$DEFAULTS"
    # Demo slides are git-ignored data: build them here when the CRC test set is on this machine.
    if [ ! -f "$REPO_DIR/data/demo_slides/manifest.json" ] && [ -f "$REPO_DIR/data/crc_test_x.npy" ] && [ -f "$REPO_DIR/data/crc_test_y.npy" ]; then
      runuser -u "$KIOSK_USER" -- "$DEVICE_DIR/.venv/bin/python" "$REPO_DIR/scripts/uno_q/make_demo_slides.py" \
        || echo "demo slides not built: the dashboard will say so when demo mode is chosen"
    fi
    systemctl daemon-reload
    systemctl enable twinread.service
    systemctl restart twinread.service
    # Live view from the tailnet only: Tailscale forwards this port to the app, which itself stays on 127.0.0.1.
    if command -v tailscale >/dev/null; then
      tailscale serve --bg --tcp 8080 tcp://127.0.0.1:8080 >/dev/null 2>&1 || echo "tailscale serve not set (not logged in?)"
    fi
    # The explainer's files never come from here, and the app runs without them: Explain then shows the reading itself.
    if [ -x "$DEVICE_DIR/llm/bin/llama-server" ] && [ -f "$REPO_DIR/data/llm/twinread-explain-qwen3-v1-Q4_0-embQ8.gguf" ]; then
      echo "explainer: llama-server and its .gguf are in place"
    else
      echo "explainer: llama-server or its .gguf is missing, so Explain will show the reading itself; device/README.md says how to add them"
    fi
    echo "installed: twinread.service active=$(systemctl is-active twinread.service), bundle $BUNDLE_DIR; the display changes at next boot" ;;
  uninstall)
    systemctl disable --now twinread.service 2>/dev/null || true
    command -v tailscale >/dev/null && tailscale serve --tcp 8080 off >/dev/null 2>&1 || true
    rm -f "$UNIT" "$LIGHTDM" /usr/share/xsessions/twinread-kiosk.desktop /usr/local/bin/twinread-kiosk
    systemctl daemon-reload
    echo "removed: the stock login screen returns at next boot" ;;
  *) echo "usage: sudo $0 [install|uninstall]" >&2; exit 2 ;;
esac
