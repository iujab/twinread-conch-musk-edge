#!/usr/bin/env bash
# Print docs/deck/twinread_deck.html to docs/twinread_deck.pdf with headless Chromium (16:9, 13.333 x 7.5 in).
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
CHROME="${CHROME:-$(ls -d "$HOME"/.cache/ms-playwright/chromium-*/chrome-linux64/chrome 2>/dev/null | tail -1)}"
[ -x "$CHROME" ] || CHROME=$(command -v chromium || command -v google-chrome)
"$CHROME" --headless=new --disable-gpu --no-sandbox --no-pdf-header-footer --virtual-time-budget=4000 \
  --print-to-pdf="$D/../twinread_deck.pdf" "file://$D/twinread_deck.html" 2>/dev/null
echo "wrote $D/../twinread_deck.pdf"
