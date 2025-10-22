#!/usr/bin/env bash
# Install Python dependencies for Bazaar Laplace's Demon
set -euo pipefail

# optional venv creation (skipped if VIRTUAL_ENV is set)
if [[ -z "${VIRTUAL_ENV:-}" && ! -d "venv" ]]; then
  python3 -m venv venv
fi

# shellcheck disable=SC1091
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  # activate if not already in a venv
  # macOS/Linux
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo "Done."

