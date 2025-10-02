#!/usr/bin/env bash
set -euo pipefail

# install_termoweb_card.sh
# Copy the TermoWeb schedule card into the public www path.
# Usage: run from an HA shell (SSH add-on), no args needed.
# Optionally pass a custom config dir as the first argument.

CONFIG_DIR="${1:-/config}"
CARD_NAME="termoweb_schedule_card.js"
DEST_DIR="${CONFIG_DIR}/www/termoweb"
DEST="${DEST_DIR}/${CARD_NAME}"

SOURCES=(
  "${CONFIG_DIR}/custom_components/termoweb/assets/${CARD_NAME}"
  "${CONFIG_DIR}/custom_components/termoweb/www/${CARD_NAME}"
  "${CONFIG_DIR}/custom_components/termoweb/${CARD_NAME}"
)

echo "[*] Using config dir: ${CONFIG_DIR}"
mkdir -p "${DEST_DIR}"

FOUND_SRC=""
for CANDIDATE in "${SOURCES[@]}"; do
  if [ -f "${CANDIDATE}" ]; then
    FOUND_SRC="${CANDIDATE}"
    break
  fi
done

if [ -n "${FOUND_SRC}" ]; then
  echo "[+] Found ${CARD_NAME} at ${FOUND_SRC}"
  cp -f "${FOUND_SRC}" "${DEST}"
  echo "[+] Copied ${FOUND_SRC} -> ${DEST}"
  echo "[i] Now add a Lovelace resource in the UI:"
  echo "    URL: /local/termoweb/${CARD_NAME}"
  echo "    Type: JavaScript Module"
else
  echo "[!] Could not find ${CARD_NAME} in any known location. Checked:"
  for CANDIDATE in "${SOURCES[@]}"; do
    echo "    - ${CANDIDATE}"
  done
fi
