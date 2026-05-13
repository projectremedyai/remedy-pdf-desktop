#!/usr/bin/env bash
# Build, sign, notarize, staple, and verify a macOS release of Remedy PDF Desktop.
#
# Reads signing + notarization credentials from .env.local at the repo root:
#   APPLE_SIGNING_IDENTITY  e.g. "Developer ID Application: Quang Phung (7XU3QW326W)"
#   APPLE_TEAM_ID           e.g. 7XU3QW326W
#   APPLE_ID                Apple ID used for notarytool
#   APPLE_PASSWORD          App-specific password (xxxx-xxxx-xxxx-xxxx)
#
# Flags:
#   --dry-run     Run all checks and print what would happen; do not build.
#   --skip-sidecar  Reuse existing dist/remedy-pdf-desktop-backend (faster iteration).
#   --skip-web      Reuse existing web/dist (faster iteration; tauri normally rebuilds).
#
# Exit codes are non-zero on any failure. The script is idempotent; rerunning
# from a clean state should reproduce the same artifacts.

set -euo pipefail
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DRY_RUN=0
SKIP_SIDECAR=0
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=1 ;;
        --skip-sidecar) SKIP_SIDECAR=1 ;;
        *) echo "Unknown flag: ${arg}" >&2; exit 2 ;;
    esac
done

log()  { printf '\033[1;34m▶\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m✓\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!\033[0m %s\n' "$*" >&2; }
fail() { printf '\033[1;31m✗\033[0m %s\n' "$*" >&2; exit 1; }

# ───────────────────────────────────────────────────────────────
# Stage 0: load .env.local and validate signing credentials
# ───────────────────────────────────────────────────────────────
log "Loading .env.local"
[[ -f .env.local ]] || fail ".env.local not found at ${REPO_ROOT}/.env.local"
set -a
# shellcheck disable=SC1091
source .env.local
set +a

for var in APPLE_SIGNING_IDENTITY APPLE_TEAM_ID APPLE_ID APPLE_PASSWORD; do
    if [[ -z "${!var:-}" ]]; then
        fail "${var} is unset — populate it in .env.local"
    fi
done

if [[ "${APPLE_PASSWORD}" == *"PASTE-"* || "${APPLE_PASSWORD}" == *"PLACEHOLDER"* ]]; then
    fail "APPLE_PASSWORD still holds the placeholder — paste the real app-specific password"
fi

if ! [[ "${APPLE_PASSWORD}" =~ ^[a-z]{4}-[a-z]{4}-[a-z]{4}-[a-z]{4}$ ]]; then
    warn "APPLE_PASSWORD doesn't match Apple's app-specific password format (xxxx-xxxx-xxxx-xxxx). Continuing anyway."
fi
ok "Env loaded — signing as ${APPLE_SIGNING_IDENTITY}"

# ───────────────────────────────────────────────────────────────
# Stage 1: verify required tools
# ───────────────────────────────────────────────────────────────
log "Checking toolchain"
for cmd in cargo npm python xcrun codesign spctl; do
    command -v "${cmd}" >/dev/null 2>&1 || fail "${cmd} not found on PATH"
done
if (( SKIP_SIDECAR == 0 )); then
    command -v pyinstaller >/dev/null 2>&1 || fail "pyinstaller not found; pip install pyinstaller or use --skip-sidecar"
fi
ok "Toolchain present"

# ───────────────────────────────────────────────────────────────
# Stage 2: verify signing identity is in the keychain
# ───────────────────────────────────────────────────────────────
log "Checking signing identity in keychain"
if ! security find-identity -v -p codesigning | grep -qF "${APPLE_SIGNING_IDENTITY}"; then
    fail "Signing identity not found: ${APPLE_SIGNING_IDENTITY}"
fi
ok "Signing identity available"

# ───────────────────────────────────────────────────────────────
# Stage 3: validate notarization credentials against Apple
# ───────────────────────────────────────────────────────────────
log "Validating notarytool credentials (talks to Apple)"
if ! xcrun notarytool history \
        --apple-id "${APPLE_ID}" \
        --password "${APPLE_PASSWORD}" \
        --team-id "${APPLE_TEAM_ID}" >/dev/null 2>&1; then
    fail "notarytool rejected the supplied credentials. Check APPLE_ID, APPLE_TEAM_ID, and APPLE_PASSWORD."
fi
ok "notarytool credentials accepted"

if (( DRY_RUN == 1 )); then
    ok "Dry run complete — all pre-flight checks passed. Re-run without --dry-run to build."
    exit 0
fi

# ───────────────────────────────────────────────────────────────
# Stage 4: rebuild Python sidecar via PyInstaller
# ───────────────────────────────────────────────────────────────
if (( SKIP_SIDECAR == 0 )); then
    log "Building Python sidecar (PyInstaller)"
    rm -rf build/project_remedy_backend dist/remedy-pdf-desktop-backend
    pyinstaller --clean --noconfirm backend/project_remedy_backend.spec
    [[ -x dist/remedy-pdf-desktop-backend/remedy-pdf-desktop-backend ]] \
        || fail "Sidecar binary missing after PyInstaller run"
    ok "Sidecar built at dist/remedy-pdf-desktop-backend/"
else
    warn "Skipping sidecar rebuild (--skip-sidecar); reusing existing dist/remedy-pdf-desktop-backend/"
fi

# ───────────────────────────────────────────────────────────────
# Stage 4b: pre-sign nested Mach-O files in sidecar + ollama trees
# ───────────────────────────────────────────────────────────────
# Tauri's bundler only signs the outer Contents/MacOS/app and the .app
# wrapper. It does NOT recurse into Contents/Resources/_up_/... where the
# PyInstaller sidecar and bundled ollama tree live. Apple's notary service
# rejects any unsigned Mach-O in the bundle. We pre-sign the source
# directories so Tauri copies already-signed binaries into the .app.
#
# Each binary needs:
#   --force            overwrite any prior signature
#   --options runtime  enable hardened runtime (notarization requirement)
#   --timestamp        embed a secure timestamp (notarization requirement)
#   --sign IDENTITY    Developer ID Application certificate
#
# Signing order: leaves first (deepest paths) so that any containers we
# sign later see valid nested signatures.

sign_mach_o_tree() {
    local root="$1"
    [[ -d "${root}" ]] || { warn "sign_mach_o_tree: ${root} does not exist; skipping"; return 0; }

    local count=0
    local errors=0
    # Sort by descending path depth so leaves get signed before containers.
    while IFS= read -r path; do
        # Skip symlinks — codesign follows them and signs the target instead,
        # which would double-sign or fail on already-processed targets.
        [[ -L "${path}" ]] && continue
        # Detect Mach-O by file type. .dylib/.so are the obvious cases; also
        # catches PyInstaller's bootloader binary and ollama's main executable.
        if file -b "${path}" | grep -qE '^(Mach-O|current ar archive)'; then
            if codesign --force --options runtime --timestamp \
                    --sign "${APPLE_SIGNING_IDENTITY}" "${path}" >/dev/null 2>&1; then
                count=$((count + 1))
            else
                errors=$((errors + 1))
                warn "codesign failed on ${path}"
            fi
        fi
    done < <(find "${root}" -type f \
                | awk -F/ '{print NF" "$0}' \
                | sort -rn -k1,1 \
                | cut -d' ' -f2-)
    if (( errors > 0 )); then
        fail "Pre-sign had ${errors} failures in ${root}"
    fi
    ok "Pre-signed ${count} Mach-O files under ${root}"
}

log "Pre-signing Mach-O leaves in sidecar tree"
sign_mach_o_tree "dist/remedy-pdf-desktop-backend"

log "Pre-signing Mach-O leaves in bundled ollama tree"
sign_mach_o_tree "resources/ollama/macos"

# ───────────────────────────────────────────────────────────────
# Stage 5: build, sign, and notarize via Tauri
# ───────────────────────────────────────────────────────────────
log "Running tauri build (sign + notarize via env vars)"
# Tauri v2 picks up APPLE_SIGNING_IDENTITY / APPLE_TEAM_ID / APPLE_ID / APPLE_PASSWORD
# from the environment and runs codesign + notarytool + stapler inline.
npm run tauri:build
ok "tauri build finished"

# ───────────────────────────────────────────────────────────────
# Stage 6: locate produced artifacts
# ───────────────────────────────────────────────────────────────
BUNDLE_DIR="src-tauri/target/release/bundle"
APP_PATH="$(find "${BUNDLE_DIR}/macos" -maxdepth 1 -name '*.app' -print -quit 2>/dev/null || true)"
DMG_PATH="$(find "${BUNDLE_DIR}/dmg" -maxdepth 1 -name '*.dmg' -print -quit 2>/dev/null || true)"
[[ -n "${APP_PATH}" && -d "${APP_PATH}" ]] || fail "No .app bundle found under ${BUNDLE_DIR}/macos"
[[ -n "${DMG_PATH}" && -f "${DMG_PATH}" ]] || warn "No .dmg found under ${BUNDLE_DIR}/dmg — only verifying .app"
ok "Bundle:  ${APP_PATH}"
[[ -n "${DMG_PATH}" ]] && ok "Installer: ${DMG_PATH}"

# ───────────────────────────────────────────────────────────────
# Stage 7: verify signature, hardened runtime, notarization, gatekeeper
# ───────────────────────────────────────────────────────────────
log "Verifying code signature (deep, strict)"
codesign --verify --deep --strict --verbose=2 "${APP_PATH}" \
    || fail "codesign verification failed for ${APP_PATH}"

log "Checking hardened runtime flag"
# codesign -dvv writes its display to stderr and may exit non-zero on bundles
# under pipefail, so capture once and inspect the string directly rather than
# piping into grep.
codesign_info="$(codesign -dvv "${APP_PATH}" 2>&1 || true)"
if ! grep -q 'flags=.*runtime' <<<"${codesign_info}"; then
    echo "${codesign_info}" >&2
    fail "Hardened runtime not enabled on ${APP_PATH} — notarization will fail downstream"
fi

log "Checking notarization ticket is stapled on .app"
if ! xcrun stapler validate "${APP_PATH}" >/dev/null 2>&1; then
    warn "Stapler ticket missing on .app — attempting to staple now"
    xcrun stapler staple "${APP_PATH}" || fail "stapler staple failed for ${APP_PATH}"
fi

# Tauri v2 does NOT notarize the .dmg, only the .app. Submit the .dmg to
# notarytool and staple separately so Gatekeeper accepts it offline. Apple
# usually issues the ticket fast since the inner .app is already on record.
if [[ -n "${DMG_PATH}" ]]; then
    if ! xcrun stapler validate "${DMG_PATH}" >/dev/null 2>&1; then
        log "Notarizing .dmg (Tauri does not do this itself)"
        xcrun notarytool submit "${DMG_PATH}" \
            --apple-id "${APPLE_ID}" \
            --password "${APPLE_PASSWORD}" \
            --team-id "${APPLE_TEAM_ID}" \
            --wait \
            || fail "notarytool rejected the .dmg"
        log "Stapling ticket onto .dmg"
        xcrun stapler staple "${DMG_PATH}" || fail "stapler staple failed for ${DMG_PATH}"
    else
        ok ".dmg already has a stapled ticket"
    fi
fi

log "Gatekeeper assessment (spctl)"
spctl --assess --type execute --verbose=4 "${APP_PATH}" \
    || fail "spctl rejected the .app — Gatekeeper would block it"

if [[ -n "${DMG_PATH}" ]]; then
    spctl --assess --type open --context context:primary-signature --verbose=4 "${DMG_PATH}" \
        || fail "spctl rejected the .dmg — Gatekeeper would block it"
fi

# ───────────────────────────────────────────────────────────────
# Done
# ───────────────────────────────────────────────────────────────
echo
ok "Release artifacts are signed, notarized, stapled, and accepted by Gatekeeper."
echo "    App: ${APP_PATH}"
[[ -n "${DMG_PATH}" ]] && echo "    DMG: ${DMG_PATH}"
