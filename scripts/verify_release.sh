#!/usr/bin/env bash
# Pre-release verification gate for Remedy PDF Desktop.
#
# Runs the full pytest suite (unit + integration) and the slow E2E pipeline
# suite against the corpus under tests/fixtures/corpus/. Both must be green
# for this script to exit 0.
#
# The E2E tests use FastAPI's TestClient, which spins the backend up in the
# current process — there is no separate uvicorn / Ollama daemon to start
# or tear down. Vision is intentionally disabled inside the test so this
# script runs cleanly on machines without a local Ollama.
#
# Usage:
#     bash scripts/verify_release.sh

set -euo pipefail

# Resolve repo root from this script's location so the suite can be run
# from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Prefer a project-local virtualenv / python if one exists; otherwise fall
# back to whatever ``python`` resolves to on PATH.
PYTHON="${PYTHON:-python}"
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "FAIL: python interpreter '${PYTHON}' not found on PATH" >&2
    exit 1
fi

echo "=============================================================="
echo "Remedy PDF Desktop — pre-release verification"
echo "=============================================================="
echo "Repo root : ${REPO_ROOT}"
echo "Python    : $(${PYTHON} -c 'import sys; print(sys.executable)')"
echo "Version   : $(${PYTHON} --version)"
echo "Started   : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo

START_EPOCH=$(date +%s)

PASS_COUNT_UNIT=0
PASS_COUNT_E2E=0

pytest_pass_count() {
    # Parse "N passed" out of the last line of pytest output if present.
    # Used only for the final summary — does not gate exit status.
    local log="$1"
    local match
    match="$(grep -oE '[0-9]+ passed' "${log}" | tail -n 1 || true)"
    if [[ -n "${match}" ]]; then
        echo "${match%% *}"
    else
        echo "0"
    fi
}

run_stage() {
    local label="$1"
    shift
    echo "--------------------------------------------------------------"
    echo "▶ ${label}"
    echo "  $ $*"
    echo "--------------------------------------------------------------"
    "$@"
    local rc=$?
    if [[ ${rc} -ne 0 ]]; then
        echo "FAIL: ${label} exited with status ${rc}" >&2
        exit ${rc}
    fi
    echo
}

# Capture pytest output so we can extract pass counts for the final line.
UNIT_LOG="$(mktemp -t remedy-verify-unit.XXXXXX.log)"
E2E_LOG="$(mktemp -t remedy-verify-e2e.XXXXXX.log)"
trap 'rm -f "${UNIT_LOG}" "${E2E_LOG}"' EXIT

# ---------------------------------------------------------------------
# Stage 1: full unit + integration suite (no markers, 0-failure gate).
# ---------------------------------------------------------------------
echo "--------------------------------------------------------------"
echo "▶ Stage 1: pytest tests/ -q (full suite, no markers)"
echo "--------------------------------------------------------------"
if ! "${PYTHON}" -m pytest tests/ -q 2>&1 | tee "${UNIT_LOG}"; then
    echo
    echo "FAIL: unit/integration suite (stage 1) had failures — see output above." >&2
    exit 1
fi
PASS_COUNT_UNIT=$(pytest_pass_count "${UNIT_LOG}")
echo

# ---------------------------------------------------------------------
# Stage 2: slow E2E corpus suite (opt-in via --runslow).
# ---------------------------------------------------------------------
echo "--------------------------------------------------------------"
echo "▶ Stage 2: pytest tests/test_e2e_pipeline.py -v --runslow"
echo "--------------------------------------------------------------"
if ! "${PYTHON}" -m pytest tests/test_e2e_pipeline.py -v --runslow 2>&1 | tee "${E2E_LOG}"; then
    echo
    echo "FAIL: E2E pipeline suite (stage 2) had failures — see output above." >&2
    exit 1
fi
PASS_COUNT_E2E=$(pytest_pass_count "${E2E_LOG}")

END_EPOCH=$(date +%s)
ELAPSED=$((END_EPOCH - START_EPOCH))
TOTAL=$((PASS_COUNT_UNIT + PASS_COUNT_E2E))

echo
echo "=============================================================="
echo "PASS: ${TOTAL} tests green in ${ELAPSED} seconds"
echo "       (unit+integration: ${PASS_COUNT_UNIT}, E2E corpus: ${PASS_COUNT_E2E})"
echo "=============================================================="
exit 0
