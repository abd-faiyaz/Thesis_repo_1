#!/usr/bin/env bash
# Locate repo scripts/resolve_thesis_python.sh from pipeline ROOT and source it.
# Usage (in a pipeline script, after ROOT=...):
#   source "$(dirname "$0")/../../scripts/source_thesis_python.sh"   # BM1: adjust depth
# Or from repo: source scripts/source_thesis_python.sh with ROOT exported.
#
# Prefer: call _source_thesis_python_from_root after setting ROOT.

_source_thesis_python_from_root() {
  : "${ROOT:?ROOT must be set}"
  local _d="$ROOT"
  while [[ "$_d" != "/" ]]; do
    if [[ -f "$_d/scripts/resolve_thesis_python.sh" ]]; then
      # shellcheck source=/dev/null
      source "$_d/scripts/resolve_thesis_python.sh"
      return 0
    fi
    _d="$(dirname "$_d")"
  done
  PYTHON="${PYTHON:-python3}"
  export PYTHON
  return 1
}

if [[ -n "${ROOT:-}" ]]; then
  _source_thesis_python_from_root
fi
