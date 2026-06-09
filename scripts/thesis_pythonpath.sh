#!/usr/bin/env bash
# Prepend thesis repo root to PYTHONPATH for shared_splits / shared_calibration.
# Source after ROOT (and optionally REPO_ROOT) is set.

: "${ROOT:?ROOT must be set before sourcing scripts/thesis_pythonpath.sh}"

if [[ -z "${REPO_ROOT:-}" ]]; then
  _search="$ROOT"
  while [[ "$_search" != "/" ]]; do
    if [[ -f "$_search/requirements-thesis-all.txt" ]]; then
      REPO_ROOT="$_search"
      export REPO_ROOT
      break
    fi
    _search="$(dirname "$_search")"
  done
fi

if [[ -n "${REPO_ROOT:-}" ]]; then
  export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
fi
