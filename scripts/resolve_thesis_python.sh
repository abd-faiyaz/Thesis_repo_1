# Shared Python resolver for BM1, Pattern A, and Pattern B.
# Source from a pipeline run script after ROOT is set.
#
# Environment overrides (optional):
#   PYTHON=/path/to/python     — use this interpreter (skip auto-detect)
#   THESIS_VENV=/path/to/venv  — venv directory (default: <repo>/thesis_venv)
#
# Sets: PYTHON, and THESIS_VENV when thesis_venv is found or set.

: "${ROOT:?ROOT must be set before sourcing scripts/resolve_thesis_python.sh}"

if [[ -z "${PYTHON:-}" ]]; then
  _tv="${THESIS_VENV:-}"
  if [[ -z "$_tv" ]]; then
    _search="$ROOT"
    while [[ "$_search" != "/" ]]; do
      if [[ -x "$_search/thesis_venv/bin/python" ]]; then
        _tv="$_search/thesis_venv"
        break
      fi
      _search="$(dirname "$_search")"
    done
  fi
  if [[ -n "$_tv" ]] && [[ -x "$_tv/bin/python" ]]; then
    THESIS_VENV="$_tv"
    PYTHON="$_tv/bin/python"
  else
    PYTHON="python3"
  fi
  export PYTHON
  [[ -n "${THESIS_VENV:-}" ]] && export THESIS_VENV
fi

# Repo root on PYTHONPATH so shared_splits / shared_calibration import without extra setup.
if [[ -z "${THESIS_REPO_ROOT:-}" ]]; then
  _search="${ROOT:-}"
  while [[ "$_search" != "/" ]]; do
    if [[ -f "$_search/requirements-thesis-all.txt" ]]; then
      THESIS_REPO_ROOT="$_search"
      export THESIS_REPO_ROOT
      break
    fi
    _search="$(dirname "$_search")"
  done
fi
if [[ -n "${THESIS_REPO_ROOT:-}" ]]; then
  export PYTHONPATH="${THESIS_REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
fi

thesis_all_requirements_path() {
  local _s="$ROOT"
  while [[ "$_s" != "/" ]]; do
    if [[ -f "$_s/requirements-thesis-all.txt" ]]; then
      echo "$_s/requirements-thesis-all.txt"
      return 0
    fi
    _s="$(dirname "$_s")"
  done
  echo "$ROOT/requirements.txt"
}
