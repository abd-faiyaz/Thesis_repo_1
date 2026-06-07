# Activate shared thesis_venv for this pipeline. Source after ROOT is set.
: "${ROOT:?ROOT must be set before sourcing activate_thesis_env.sh}"
_d="$ROOT"
while [[ "$_d" != "/" ]]; do
  if [[ -f "$_d/scripts/source_thesis_python.sh" ]]; then
    # shellcheck source=/dev/null
    source "$_d/scripts/source_thesis_python.sh"
    return 0 2>/dev/null || exit 0
  fi
  _d="$(dirname "$_d")"
done
PYTHON="${PYTHON:-python3}"
export PYTHON
return 1 2>/dev/null || exit 1
