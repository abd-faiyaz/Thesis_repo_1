#!/usr/bin/env bash
# Resolve thesis_venv python for this pipeline.
PIPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$PIPE_ROOT/.." && pwd)"
if [[ -x "$REPO_ROOT/thesis_venv/bin/python" ]]; then
  echo "$REPO_ROOT/thesis_venv/bin/python"
else
  echo "${PYTHON:-python3}"
fi
