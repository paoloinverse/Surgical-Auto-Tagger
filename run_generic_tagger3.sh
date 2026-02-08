#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_NVIDIA_PATH=$(find "$DIR/venv/lib" -maxdepth 3 -type d -path "*/site-packages/nvidia" 2>/dev/null | head -n 1)
if [ -d "$VENV_NVIDIA_PATH" ]; then
EXTRA_LIBS=$(find "$VENV_NVIDIA_PATH" -type d -name "lib" | tr '\n' ':')
export LD_LIBRARY_PATH="${EXTRA_LIBS}${LD_LIBRARY_PATH}"
fi
exec "$DIR/venv/bin/python" "$DIR/tagger3.py"