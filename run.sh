#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_NVIDIA_PATH="$DIR/venv/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH=$(find "$VENV_NVIDIA_PATH" -type d -name "lib" | tr '\n' ':')$LD_LIBRARY_PATH
"$DIR/venv/bin/python" "$DIR/tagger.py"

