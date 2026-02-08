#!/bin/bash

CUDNN_PATH="venv/lib/python3.12/site-packages/nvidia/cudnn/lib"

if [ -d "$CUDNN_PATH" ]; then
export LD_LIBRARY_PATH="$CUDNN_PATH:$LD_LIBRARY_PATH"

if ! grep -q "LD_LIBRARY_PATH" venv/bin/activate; then
    echo "" >> venv/bin/activate
    echo "export LD_LIBRARY_PATH=\"$CUDNN_PATH:\$LD_LIBRARY_PATH\"" >> venv/bin/activate
fi

python tagger.py


else
echo "Directory not found: $CUDNN_PATH"
exit 1
fi
