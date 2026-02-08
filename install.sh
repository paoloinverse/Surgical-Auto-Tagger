#!/bin/bash
#
#mkdir -p ~/AI_standalone_tagger
#cd ~/AI_standalone_tagger
#python3 -m venv venv
#source venv/bin/activate
#pip install --upgrade pip setuptools wheel
#pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
#pip install opencv-python-headless pillow numpy huggingface_hub pandas PySide6
#pip install nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12


#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python3 -m venv "$DIR/venv"
source "$DIR/venv/bin/activate"
pip install --upgrade pip setuptools wheel
pip install numpy pandas pillow opencv-python-headless huggingface_hub PySide6
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
pip install nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12