# Booru Autotagger v2026.4

This application automates the first stage of image dataset curation by loading images from a directory, executing multi-model booru-style autotagging, performing tag reranking, and enabling manual editing before mass-exporting the metadata.

---

## Technical Overview

* **Processing Pipeline:** Loads local image directories, processes images through multiple concurrent tagging models, cross-references outputs, and applies a reranking algorithm to finalize tag confidence.
* **Hardware Execution:** Optimized for local execution. It automatically detects system capabilities and can run entirely within a CPU/RAM environment without dedicated GPU acceleration.
* **Model Management:** Required weights and model configurations are fetched and cached automatically upon the first execution.
* **Data Management:** Provides an interface to review and edit tags individually prior to executing a batch export of the metadata.

---

## Installation

The application executes within a isolated Python virtual environment (`venv`). The launcher script automatically detects and links nested CUDA binaries (such as NVIDIA site-packages) inside the virtual environment to `LD_LIBRARY_PATH` to ensure proper hardware execution.

### Prerequisites

* Python 3.10 or higher
* `bash` shell environment
* Internet connection (initial run only, for automated model downloading)

### Setup Steps

1. Clone or copy the repository to your local machine.
2. Navigate to the root directory containing `tagger5.py` and `run_generic_tagger5.sh`.
3. Create the virtual environment, upgrade baseline package managers, and install dependencies:

```bash
python3 -m venv venv
source venv/bin/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
deactivate

```

---

## Usage

Do not invoke the Python script directly if you require automated hardware library mapping / GPU execution. Execute the provided wrapper script to handle environment variables and launch the application:

```bash
chmod +x run_generic_tagger5.sh
./run_generic_tagger5.sh

```
