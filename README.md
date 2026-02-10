#Surgical Auto-Tagger v2.6 (SAT)

A high-precision image tagging utility designed for dataset curation. This tool focuses on "surgical" precision, allowing for fine-grained threshold control and batch editing of booru-style tags across large datasets.


<img width="1920" height="1055" alt="image" src="https://github.com/user-attachments/assets/cce027db-2b52-452d-b592-bde11ceac8a8" />



#Core Features

#Multi-Model Ensemble:

  Simultaneously run multiple state-of-the-art ONNX taggers:
  
  WD-EVA02-Large-v3
  
  WD-ViT-Large-v3
  
  WD-swinv2
  
  WD-ConvNext-v3 (tagger2.py and tagger3.py only)
  
  CL-Auto-Latest (tagger2.py and tagger3.py only)

  
  
#Aggregates results to maximize recall while maintaining high precision.

#Threshold Logic:

  Features separate threshold mapping for each model.
  Provides high-granularity control in the 0.05 to 0.40 range, where tag precision matters most.


#Tag Editing Suite:

  Batch Replace: Rename tags across the entire loaded dataset.
  Batch Prepend: Add specific trigger words or quality tags to the start of every file.
  Live Preview: Dedicated image previewer to verify visual accuracy against generated tags.
  In-Memory Buffer: Edit tags in the table UI and commit to disk only when satisfied.
  Smart Rating Detection:
  Broad-spectrum scanning for rating tags (general, sensitive, questionable, explicit).
  Automatically cleans and formats rating postfixes for standardized dataset structures.


#Installation

#Prerequisites

  Python 3.10+
  NVIDIA GPU with CUDA 12.x support (for onnxruntime-gpu)
  Minimum 8GB VRAM recommended for ensemble workflows.
  CPU fallback in case of no GPU. Any decent recent CPU can work reasonably fast. 

#Quick Start

#Clone and Enter:

  git clone https://github.com/paoloinverse/Surgical-Auto-Tagger.git
  
  cd Surgical-Auto-Tagger


#Environment Setup:

  Run the provided install script to handle dependencies and avoid ONNX/PySide conflicts:

  chmod 755 ./*.sh
  ./install.sh 


#Activate the Python environment if you didn’t already:

  source venv/bin/activate

#Launch tagger.py (fastest version):
  
  ./run.sh 

  #PLEASE NOTE: the first run of a model requires to download the model data. Some models are quite large, at oger 1.2GB in size and will take time. Subsequent runs will be much faster.
  

#Generic launcher (agnostic, works across most configurations), note: this will preset required environment variables for CUDA:

  ./run_generic.sh 

#Generic launcher for tagger3 (agnostic, works across most configurations), note: this will launch the latest version:

  ./run_generic_tagger3.sh

#Usage Workflow

  Load: Use Open Dataset to select your image directory.

  Configure: Adjust the quadratic sliders or numerical inputs for each model in the ensemble. Enable Force Rating Postfix if required for your training pipeline.

  Process: Click START ENSEMBLE. Watch the diagnostic log for real-time rating detection and model loading status.

  Refine: Use the Find/Replace or Prepend tools to clean up specific tags.

  Commit: Click SAVE ALL TXT to write changes back to your .txt files.


#Credits

  Models developed by SmilingWolf.
  Built with PySide6 and ONNX Runtime.
