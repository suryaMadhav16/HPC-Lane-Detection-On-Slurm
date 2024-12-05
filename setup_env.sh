#!/bin/bash

# Create and activate new conda environment
echo "Creating conda environment: hpc-tusimple"
conda env create -f environment.yml

# Activate environment
echo "Activating environment..."
conda activate hpc-tusimple

# Verify installation
echo "Verifying PyTorch installation..."
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo "Verifying CUDA version..."
python -c "import torch; print('CUDA version:', torch.version.cuda)"

# Print environment information
echo "Environment Setup Complete. Installed packages:"
conda list

echo """
Setup complete! To activate the environment:
    conda activate hpc-tusimple

To deactivate:
    conda deactivate
"""