# Installation Guide

This guide will help you install and set up WinstonAI on your system.

## Prerequisites

Before installing WinstonAI, ensure you have:

- **Python 3.8 or higher**
- **pip** (Python package manager)
- **Git** (for cloning the repository)
- **NVIDIA GPU** with CUDA support (recommended but not required)
- **16GB+ RAM** (32GB recommended for training)
- **10GB+ free disk space**

### GPU Requirements (Optional but Recommended)

For optimal performance:
- NVIDIA GPU with 8GB+ VRAM (12GB recommended)
- CUDA 11.8 or higher
- cuDNN 8.x

## Installation Methods

### Method 1: From Source (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ChipaDevTeam/WinstonAI.git
   cd WinstonAI
   ```

2. **Create a virtual environment:**
   ```bash
   # On Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import torch; print('PyTorch version:', torch.__version__)"
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

### Method 2: Using pip (Package Installation)

```bash
pip install -e .
```

This will install WinstonAI and all its dependencies.

### Method 3: Development Installation

For contributors:

```bash
# Clone the repository
git clone https://github.com/ChipaDevTeam/WinstonAI.git
cd WinstonAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## CUDA Installation (GPU Users)

### Linux

1. **Install NVIDIA drivers:**
   ```bash
   sudo apt update
   sudo apt install nvidia-driver-535  # Or latest version
   ```

2. **Install CUDA Toolkit:**
   ```bash
   # Download from NVIDIA website or use package manager
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
   sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
   sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
   sudo apt update
   sudo apt install cuda
   ```

3. **Install cuDNN:**
   - Download from [NVIDIA cuDNN page](https://developer.nvidia.com/cudnn)
   - Follow installation instructions

4. **Verify installation:**
   ```bash
   nvidia-smi
   nvcc --version
   ```

### Windows

1. **Install NVIDIA drivers:**
   - Download from [NVIDIA website](https://www.nvidia.com/drivers)
   - Run installer

2. **Install CUDA Toolkit:**
   - Download from [NVIDIA CUDA page](https://developer.nvidia.com/cuda-downloads)
   - Run installer
   - Add CUDA to PATH

3. **Install cuDNN:**
   - Download from [NVIDIA cuDNN page](https://developer.nvidia.com/cudnn)
   - Extract and copy files to CUDA directory

4. **Verify installation:**
   ```bash
   nvidia-smi
   nvcc --version
   ```

## Configuration

After installation, configure WinstonAI:

1. **Copy configuration templates:**
   ```bash
   cd src
   cp training_config.json training_config.local.json
   cp trading_config.json trading_config.local.json
   ```

2. **Edit configuration files:**
   - `training_config.local.json`: Training parameters
   - `trading_config.local.json`: Trading bot settings
   - Add your API keys and preferences

3. **Set up environment variables (optional):**
   ```bash
   # Create .env file
   echo "POCKETOPTION_EMAIL=your_email@example.com" > .env
   echo "POCKETOPTION_PASSWORD=your_password" >> .env
   ```

## Troubleshooting

### Common Issues

**1. CUDA not detected:**
```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA support
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**2. Out of memory errors:**
- Reduce batch size in `training_config.json`
- Close other GPU-intensive applications
- Use a GPU with more VRAM

**3. Import errors:**
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

**4. BinaryOptionsToolsV2 not found:**
```bash
# Install from source if not available on PyPI
pip install git+https://github.com/BinaryOptionsTools/BinaryOptionsToolsV2.git
```

**5. ta-lib installation fails:**

On Linux:
```bash
# Install system dependencies
sudo apt-get install build-essential
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install ta-lib
```

On Windows:
```bash
# Download pre-built wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.XX-cpXX-cpXX-win_amd64.whl
```

On Mac:
```bash
brew install ta-lib
pip install ta-lib
```

## Verification

After installation, verify everything works:

```bash
# Check Python version
python --version

# Check installed packages
pip list | grep -E "(torch|pandas|numpy|ta)"

# Run quick test
cd src
python -c "from train_gpu_optimized import AdvancedWinstonAI; print('Installation successful!')"
```

## Next Steps

Once installed:
1. Read the [Usage Guide](usage.md)
2. Check [Configuration Guide](configuration.md)
3. Review [GPU Optimization Guide](../src/README_GPU_OPTIMIZATION.md)
4. Start with [Quick Start](../README.md#quick-start)

## Getting Help

If you encounter issues:
- Check [Troubleshooting](#troubleshooting) section
- Search [existing issues](https://github.com/ChipaDevTeam/WinstonAI/issues)
- Create a new issue with details about your environment
- Join discussions on GitHub

## System Requirements Summary

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8 | 3.10+ |
| RAM | 8GB | 16GB+ |
| GPU VRAM | N/A (CPU) | 12GB |
| Disk Space | 5GB | 10GB+ |
| OS | Any | Linux |
| CUDA | N/A | 11.8+ |

---

**Note:** Training and live trading have different requirements. Live trading can work with CPU-only setups, but training benefits significantly from GPU acceleration.
