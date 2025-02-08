#!/bin/bash

# Function to print the current step
function print_step {
    echo -e "\n\033[1;32m$1\033[0m"
}

# Function to retry pip install
function retry_pip_install {
    local package=$1
    local retries=5
    local count=0
    while [ $count -lt $retries ]; do
        pip install $package && return 0
        count=$((count + 1))
        echo "Retrying pip install... ($count/$retries)"
        sleep 2
        pip cache purge
    done
    return 1
}

# Check if the script is running in WSL
print_step "Checking if the script is running in WSL..."
if grep -qEi "(Microsoft|WSL)" /proc/version &> /dev/null ; then
    echo "Running in WSL"
else
    echo "This script must be run inside WSL"
    exit 1
fi

# Ensure WSL is set to version 2
print_step "Ensuring WSL is set to version 2..."
wsl_version=$(wsl --list --verbose | grep '*' | awk '{print $4}')
if [ "$wsl_version" != "2" ]; then
    echo "Setting WSL to version 2"
    wsl --set-default-version 2
fi

# Update and upgrade the system
print_step "Updating and upgrading the system..."
sudo apt-get update && sudo apt-get upgrade -y

# Install dependencies
print_step "Installing dependencies..."
sudo apt-get install -y build-essential dkms freeglut3 freeglut3-dev libxi-dev libxmu-dev

# Remove any existing CUDA installation
print_step "Checking for and removing any existing CUDA installation..."
if dpkg-query -W -f='${Status}' cuda 2>/dev/null | grep -q "ok installed"; then
    echo "CUDA is installed. Removing CUDA."
    sudo apt-get --purge remove -y cuda*
    sudo apt-get autoremove -y
    sudo apt-get autoclean
fi

# Add NVIDIA package repositories
sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update

# Install CUDA 12.3
print_step "Installing CUDA 11.8..."
sudo apt-get remove -y cuda-12-3
sudo apt autoremove -y
sudo apt-get install -y cuda-11-8

# Install CUDA toolkit
print_step "Installing CUDA toolkit..."
sudo apt-get install -y nvidia-cuda-toolkit

# Remove any existing cuDNN installation
print_step "Checking for and removing any existing cuDNN installation..."
if [ -d "/usr/local/cuda/include/cudnn*.h" ] || [ -d "/usr/local/cuda/lib64/libcudnn*" ]; then
    echo "cuDNN is installed. Removing cuDNN."
    sudo rm -rf /usr/local/cuda/include/cudnn*.h
    sudo rm -rf /usr/local/cuda/lib64/libcudnn*
fi

# Download and install cuDNN library
print_step "Downloading and installing cuDNN library..."
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz
tar -xvf cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz
sudo cp cudnn-linux-x86_64-8.7.0.84_cuda11-archive/include/* /usr/local/cuda/include/
sudo cp cudnn-linux-x86_64-8.7.0.84_cuda11-archive/lib/* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

#also copying here 
sudo cp cudnn-linux-x86_64-8.7.0.84_cuda11-archive/include/* /usr/include/
sudo cp cudnn-linux-x86_64-8.7.0.84_cuda11-archive/lib/* /usr/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/lib64/libcudnn*

# Set environment variables for CUDA
print_step "Setting environment variables for CUDA..."
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Install software-properties-common
print_step "Installing software-properties-common..."
sudo apt-get install -y software-properties-common

# Check if Python is installed
print_step "Checking if Python is installed..."
if ! command -v python3 &> /dev/null ; then
    echo "Python is not installed. Installing Python 3.10."
    sudo apt-get install -y python3.10 python3.10-venv python3.10-dev
fi

# Ensure pip is installed
print_step "Checking if pip is installed..."
if ! command -v pip &> /dev/null ; then
    echo "pip is not installed. Installing pip."
    sudo apt-get install -y python3-pip
fi

# Install Miniconda
print_step "Checking if Miniconda is installed..."
if [ ! -d "$HOME/miniconda" ]; then
    echo "Miniconda is not installed. Installing Miniconda."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc

    source $HOME/miniconda/etc/profile.d/conda.sh
else
    echo "Miniconda already installed."
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    source $HOME/miniconda/etc/profile.d/conda.sh
fi

# Ensure conda is available in the current shell session
print_step "Ensuring conda is available in the current shell session..."
export PATH="$HOME/miniconda/bin:$PATH"
source $HOME/miniconda/etc/profile.d/conda.sh

# Delete existing conda environment if it exists
print_step "Checking for and deleting existing conda environment if it exists..."
if conda info --envs | grep -q 'algotrader'; then
    echo "Conda environment 'algotrader' already exists. Deleting the environment."
    conda remove --name algotrader --all -y
fi

# Create and activate conda environment with Python 3.10
print_step "Creating and activating conda environment with Python 3.10..."
conda create --name algotrader python=3.10 -y
source $HOME/miniconda/bin/activate algotrader

# Add conda-forge channel
print_step "Adding conda-forge channel..."
conda config --add channels conda-forge
conda config --set channel_priority strict

# Remove any existing TensorFlow installation
print_step "Removing any existing TensorFlow installation..."
pip uninstall -y tensorflow tensorflow-gpu

# Install TensorFlow 2.15.1 with GPU support using pip
print_step "Installing TensorFlow 2.14.1 with GPU support using pip..."
retry_pip_install "tensorflow[and-cuda]==2.14.1"

# Install the correct version of protobuf
print_step "Installing the correct version of protobuf..."
retry_pip_install "protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3"

# Backup and modify requirements file to avoid conflicts
print_step "Backing up and modifying requirements file to avoid conflicts..."
cp INSTALL/requirements_gpu_linux.txt INSTALL/requirements_gpu_linux_backup.txt
sed -i '/protobuf/d' INSTALL/requirements_gpu_linux.txt

# Install additional dependencies
print_step "Installing additional dependencies..."
if [ -f "INSTALL/requirements_gpu_linux.txt" ]; then
    retry_pip_install "-r INSTALL/requirements_gpu_linux.txt"
else
    echo "INSTALL/requirements_gpu_linux.txt not found. Please ensure it exists in the current directory."
fi

# Restore the original requirements file
print_step "Restoring the original requirements file..."
mv requirements_gpu_linux_backup.txt requirements_gpu_linux.txt

# Install Clang 16
print_step "Installing Clang 16..."
wget https://apt.llvm.org/llvm.sh
chmod u+x llvm.sh
sudo ./llvm.sh 16

# Check if Bazel 6.1.0 is installed, if not install it
print_step "Checking if Bazel 6.1.0 is installed..."
if ! command -v bazel &> /dev/null ; then
    echo "Bazel 6.1.0 is not installed. Installing Bazel 6.1.0."
    sudo apt-get install -y apt-transport-https curl gnupg
    curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
    sudo mv bazel-archive-keyring.gpg /usr/share/keyrings/
    sudo sh -c 'echo "deb [signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" > /etc/apt/sources.list.d/bazel.list'
    sudo apt-get update && sudo apt-get install -y bazel-6.1.0
fi

echo "Installation complete. To activate the environment, run: conda activate algotrader"

print_step "Installing tf_keras..."
pip install tf_keras==2.14.1
export TF_USE_LEGACY_KERAS=True
echo 'export TF_USE_LEGACY_KERAS=True' >> ~/.bashrc



# Verify installations
print_step "Verifying CUDA installation..."
nvcc --version

print_step "Verifying cuDNN installation..."
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2

print_step "Verifying TensorFlow installation..."
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('Num GPUs Available:', len(tf.config.experimental.list_physical_devices('GPU')))"