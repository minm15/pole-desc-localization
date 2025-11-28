#!/bin/bash

set -e

echo "--- [1/6] Installing System Dependencies (sudo password required) ---"
sudo apt update || true
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev curl llvm libncurses5-dev \
  libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev \
  libgdbm-dev libdb-dev libexpat1-dev libmpdec-dev libgmp-dev git

echo "--- [2/6] Setting up pyenv ---"
if [ ! -d "$HOME/.pyenv" ]; then
    git clone https://github.com/pyenv/pyenv.git ~/.pyenv
else
    echo "pyenv already installed in $HOME/.pyenv"
fi

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"

if command -v pyenv 1>/dev/null 2>&1; then
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"
else
    echo "Error: pyenv initialization failed."
    exit 1
fi

if ! grep -q 'export PYENV_ROOT="$HOME/.pyenv"' ~/.bashrc; then
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    echo "Added pyenv configuration to ~/.bashrc"
fi

echo "--- [3/6] Installing Python 3.8.20 ---"
if ! pyenv versions | grep -q "3.8.20"; then
    pyenv install 3.8.20
else
    echo "Python 3.8.20 is already installed."
fi

pyenv global 3.8.20
eval "$(pyenv init -)"

echo "Current Python version: $(python --version)"

echo "--- [4/6] Creating Virtual Environment (.env) ---"
if [ ! -d ".env" ]; then
    python -m venv .env
    echo "Virtual environment created at ./.env"
else
    echo "Virtual environment .env already exists."
fi

echo "--- [5/6] Installing Python Packages ---"
source .env/bin/activate

pip install --upgrade pip

echo "Installing PyTorch..."
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

if [ -f "requirements.txt" ]; then
    echo "Installing requirements.txt..."
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found. Skipping."
fi

echo "--- [6/6] Environment Setup Complete! ---"
echo "To run your code, execute the following command:"
echo "source .env/bin/activate"
echo "python src/ncltpoles.py"