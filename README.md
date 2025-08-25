# 1. Install the python 3.8 and the virtual environment
## A. Install the dependent pkg
```sudo apt update```
```
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev curl llvm libncurses5-dev \
  libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev \
  libgdbm-dev libdb-dev libexpat1-dev libmpdec-dev libgmp-dev
```

```git clone https://github.com/pyenv/pyenv.git ~/.pyenv```

## B. Add the following environment setup in the ~/.bashrc
```export PYENV_ROOT="$HOME/.pyenv"```

```export PATH="$PYENV_ROOT/bin:$PATH"```

```eval "$(pyenv init --path)"```

```eval "$(pyenv init -)"```

## C. Check and install the available pyenv version
```pyenv install --list | grep -E "^  3\.8\.[0-9]+$"```

```pyenv install 3.8.20```

```pyenv versions```

## D. Change the python version
```pyenv global 3.8.20```

```python --version```

## E. Create the venv
```python -m venv .env```

```source .env/bin/activate```

# 2. Install the requirement package
```pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124```

```pip install -r requirements.txt```

# 3. Run the nclt localization
```python src/ncltpoles.py```