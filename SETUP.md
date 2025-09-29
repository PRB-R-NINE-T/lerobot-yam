cd gello
uv pip install -e .
uv pip install -r requirements.txt

cd i2rt
uv pip install -e .

uv pip install -e .

# Fix USB permissions (requires logout/login to take effect)
sudo usermod -a -G dialout $USER

# Set PYTHONPATH for gello module access
export PYTHONPATH=/home/p/Desktop/lerobot-yam:$PYTHONPATH