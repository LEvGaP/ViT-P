import torch
import xformers

print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch CUDA version: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"xFormers version: {xformers.__version__}")

# Test if a core xformers operation is available
from xformers.ops import fmha
print("xFormers attention module loaded successfully.")