import sys
import os

# Sometimes BitsAndBytes needs to be explicitly told which CUDA version you have on Windows
os.environ["BNB_CUDA_VERSION"] = "121" 

print("--- STARTING GRANULAR IMPORT TRAP ---", flush=True)

try:
    print("1. Loading PyTorch...", flush=True)
    import torch
    print(f"   Success! (Version: {torch.__version__})", flush=True)

    print("2. Loading Triton (Windows Port)...", flush=True)
    import triton
    print("   Success!", flush=True)

    print("3. Loading xformers...", flush=True)
    import xformers
    print("   Success!", flush=True)

    print("4. Loading BitsAndBytes...", flush=True)
    import bitsandbytes as bnb
    print("   Success!", flush=True)

    print("5. Loading torchao...", flush=True)
    import torchao
    print("   Success!", flush=True)

    print("6. Loading Transformers...", flush=True)
    import transformers
    print("   Success!", flush=True)

    print("7. Loading Unsloth...", flush=True)
    from unsloth import FastLanguageModel
    print("   Success!", flush=True)

    print("--- ALL IMPORTS PASSED SAFELY ---", flush=True)

except Exception as e:
    print(f"\n[PYTHON ERROR CAUGHT]: {e}", flush=True)