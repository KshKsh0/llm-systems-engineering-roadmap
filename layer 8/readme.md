Markdown
# Layer 8: Inference Quantization Benchmarking

A clean, minimalist profiling harness engineered to evaluate the trade-offs between computational precision, memory consumption, and inference latency in Large Language Models. This subsystem isolates hardware metrics across `FP16`, `INT8`, and `INT4` tracking token generation velocity and memory thresholds.

---

## Performance Matrix

The architecture profiles runtime behavior and populates a high-contrast console matrix detailing the physical toll of quantization configurations:

| Precision | Memory Footprint | TTFT (Time to First Token) | TPOT (Time Per Output Token) | Throughput |
| :--- | :---: | :---: | :---: | :---: |
| **FP16** | Baseline VRAM | Low Latency Prefill | Optimal Compute | Max Tokens/sec |
| **INT8** | ~30% Reduction | Moderate Prefill | Quantization Overhead | Reduced Tokens/sec |
| **INT4** | ~45% Reduction | Ultra-Low Prefill | Balanced Compute | Accelerated Tokens/sec |

---

## Environment Initialization

To bypass common dependency conflicts, tensor dispatch issues, and missing Windows runtimes (such as the `fbgemm.dll` execution block), the environment must be built using explicit CUDA wheels and pinned library versions.

### 1. Provision the Environment

```bash
conda create -n quant_benchmark python=3.11 -y
conda activate quant_benchmark
```
# 2. Install Dependencies
Deploy the exact hardware-targeted stack by pointing pip directly to the PyTorch CUDA index:

```bash
pip install -r requirements.txt --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
Note: Ensure your underlying NVIDIA drivers are compatible with CUDA 12.4. For older local environments, substitute cu124 with cu121.
```
# requirements.txt

### Core Deep Learning Stack
torch==2.4.1

### Hugging Face Ecosystem
transformers==4.41.0
accelerate==1.1.1

### Quantization Backends
bitsandbytes==0.49.2

### UI Design & Formatting
rich==13.7.1

Run the master benchmarking routine to begin the automated profiling cycle. The console will display a real-time, live-updating reporting table without polluting your terminal history.

```bash
python quantization.py
```