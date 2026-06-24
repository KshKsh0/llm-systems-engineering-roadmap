import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="LLM KV Cache Memory Calculator",
    page_icon="💾",
    layout="centered"
)

st.title("💾 LLM KV Cache Memory Calculator")
st.markdown(
    "Calculate the real, physical VRAM footprint of your LLM workload's **KV Cache** "
    "before deploying to production nodes."
)
st.markdown("---")

# 2. Sidebar / Inputs Layout
st.header("🔧 Model Configuration")

col1, col2 = st.columns(2)

with col1:
    layers = st.number_input("Number of Layers ($l$)", min_value=1, value=32, help="Total transformer blocks/layers.")
    kv_heads = st.number_input("Number of KV Heads ($h_{kv}$)", min_value=1, value=8, help="Number of Key/Value attention heads (use lower values for GQA/MQA models).")
    head_dim = st.number_input("Head Dimension ($d$)", min_value=1, value=128, help="The dimension size of each individual attention head.")

with col2:
    precision = st.selectbox(
        "Precision Format ($p$)",
        options=["FP16 / BF16 (2 bytes)", "FP8 / INT8 (1 byte)", "FP32 (4 bytes)"],
        index=0
    )
    
    # Map selection to actual byte values
    precision_bytes = 2
    if "1 byte" in precision:
        precision_bytes = 1
    elif "4 bytes" in precision:
        precision_bytes = 4

    batch_size = st.number_input("Concurrent Batch Size ($b$)", min_value=1, value=10, help="Number of users/requests processing simultaneously.")
    context_length = st.number_input("Context / Sequence Length ($s$)", min_value=1, value=128000, step=1000, help="Total sequence length (Prompt + Expected Output tokens).")

st.markdown("---")

# 3. Core Logic Calculation
# Formula: 2 * batch * seq_len * layers * kv_heads * head_dim * bytes
total_bytes = 2 * batch_size * context_length * layers * kv_heads * head_dim * precision_bytes
megabytes = total_bytes / (1024 ** 2)
gigabytes = total_bytes / (1024 ** 3)

# 4. Results Display Panel
st.header("📊 Results")

# Highlighting critical numbers
if gigabytes > 80:
    st.error(f"🚨 **Extreme Risk:** The KV cache consumes **{gigabytes:,.2f} GB** of VRAM. This alone exceeds a single standard enterprise GPU (80GB A100/H100).")
elif gigabytes > 24:
    st.warning(f"⚠️ **High Memory Pressure:** The KV cache consumes **{gigabytes:,.2f} GB** of VRAM. Ensure your cluster architecture leverages quantized KV caches or continuous batching engine adjustments.")
else:
    st.success(f"✅ **Safe Range:** The KV cache footprint is **{gigabytes:,.2f} GB** of VRAM.")

# Metrics layout
m1, m2 = st.columns(2)
m1.metric(label="Total VRAM (Gigabytes)", value=f"{gigabytes:,.2f} GB")
m2.metric(label="Total VRAM (Megabytes)", value=f"{megabytes:,.2f} MB")

# Quick breakdown math display
st.markdown("### 🧮 Formula Breakdown")
st.code(
    f"2 × {batch_size} (batch) × {context_length:,} (seq_len) × {layers} (layers) "
    f"× {kv_heads} (kv_heads) × {head_dim} (head_dim) × {precision_bytes} (bytes) \n"
    f"= {total_bytes:,} Raw Bytes"
)