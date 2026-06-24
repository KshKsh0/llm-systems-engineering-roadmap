import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

print("--- [DEBUG] 1. Loading Base Model ---", flush=True)

# 1. Standard 4-bit Quantization Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)

base_model_id = "unsloth/llama-3-8b-Instruct-bnb-4bit"
adapter_path = "hf_sft_adapter" # The folder you just saved!

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token 

# Load the base model directly to the GPU
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map={"": 0}, 
    low_cpu_mem_usage=True 
)

print(f"--- [DEBUG] 2. Attaching Your Custom LoRA Adapter from '{adapter_path}' ---", flush=True)
# This merges your newly learned domain expertise into the model
model = PeftModel.from_pretrained(base_model, adapter_path)

print("--- [DEBUG] 3. Model Ready! Starting Final Eval... ---", flush=True)

prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a strict data extraction assistant. Output ONLY the raw JSON object.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Text to extract from:
"{}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

test_samples = [
    "Subject P-9042 was monitored using standard 10-20 system EEG configurations and a secondary digital pulse oximeter. Initial readings showed brief spike-and-wave discharges lasting 3 seconds in the temporal lobe, matching classic focal seizure patterns. The clinical team notes a follow-up consultation is scheduled for Tuesday morning.",
    "Routine diagnostic check performed on reference unit. Signals are entirely clean with normal alpha rhythms observed across all occipital leads. No anomalous discharges or abnormalities detected. Patient discharged home immediately.",
    "Emergency intake under ID code E-441. The patient was initially hooked up to a portable telemetry monitor, but it malfunctioned, so they switched to a standard bed-side ECG unit. Flashing light triggers caused transient slowing, but nothing definitive for a seizure diagnosis. Dr. Vance decided to hold off on scheduling an advanced follow-up scan until the lab results return.",
]

for i, sample in enumerate(test_samples, 1):
    print(f"--- [DEBUG] Processing Test Case {i} ---", flush=True)
    

    formatted_prompt = prompt_template.format(sample)
    
    inputs = tokenizer([formatted_prompt], return_tensors='pt').to('cuda')
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=256, 
        use_cache=True,
        temperature=0.1,  
    )
    
    decoded_output = tokenizer.batch_decode(outputs)[0]
    assistant_response = decoded_output.split("<|start_header_id|>assistant<|end_header_id|>")[-1].replace("<|eot_id|>", "").strip()
    
    print(f"\n### FINAL SFT OUTPUT: TEST CASE {i} ###", flush=True)
    print("Model Output:", flush=True)
    print(assistant_response, flush=True)
    print("-" * 50 + "\n", flush=True)