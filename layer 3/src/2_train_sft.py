import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

print("--- [DEBUG] 1. Initializing Environment ---", flush=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)

model_id = "unsloth/llama-3-8b-Instruct-bnb-4bit"

print("--- [DEBUG] 2. Loading Tokenizer & Model (Watch RAM!) ---", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token 

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": 0}, 
    low_cpu_mem_usage=True 
)

print("--- [DEBUG] 3. Configuring LoRA (Standard HF) ---", flush=True)
# This enables the memory-saving gradients native to PyTorch
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

print("--- [DEBUG] 4. Preparing Training Data ---", flush=True)
# We show the model EXACTLY what perfect outputs look like
train_data = [
    {
        "input": "Patient John Doe (ID: 11223) underwent a standard EKG. Results showed normal sinus rhythm. No further action needed.",
        "output": '{\n  "patient_id": "11223",\n  "equipment_used": [\n    "EKG"\n  ],\n  "findings": [\n    "normal sinus rhythm"\n  ],\n  "requires_followup": false\n}'
    },
    {
        "input": "Quick check in the ER. Hooked up the pulse oximeter and noticed slightly low oxygen levels, but nothing critical. Send them to the ward for observation.",
        "output": '{\n  "patient_id": null,\n  "equipment_used": [\n    "pulse oximeter"\n  ],\n  "findings": [\n    "slightly low oxygen levels"\n  ],\n  "requires_followup": true\n}'
    }
]


train_data = train_data * 15 
dataset = Dataset.from_list(train_data)

prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a strict data extraction assistant. Output ONLY the raw JSON object.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Text to extract from:
"{}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{}<|eot_id|>"""

def formatting_prompts_func(examples):
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for text_input, text_output in zip(inputs, outputs):
        texts.append(prompt_template.format(text_input, text_output))
    return { "text" : texts }

formatted_dataset = dataset.map(formatting_prompts_func, batched=True)

print("--- [DEBUG] 5. Starting SFT Training ---", flush=True)
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=formatted_dataset,
    args=SFTConfig(
        dataset_text_field="text",  # <--- Moved inside SFTConfig!
        max_length=512,         # <--- Moved inside SFTConfig!
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=30,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="paged_adamw_8bit",
        output_dir="hf_outputs",
    ),
)

trainer.train()
print("--- [DEBUG] 6. Saving Adapter ---", flush=True)
model.save_pretrained("hf_sft_adapter")
tokenizer.save_pretrained("hf_sft_adapter")
print("Done! Standard HF Adapter saved to 'hf_sft_adapter' folder.", flush=True)