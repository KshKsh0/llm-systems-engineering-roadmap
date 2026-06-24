print("--- [DEBUG] 1. Importing libraries ---")
from unsloth import FastLanguageModel
import torch

print("--- [DEBUG] 2. Starting model load (Check your RAM usage!) ---")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length = 2048, 
    dtype = None, 
    load_in_4bit = True,
)

print("--- [DEBUG] 3. Model loaded successfully into VRAM! ---")

prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a strict data extraction assistant. Your job is to extract entities from the provided text and output them as a valid JSON object. 
Do not include any conversational text, introductory remarks, or explanations. Output ONLY the raw JSON object.
The output JSON must strictly follow this schema:
{{
  "patient_id": string or null,
  "equipment_used": array of strings,
  "findings": array of strings,
  "requires_followup": boolean
}}<|eot_id|><|start_header_id|>user<|end_header_id|>
Text to extract from:
"{}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

test_samples = [
    "Subject P-9042 was monitored using standard 10-20 system EEG configurations and a secondary digital pulse oximeter. Initial readings showed brief spike-and-wave discharges lasting 3 seconds in the temporal lobe, matching classic focal seizure patterns. The clinical team notes a follow-up consultation is scheduled for Tuesday morning.",
]

print("--- [DEBUG] 4. Switching to Inference Mode ---")
FastLanguageModel.for_inference(model)

print("--- [DEBUG] 5. Starting generation ---")
for i, sample in enumerate(test_samples, 1):
    print(f"--- [DEBUG] Processing Test Case {i} ---")
    formatted_prompt = prompt_template.format(sample)
    
    print("--- [DEBUG] Tokenizing... ---")
    inputs = tokenizer([formatted_prompt], return_tensors='pt').to('cuda')
    
    print("--- [DEBUG] Generating (GPU is working...) ---")
    outputs = model.generate(
        **inputs, 
        max_new_tokens=256, 
        use_cache=True,
        temperature=0.1,
    )
    
    print("--- [DEBUG] Decoding output ---")
    decoded_output = tokenizer.batch_decode(outputs)[0]
    assistant_response = decoded_output.split("<|start_header_id|>assistant<|end_header_id|>")[-1].replace("<|eot_id|>", "").strip()
    
    print(f"\n### TEST CASE {i} ###")
    print("Model Output:")
    print(assistant_response)
    print("-" * 50 + "\n")

print("--- [DEBUG] 6. Script finished successfully! ---")