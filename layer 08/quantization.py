import time
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from threading import Thread

# --- New Imports for UI ---
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

console = Console()

class QuantizationBenchmark:
    def __init__(self, base_model_id: str):
        self.base_model_id = base_model_id
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.device == "cpu":
            console.print("[bold red]WARNING:[/bold red] CUDA not available. VRAM measurements will fail.")

    def clear_vram(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def load_model(self, quant_type: str, custom_model_id: str = None):
        self.clear_vram()
        model_id = custom_model_id or self.base_model_id
        
        start_load = time.time()
        
        if quant_type == "fp16":
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        elif quant_type == "int8":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
        elif quant_type == "int4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
            model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
        else:
            raise ValueError("Unsupported quantization type.")

        load_time = time.time() - start_load
        
        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
            console.print(f"   ↳ [dim]Loaded in {load_time:.2f}s | Resting VRAM: {vram_mb:.1f} MB[/dim]")
            
        return model

    def benchmark_inference(self, model, prompt: str, max_new_tokens=100):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        
        start_time = time.time()
        thread.start()
        
        first_token_time = None
        generated_text = ""
        token_count = 0
        
        for text in streamer:
            if first_token_time is None:
                first_token_time = time.time()
                ttft = first_token_time - start_time
            generated_text += text
            token_count += 1
            
        thread.join()
        end_time = time.time()
        
        total_time = end_time - start_time
        tpot = (end_time - first_token_time) / token_count if token_count > 0 else 0
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0

        metrics = {
            "TTFT (s)": round(ttft, 4),
            "TPOT (s)": round(tpot, 4),
            "Throughput (tok/s)": round(token_count / total_time, 2),
            "Peak VRAM": round(peak_vram, 2)
        }
        return generated_text, metrics

    def run_eval(self, model, eval_dataset):
        correct = 0
        total = len(eval_dataset)
        
        for item in eval_dataset:
            inputs = self.tokenizer(item["prompt"], return_tensors="pt").to(self.device)
            outputs = model.generate(**inputs, max_new_tokens=item["max_tokens"], do_sample=False)
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            if item["expected_output"].strip().lower() in response.strip().lower():
                correct += 1
                
        return (correct / total) * 100

def print_summary_table(results):
    table = Table(title="Quantization Performance Benchmark", show_header=True, header_style="bold white")
    
    table.add_column("Format", style="cyan", justify="center")
    table.add_column("Peak VRAM (MB)", justify="right")
    table.add_column("TTFT (s)", justify="right")
    table.add_column("TPOT (s)", justify="right")
    table.add_column("Throughput (tok/s)", justify="right")
    table.add_column("Eval Accuracy", justify="right")

    for config, data in results.items():
        m = data['metrics']
        table.add_row(
            config,
            f"{m['Peak VRAM']:.1f}",
            f"{m['TTFT (s)']:.4f}",
            f"{m['TPOT (s)']:.4f}",
            f"{m['Throughput (tok/s)']:.1f}",
            f"{data['accuracy']:.1f}%"
        )
        
    console.print("\n")
    console.print(table)

if __name__ == "__main__":
    BASE_MODEL = "Qwen/Qwen1.5-1.8B" 
    
    EVAL_DATASET = [
        {"prompt": "Extract the capital of France in JSON format: {'capital': '...'}", "expected_output": '{"capital": "paris"}', "max_tokens": 15},
        {"prompt": "Calculate 15 * 3. Just provide the final number.", "expected_output": "45", "max_tokens": 5}
    ]
    
    console.print(Panel.fit(f"[bold]LLM Systems Inference Benchmark[/bold]\nTarget: {BASE_MODEL}", border_style="cyan"))
    
    benchmark = QuantizationBenchmark(base_model_id=BASE_MODEL)
    results = {}
    formats_to_test = ["fp16", "int8", "int4"]

    for quant in formats_to_test:
        with console.status(f"[bold green]Benchmarking {quant.upper()}...", spinner="dots"):
            model = benchmark.load_model(quant_type=quant)
            _, metrics = benchmark.benchmark_inference(model, prompt="Explain the difference between throughput and latency.")
            accuracy = benchmark.run_eval(model, EVAL_DATASET)
            
            results[quant.upper()] = {'metrics': metrics, 'accuracy': accuracy}
            del model
            benchmark.clear_vram()

    print_summary_table(results)