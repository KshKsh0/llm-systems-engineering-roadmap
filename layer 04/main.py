import json
import os
from src.models import LocalModelClient
from src.tools import ToolBox
from src.harness import ReasoningEvalHarness

def main():
    print("Initializing  Eval Harness...")
    
  
    os.makedirs("output", exist_ok=True)
    
   
    try:
        with open("data/test_cases.json", "r") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print("Error: data/dataset.json not found. Please create it first.")
        return

    client = LocalModelClient()
    tools = ToolBox()
    harness = ReasoningEvalHarness(client, tools)
    
    all_results = []

    print(f"Loaded {len(dataset)} tasks. Starting evaluation pipeline...\n")
    
    for idx, task in enumerate(dataset):
        print(f"Evaluating Task {idx + 1}/{len(dataset)}: {task['id']}")
        
        try:
            task_results = harness.evaluate_task(task)
            all_results.extend(task_results)
            print(f"  -> {task['id']} completed successfully.")
        except Exception as e:
            print(f"  -> Error evaluating {task['id']}: {e}")

    # Save artifact
    output_path = "output/eval_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)
        
    print(f"\nEvaluation Complete! Artifact saved to {output_path}")
    print("Review the JSON file to compare accuracy vs. latency tradeoffs.")

if __name__ == "__main__":
    main()