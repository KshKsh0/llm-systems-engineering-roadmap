import time
import json
from pipeline import ModularRAGPipeline
from chunking import TextChunker

# ==========================================
# 1. THE GOLDEN DATASET
# ==========================================
# In production, this would be a JSON/CSV file with hundreds of rows covering
# edge cases, hard questions, and negative/out-of-scope questions.
GOLDEN_DATASET = [
    {
        "id": "q1",
        "query": "What is the standard severance package for a 5-year employee?",
        "expected_facts": ["two weeks per year of service", "COBRA coverage for 3 months"],
        "expected_chunk_id": 2  # Corrected to match index 2 in dummy_documents
    },
    {
        "id": "q2",
        "query": "What is the error code ERR_DB_0x992 mean?",
        "expected_facts": ["database connection timeout", "port 5432 blocked"],
        "expected_chunk_id": 1  # Corrected to match index 1 in dummy_documents
    }
]

# ==========================================
# 2. THE EVALUATOR ENGINE
# ==========================================
class RAGEvaluator:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.results = []

    def evaluate_context_recall(self, trace, expected_chunk_id):
        """Did the retriever actually find the target document in its final pool?"""
        if expected_chunk_id in trace["final_context_indices"]:
            return 1.0  # Perfect recall
        elif expected_chunk_id in trace["fused_indices"]:
            return 0.5  # Found by base retrievers, but reranker dropped it (Reranker tuning needed)
        else:
            return 0.0  # Completely missed (Embedding/BM25 tuning needed)

    def evaluate_context_precision(self, trace, expected_chunk_id):
        """Is the most relevant document at the very top of the context window?"""
        try:
            rank = trace["final_context_indices"].index(expected_chunk_id)
            # MRR (Mean Reciprocal Rank) logic: 1st = 1.0, 2nd = 0.5, 3rd = 0.33
            return 1.0 / (rank + 1)
        except ValueError:
            return 0.0

    def evaluate_faithfulness(self, trace, expected_facts):
        """
        In a real system, you use an LLM-as-a-judge here.
        For this programmatic dashboard, we simulate it by checking if the 
        expected facts actually exist in the final retrieved text blocks.
        """
        retrieved_text = " ".join([self.pipeline.chunks[idx].lower() for idx in trace["final_context_indices"]])
        
        facts_found = 0
        for fact in expected_facts:
            if fact.lower() in retrieved_text:
                facts_found += 1
                
        return facts_found / len(expected_facts) if expected_facts else 0.0

    def run_eval_suite(self, dataset):
        print("Starting RAG Evaluation Suite...\n")
        total_latency = 0
        
        for item in dataset:
            start_time = time.time()
            
           
            trace = self.pipeline.run(item["query"])
            
            latency = time.time() - start_time
            total_latency += latency
            
            
            recall = self.evaluate_context_recall(trace, item["expected_chunk_id"])
            precision = self.evaluate_context_precision(trace, item["expected_chunk_id"])
            faithfulness = self.evaluate_faithfulness(trace, item["expected_facts"])
            
            # 3. Log results
            self.results.append({
                "query": item["query"],
                "latency_ms": round(latency * 1000, 2),
                "recall": recall,
                "precision": precision,
                "faithfulness": faithfulness,
                "trace": trace
            })
            
            print(f"✅ Evaluated: '{item['query']}'")
            
        self.generate_report()

    def generate_report(self):
        """Outputs a console dashboard of the system's health."""
        avg_recall = sum(r["recall"] for r in self.results) / len(self.results)
        avg_precision = sum(r["precision"] for r in self.results) / len(self.results)
        avg_faithfulness = sum(r["faithfulness"] for r in self.results) / len(self.results)
        avg_latency = sum(r["latency_ms"] for r in self.results) / len(self.results)

        print("\n" + "="*50)
        print("📊 RAG SYSTEM PRODUCTION DASHBOARD")
        print("="*50)
        print(f"Total Queries Evaluated : {len(self.results)}")
        print(f"Average Latency         : {avg_latency:.2f} ms per query")
        print("-" * 50)
        print(f"Context Recall          : {avg_recall:.2f}  (Are we finding the documents?)")
        print(f"Context Precision       : {avg_precision:.2f}  (Are they ranked at the top?)")
        print(f"Faithfulness            : {avg_faithfulness:.2f}  (Is the context sufficient?)")
        print("="*50)

        # Flag specific failures for the developer to debug
        for res in self.results:
            if res["recall"] < 1.0 or res["faithfulness"] < 1.0:
                print(f"\n⚠️ WARNING on Query: '{res['query']}'")
                if res["recall"] == 0.0:
                    print("  Root Cause: Retrieval Failure. The chunk was not found by BM25 or Dense search.")
                elif res["recall"] == 0.5:
                    print("  Root Cause: Reranking Failure. The cross-encoder dropped the correct chunk.")
                elif res["faithfulness"] < 1.0:
                    print("  Root Cause: Missing Facts. The chunks retrieved do not contain the complete answer.")

# ==========================================
# 3. EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # In a real environment, you would load your actual documents here
    dummy_documents = [
        "We do not offer remote work.",
        "To fix a database connection timeout, ensure port 5432 is not blocked. This resolves ERR_DB_0x992.",
        "The standard severance package for a 5-year employee is two weeks per year of service, plus COBRA coverage for 3 months."
    ]
    
    # Initialize your modular system
    chunker = TextChunker(chunk_size=50)
    pipeline = ModularRAGPipeline(dummy_documents, chunker)
    
    # Run the dashboard
    dashboard = RAGEvaluator(pipeline)
    dashboard.run_eval_suite(GOLDEN_DATASET)