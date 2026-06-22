from retrievers import BM25Retriever, DenseRetriever
from reranker import DocumentReranker


def reciprocal_rank_fusion(list_a, list_b, k=60):
    """
    Merges two ranked lists of document IDs using Reciprocal Rank Fusion.
    """
    rrf_scores = {}
    
    for rank, doc_id in enumerate(list_a):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))
        
    for rank, doc_id in enumerate(list_b):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))
    
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return just the ordered IDs
    return [doc_id for doc_id, score in sorted_docs]

class ModularRAGPipeline:
    def __init__(self, raw_documents, chunker):
        # 1. Process Data
        self.chunks = []
        for doc in raw_documents:
            self.chunks.extend(chunker.chunk_by_paragraphs(doc))
            
        # 2. Initialize Mechanisms
        self.bm25_retriever = BM25Retriever(self.chunks)
        self.dense_retriever = DenseRetriever(self.chunks)
        self.reranker = DocumentReranker()

    def run(self, query):
        trace = {"query": query}
        
        # Step A: Dual-Path Retrieval
        trace["bm25_raw"] = self.bm25_retriever.search(query, top_k=30)
        trace["dense_raw"] = self.dense_retriever.search(query, top_k=30)
        
        # Step B: Fusion
        trace["fused_indices"] = reciprocal_rank_fusion(
            trace["bm25_raw"], 
            trace["dense_raw"]
        )
        
        # Step C: Reranking
        trace["final_context_indices"] = self.reranker.rerank(
            query, 
            trace["fused_indices"], 
            self.chunks, 
            top_k=5
        )
        
        # Step D: Prompt & Citation Construction
        context_blocks = []
        for idx in trace["final_context_indices"]:
            context_blocks.append(f"[Chunk_ID: {idx}]\n{self.chunks[idx]}")
            
        trace["prompt"] = (
            "Answer the query using ONLY the provided context blocks. "
            "Append the [Chunk_ID: X] to the end of any sentence containing facts from that chunk.\n\n"
            f"Context:\n{chr(10).join(context_blocks)}\n\n"
            f"Query: {query}\n"
        )
        
        return trace