from sentence_transformers import CrossEncoder

class DocumentReranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.reranker = CrossEncoder(model_name)

    def rerank(self, query, candidate_ids, original_chunks, top_k=5):
        if not candidate_ids:
            return []
            
        # Create pairs of [Query, Document] for the Cross-Encoder
        pairs = [[query, original_chunks[idx]] for idx in candidate_ids]
        
        # Predict relevance scores
        scores = self.reranker.predict(pairs)
        
        # Zip IDs with scores and sort them
        scored_candidates = list(zip(candidate_ids, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return the top K document IDs
        return [idx for idx, score in scored_candidates[:top_k]]