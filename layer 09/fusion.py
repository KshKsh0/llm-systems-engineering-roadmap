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