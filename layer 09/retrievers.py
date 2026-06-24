import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


#simple tokenizer it can apply to different ones 

class BM25Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.tokenized_corpus = [chunk.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query, top_k=50):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        return np.argsort(scores)[::-1][:top_k].tolist()


# simply we take the embedding from a text 
class DenseRetriever:
    def __init__(self, chunks, model_name='all-MiniLM-L6-v2'):
        self.chunks = chunks
        self.embedder = SentenceTransformer(model_name)
        self.corpus_embeddings = self.embedder.encode(chunks, show_progress_bar=False)

    def search(self, query, top_k=50):
        query_embedding = self.embedder.encode([query], show_progress_bar=False)
        scores = np.dot(self.corpus_embeddings, query_embedding.T).flatten()
        return np.argsort(scores)[::-1][:top_k].tolist()