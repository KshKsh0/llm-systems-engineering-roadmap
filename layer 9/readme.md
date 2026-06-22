# Layer 9: RAG Systems — Core Mechanics

## Objective
Build retrieval systems that ground LLM outputs in external knowledge.

## Core Pipeline Architecture
A production-grade RAG pipeline consists of 12 sequential phases:
1. Parsing: Extracting raw text from diverse formats (PDFs, Markdown, Word, HTML, JSON)[cite: 1].
2. Cleaning: Stripping boilerplate code, headers, footers, and formatting noise[cite: 1].
3. Chunking: Splitting raw continuous text into discrete segments[cite: 1].
4. Embedding: Transforming text chunks into high-dimensional dense vectors via an embedding model[cite: 1].
5. Indexing: Storing vectors and token indices in a database for lookups[cite: 1].
6. Retrieval: Querying the database to fetch matching document chunks[cite: 1].
7. Reranking: Re-scoring the first-pass retrieved pool with a cross-encoder[cite: 1].
8. Prompt Construction: Injecting the final chunks cleanly into the LLM context window[cite: 1].
9. Generation: Processing the engineered prompt to output a response[cite: 1].
10. Citation Validation: Programmatically cross-referencing output text against the actual source context[cite: 1].
11. Evaluation: Calculating system-level metrics (e.g., context recall, faithfulness)[cite: 1].

## Key Mechanisms to Understand Deeply

### Chunking Strategies
* **Fixed-size chunks:** Split text by token/character count with a sliding window[cite: 1].
* **Semantic chunks:** Split data based on shifts in semantic vector similarity[cite: 1].
* **Section-based chunks:** Respect layout elements like headers and tables[cite: 1].
* **Parent-child chunks:** Embed precise small child slices for retrieval, but return the broader parent chunk to the LLM for generation context[cite: 1].

### Hybrid Retrieval & Reciprocal Rank Fusion (RRF)
Production architectures combine keyword matching (BM25) with semantic matching (Dense retrieval)[cite: 1]. To merge these ranked lists without scale mismatches, apply RRF[cite: 1]:

$$RRF\_Score(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

### Reranking
A cross-encoder evaluates query-document pairs simultaneously to compute a high-precision relevance score[cite: 1]. This filters down a wide retrieval pool (e.g., top 50) to the absolute best context chunks (top 5–10)[cite: 1].

## Systematic Failure Modes
* **Parsing/Chunking failures:** Fragmented answers or missing context due to bad layout boundaries[cite: 1].
* **Retrieval failures:** The database fetches irrelevant noise or misses the target chunk completely[cite: 1].
* **Generation failures:** The LLM receives the correct context but hallucinates or ignores the data[cite: 1].

## Evaluation Gate
You pass this layer if you can programmatically isolate a retrieval failure from a generation failure using full trace visibility[cite: 1].