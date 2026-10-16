Retrieval for open-domain question answering over 131,921 OCR-scanned historical newspaper documents (ChroniclingAmericaQA), where OCR noise breaks standard keyword matching. Developed as a group project.

Approaches

Character n-gram fuzzy matching: documents and queries are tokenized into character 4- and 5-grams and indexed with BM25, making retrieval robust to OCR spelling errors. 5-grams significantly outperform 4-grams (MAP 0.80 vs 0.66 on a 100-query test). Weighted linear fusion and reciprocal rank fusion of the two are also compared.
SPLADE learned sparse retrieval: the full collection is encoded on GPU with naver/splade-cocondenser-ensembledistil and indexed as a Lucene impact index with Pyserini. Evaluated on 10,000 queries: MAP 0.71, nDCG@5 0.73, Recall@5 0.80.

Notebooks

- experiments.ipynb: the full pipeline. Covers OCR-aware spell correction (SymSpell + DistilBERT), spaCy entity extraction, query expansion anchored by fusion, and the ablation experiments. Set UPLOAD_VARIABLES = False to rebuild everything from the dataset.
- Historical_News_Question_Answering.ipynb: character n-gram fuzzy matching, comparing 4-grams and 5-grams with BM25 and linear vs reciprocal rank fusion.
- Splade_final.ipynb: SPLADE learned sparse retrieval (naver/splade-cocondenser-ensembledistil) with a Pyserini impact index. MAP 0.71 and nDCG@5 0.73 on 10,000 queries.

Tools: Python, PyTerrier, Pyserini, PyTorch, Hugging Face Transformers
