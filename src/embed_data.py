import os
from sentence_transformers import SentenceTransformer
from vector_db import VectorDB
import glob

EMBED_MODEL = 'all-MiniLM-L6-v2'
DATA_DIR = '../data/'

if __name__ == '__main__':
    embedder = SentenceTransformer(EMBED_MODEL)
    vectordb = VectorDB(dim=384)
    files = glob.glob(os.path.join(DATA_DIR, '*.txt'))
    texts, metas, vectors = [], [], []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
            texts.append(text)
            metas.append({'text': text, 'source': file})
    vectors = embedder.encode(texts)
    vectordb.add(vectors, metas)
    print(f"Embedded {len(texts)} documents.")
