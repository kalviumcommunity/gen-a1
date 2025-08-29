import os
from sentence_transformers import SentenceTransformer
from .vector_db import VectorDB
from .llm_call import call_gemini
import logging
import time

EMBED_MODEL = 'all-MiniLM-L6-v2'

class RAGPipeline:
    def __init__(self, db_path='embeddings/vector.index', meta_path='embeddings/meta.pkl'):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.vectordb = VectorDB(dim=384, db_path=db_path, meta_path=meta_path)
        logging.basicConfig(filename='logs/tokens.log', level=logging.INFO)

    def embed(self, text):
        return self.embedder.encode([text])[0]

    def retrieve(self, query, k=3):
        qvec = self.embed(query)
        return self.vectordb.search(qvec, k=k)

    def run(self, user_query, prompt_template, temperature=0.7, top_p=0.95, top_k=40, stop=None):
        start = time.time()
        docs = self.retrieve(user_query)
        context = '\n'.join([doc[0]['text'] for doc in docs])
        prompt = prompt_template.format(context=context, query=user_query)
        response = call_gemini(prompt, temperature, top_p, top_k, stop)
        tokens_used = len(prompt.split()) + len(response.split())
        logging.info(f"Tokens: {tokens_used}, Query: {user_query}, Time: {time.time()-start:.2f}s")
        return response
