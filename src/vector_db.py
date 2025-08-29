import faiss
import numpy as np
import os
import pickle

class VectorDB:
    def __init__(self, dim, db_path='embeddings/vector.index', meta_path='embeddings/meta.pkl'):
        self.dim = dim
        self.db_path = db_path
        self.meta_path = meta_path
        self.index = faiss.IndexFlatL2(dim)
        self.meta = []
        if os.path.exists(db_path):
            self.index = faiss.read_index(db_path)
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                self.meta = pickle.load(f)

    def add(self, vectors, metas):
        self.index.add(np.array(vectors).astype('float32'))
        self.meta.extend(metas)
        self.save()

    def search(self, vector, k=5):
        D, I = self.index.search(np.array([vector]).astype('float32'), k)
        return [(self.meta[i], D[0][idx]) for idx, i in enumerate(I[0]) if i < len(self.meta)]

    def save(self):
        faiss.write_index(self.index, self.db_path)
        with open(self.meta_path, 'wb') as f:
            pickle.dump(self.meta, f)
