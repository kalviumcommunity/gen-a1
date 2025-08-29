import faiss
import numpy as np
import os
import pickle

class VectorDB:
    """
    A simple vector database using FAISS for similarity search.
    Stores vectors and associated metadata, supports add, search, and persistence.
    """
    def __init__(self, dim, db_path='embeddings/vector.index', meta_path='embeddings/meta.pkl'):
        """
        Initialize the vector database.
        Args:
            dim (int): Dimension of the vectors.
            db_path (str): Path to store the FAISS index.
            meta_path (str): Path to store the metadata.
        """
        self.dim = dim
        self.db_path = db_path
        self.meta_path = meta_path
        self.index = faiss.IndexFlatL2(dim)  # L2 (Euclidean) distance index
        self.meta = []  # List of metadata dicts
        # Load existing index and metadata if available
        if os.path.exists(db_path):
            self.index = faiss.read_index(db_path)
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                self.meta = pickle.load(f)

    def add(self, vectors, metas):
        """
        Add vectors and their metadata to the database.
        Args:
            vectors (list or np.ndarray): List of vectors to add.
            metas (list): List of metadata dicts for each vector.
        """
        self.index.add(np.array(vectors).astype('float32'))
        self.meta.extend(metas)
        self.save()

    def search(self, vector, k=5):
        """
        Search for the k most similar vectors to the input vector.
        Args:
            vector (np.ndarray): Query vector.
            k (int): Number of results to return.
        Returns:
            List of (metadata, distance) tuples.
        """
        D, I = self.index.search(np.array([vector]).astype('float32'), k)
        return [(self.meta[i], D[0][idx]) for idx, i in enumerate(I[0]) if i < len(self.meta)]

    def save(self):
        """
        Save the FAISS index and metadata to disk.
        """
        faiss.write_index(self.index, self.db_path)
        with open(self.meta_path, 'wb') as f:
            pickle.dump(self.meta, f)
