import numpy as np
from vector_db import VectorDB

# Example: create a vector DB, add vectors, and search
if __name__ == '__main__':
    db = VectorDB(dim=3, db_path='embeddings/demo.index', meta_path='embeddings/demo_meta.pkl')
    # Add some example vectors (3D for demo)
    vectors = [np.array([1, 2, 3]), np.array([2, 3, 4]), np.array([10, 10, 10])]
    metas = [
        {'text': 'Vector A', 'info': 'First'},
        {'text': 'Vector B', 'info': 'Second'},
        {'text': 'Vector C', 'info': 'Third'}
    ]
    db.add(vectors, metas)
    # Search for nearest to [1,2,2]
    query = np.array([1, 2, 2])
    results = db.search(query, k=2)
    print('Search results:')
    for meta, dist in results:
        print(f"{meta['text']} (distance: {dist:.2f})")
