import numpy as np

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def l2_distance(a, b):
    a, b = np.array(a), np.array(b)
    return np.linalg.norm(a - b)

def dot_product(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b)
