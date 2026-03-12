import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = np.load("data/embeddings.npy", allow_pickle=True)
files = np.load("data/files.npy", allow_pickle=True)

query = input("Search: ")

q = model.encode([query])

scores = np.dot(embeddings, q.T)

best = scores.argmax()

print("Best match:", files[best])