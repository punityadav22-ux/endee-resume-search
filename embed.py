from sentence_transformers import SentenceTransformer
import os
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = []
files = []

for file in os.listdir("resumes"):
    with open("resumes/" + file, "r") as f:
        texts.append(f.read())
        files.append(file)

embeddings = model.encode(texts)

np.save("data/embeddings.npy", embeddings)
np.save("data/files.npy", files)

print("Embeddings saved")