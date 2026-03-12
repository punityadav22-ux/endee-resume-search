from fastapi import FastAPI
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = np.load("data/embeddings.npy", allow_pickle=True)
files = np.load("data/files.npy", allow_pickle=True)


@app.get("/")
def home():
    return {"message": "AI Resume Search using Endee Vector DB"}


@app.get("/search")
def search(query: str):

    q = model.encode([query])

    scores = np.dot(embeddings, q.T)

    best = int(scores.argmax())

    return {
        "query": query,
        "best_match": str(files[best])
    }