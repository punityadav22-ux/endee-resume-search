# AI Resume Semantic Search using Endee Vector Database

## Overview

This project demonstrates semantic search using vector embeddings and the Endee vector database.

The system stores resume text as embeddings and retrieves the most relevant resume based on search query.

## Features

- Resume embedding generation
- Vector similarity search
- Endee vector database integration
- Semantic search
- AI / NLP based retrieval

## Tech stack

Python
Sentence Transformers
Endee Vector DB
NumPy

## How it works

1. Convert resumes into embeddings
2. Store embeddings in vector database
3. Convert search query to embedding
4. Find closest vector
5. Return best resume

## Setup

pip install -r requirements.txt

python embed.py

python search.py

## Endee Usage

This project uses the Endee vector database as the backend for similarity search.