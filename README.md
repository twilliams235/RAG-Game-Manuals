# Game Rulebook RAG (100% Local)
## Catan and Codenames currently

A fully on-device end-to-end RAG pipeline for querying board game instruction manuals
stored as PDFs in game_manuals. Uses Ollama for local LLM + embeddings and Chroma for
the vector database. No cloud services required. Database set up where you can
dynamically add or remove new instruction manuals.

### Features

- PDF Ingestion from game_manuals via PyPDFDirectoryLoader

- Chunking with RecursiveCharacterTextSplitter

- Indexing into Chroma with stable Chunk IDs

- Local embeddings with Ollama (nomic-embed-text)

- Deterministic pytest for simple performance checks

### Repo Structure

- game_manuals/      --> PDF rulebooks

- set_database.py    --> Ingests & indexes PDFs into Chroma

- query.py           --> Runs retrieval + LLM answer generation

- get_embedding.py   --> Configures Ollama embeddings (nomic-embed-text)

- requirements.txt   --> Python dependencies

- test_rag.py        --> Simple tests / examples

### Models (local)

```ollama pull nomic-embed-text```

```ollama pull llama2```

### Setup

```python -m venv venv```

```source venv/bin/activate```

```pip install -r requirements.txt```

### Build Vector DB

```python set_database.py```

### Query

```python query.py "How many words does the starting team guess in codenames?"```

### Testing

```python -m pytest```