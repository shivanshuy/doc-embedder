# doc-embedder  
`doc-embedder` is a Python project for scraping, chunking, embedding, and storing documents from web sources (such as blogs) into a vector database using LangChain and ChromaDB. The main use case is semantic search and retrieval of blog content, with a focus on the Chip Huyen blog. 

## Features
- Scrapes web pages and extracts internal links.

- Chunks document text for efficient embedding.

- Embeds text chunks using LangChain-compatible models.

- Stores embeddings in a ChromaDB collection.

- Maintains an index document summarizing all processed URLs.
  

## Installation

1.  **Clone the repository**

2.  **Install dependencies** (requires Python 3.12–3.14):
```
    - python -m pip install poetry
    - poetry install
   ```  

## Usage
Run the main embedder script to process and embed documents:
```
python chip_huyen_blog_embedder.py
``` 
This will: 

- Scrape and chunk content from the Chip Huyen blog.
- Embed and store chunks in the ChromaDB database at chip_huyen_db.
- Create an index document summarizing all processed URLs.

## Configuration
The database path and collection name are set in doc_embedder/web_embedder/chip_huyen_blog_embedder.py.
Modify BASE_DOMAIN and BASE_URL in the script to target different sources.

## Dependencies
- LangChain
- ChromaDB
- Pydantic
- Unstructured
- BeautifulSoup (bs4)
- Requests

See pyproject.toml for full details.  

## License
MIT  

## Author
shivanshuy (shivanshuy@gmail.com) 