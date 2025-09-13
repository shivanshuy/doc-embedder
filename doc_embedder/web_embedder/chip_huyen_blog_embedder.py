import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
import chromadb
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.callbacks import (
    CallbackManager,
    StdOutCallbackHandler,
)
from langchain_core.documents import Document
import logging

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


callback_manager = CallbackManager([StdOutCallbackHandler()])


BASE_DOMAIN = "https://huyenchip.com"
BASE_URL = BASE_DOMAIN + "/blog/"

text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)


def scrape_page(url):
    """Scrapes a page using SeleniumURLLoader, extracts links, and follows them recursively."""
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()

        page_text = docs[0].page_content

        usable_page_text = page_text[
            : page_text.index(
                "Please enable JavaScript to view the comments powered by Disqus"
            )
        ].strip()
        docs[0].page_content = usable_page_text
        # print(f"Loaded {usable_page_text} documents from Loader for {url}")

        return docs

    except Exception as e:
        logger.error(f" Error scraping {url}: {e}")
        return None


def extract_links(BASE_DOMAIN, url):
    """Extract all valid internal links from the given page."""
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")
        links = set()
        for a_tag in soup.find_all("a", href=True):
            sub_url = a_tag["href"]
            # Ignore navigation links (e.g., #section1)
            if "#" in sub_url:
                continue
            # Convert relative URLs to absolute URLs
            if sub_url.startswith("/"):
                sub_url = BASE_DOMAIN + sub_url
            # Only allow links within Huyen Chip
            if sub_url.startswith(BASE_DOMAIN):  # and sub_url not in visited_urls:
                links.add(sub_url)
        return links
    except Exception as e:
        logger.warning(f"Error extracting links from {url}: {e}")
        return None


def create_chunks(document):
    return text_splitter.split_documents(document)


def get_client(db_name):
    client = chromadb.PersistentClient(path=db_name)
    return client


def get_collection(client, collection_name):
    # Create embedding function
    embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L12-v2"
    )

    return client.get_or_create_collection(
        collection_name, embedding_function=embed_func
    )


def add_documents_to_collection(collection, url_index, chunks, metadata):
    unique_ids = [f"doc_{url_index}_{i}" for i in range(len(chunks))]
    collection.add(
        documents=[tc.page_content for tc in chunks],
        metadatas=[metadata for i in range(len(chunks))],
        ids=unique_ids,
    )
    return collection


def delete_documents(collection):
    existing_docs = collection.get()
    doc_ids = [doc_id for doc_id in existing_docs["ids"]]
    logger.info(f"Existing document IDs in collection: {doc_ids}")

    if doc_ids:
        collection.delete(ids=doc_ids)

    remaining_docs = collection.get()
    remaining_doc_ids = [doc_id for doc_id in remaining_docs["ids"]]
    logger.info(f"Remaining document IDs in collection: {remaining_doc_ids}")


client = get_client("C:/shiv/projects/ai-projects/doc-embedder/chip_huyen_db")
collection = get_collection(client, "chip_huyen_collection")


def process_docs():
    links = extract_links(BASE_DOMAIN, BASE_URL)
    urls = list(links)
    logger.info(f"Extracted {urls} links from {BASE_DOMAIN}")
    content_urls = []
    for url_index, sub_url in enumerate(urls):
        logger.info(f"####################  Processing Blog {sub_url}: START")
        docs = scrape_page(sub_url)        
        if docs:
            content_urls.append(sub_url)
            text_chunks = create_chunks(docs)
            logger.info(
                f"Processing Embedding for {len(text_chunks)} chunks from {sub_url} : START"
            )
            add_documents_to_collection(
                collection, url_index, text_chunks, {"url": sub_url}
            )
            logger.info(
                f"Processing Embedding for {len(text_chunks)} chunks from {sub_url} : END"
            )
        else:
            logger.warning(f"No documents found for {sub_url}")

        logger.info(f"####################  Processing Blog {sub_url}: END")
    logger.info(
        f"####################  content_urls : {content_urls}"
    )
    if content_urls:
        logger.info(f"Adding index document for chip_huyen_blog_embedder : START")
        content_urls.insert(
            0, "Index list and content summary of all blogs items in Chip Huyen Blog."
        )
        index_docs = [Document(page_content="\n".join(content_urls))]
        add_documents_to_collection(
            collection,
            f"chip_huyen_{0}",
            index_docs,
            {"url": BASE_URL, "is_index": True},
        )
        logger.info(f"Adding index document for chip_huyen_blog_embedder : END")

    existing_docs = collection.get()
    doc_ids = [doc_id for doc_id in existing_docs["ids"]]
    logger.info(f"Processed document IDs in collection: {doc_ids}")


delete_documents(collection)
process_docs()
