
"""Document ingestion: load, split and index files in Chroma."""

import csv
import logging
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings

logger = logging.getLogger(__name__)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TEXT_SUFFIXES = {".md", ".txt"}


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)



def load_documents(directory: str | None = None) -> list[Document]:
    """Load Markdown, text, PDF and CSV files from the configured directories."""
    base = Path(directory or settings.documents_dir)
    documents: list[Document] = []

    # Include the tables directory when using the default document directory.
    directories = [base]
    if directory is None:
        tables_dir = Path("data/tables")
        if tables_dir.resolve() != base.resolve():
            directories.append(tables_dir)

    for current_dir in directories:
        if not current_dir.exists():
            logger.warning(
                "Document directory does not exist: %s", current_dir
            )
            continue

        for path in sorted(current_dir.rglob("*")):
            if not path.is_file():
                continue

            suffix = path.suffix.lower()

            if suffix in TEXT_SUFFIXES:
                documents.append(
                    Document(
                        page_content=path.read_text(encoding="utf-8"),
                        metadata={"source": path.name},
                    )
                )

            elif suffix == ".pdf":
                from langchain_community.document_loaders import PyPDFLoader

                documents.extend(PyPDFLoader(str(path)).load())

            elif suffix == ".csv":
                # Convert each CSV row into searchable text.
                with path.open(
                    "r", encoding="utf-8-sig", newline=""
                ) as file:
                    reader = csv.DictReader(file)

                    for row_number, row in enumerate(reader, start=2):
                        content = "\n".join(
                            f"{key}: {value}"
                            for key, value in row.items()
                            if key is not None and value is not None
                        )

                        if content.strip():
                            documents.append(
                                Document(
                                    page_content=content,
                                    metadata={
                                        "source": path.name,
                                        "row": row_number,
                                    },
                                )
                            )

    logger.info(
        "Loaded %d documents from %s",
        len(documents),
        ", ".join(str(path) for path in directories),
    )
    return documents



def split_documents(documents: list[Document]) -> list[Document]:
    """Split long documents into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def get_vectorstore() -> Chroma:
    """Connect to the persistent local Chroma collection."""
    return Chroma(
        collection_name=settings.collection_name,
        persist_directory=settings.chroma_dir,
        embedding_function=get_embeddings(),
    )


def ingest(directory: str | None = None) -> int:
    """Rebuild the index from supported files in the selected directory."""
    chunks = split_documents(load_documents(directory))
    store = get_vectorstore()
    store.reset_collection()

    if chunks:
        store.add_documents(chunks)

    logger.info(
        "Indexed %d chunks into '%s'",
        len(chunks),
        settings.collection_name,
    )
    return len(chunks)


def ensure_index() -> None:
    """Build the index if the persistent collection is empty."""
    if get_vectorstore()._collection.count() == 0:
        ingest()


def get_retriever(k: int | None = None):
    """Return a retriever for the indexed documents."""
    return get_vectorstore().as_retriever(
        search_kwargs={"k": k or settings.retrieval_k}
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    count = ingest()
    print(f"Done. {count} chunks indexed in {settings.chroma_dir}")


if __name__ == "__main__":
    main()
