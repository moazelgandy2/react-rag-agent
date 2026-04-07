from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.progress import Progress, TextColumn

from .config import settings

console = Console()


def load_documents(data_dir: str) -> list[Document]:
    base_dir = Path(data_dir)
    if not base_dir.exists():
        console.print(f"[yellow]Warning:[/] Data directory not found: {base_dir}")
        return []

    allowed_ext = {".pdf", ".md", ".txt"}
    documents: list[Document] = []
    loaded_files = 0

    for file_path in base_dir.rglob("*"):
        if not file_path.is_file():
            continue

        extension = file_path.suffix.lower()
        if extension not in allowed_ext:
            console.print(f"[yellow]Skipping unsupported file:[/] {file_path}")
            continue

        try:
            if extension == ".pdf":
                loader = PyPDFLoader(str(file_path))
            else:
                loader = TextLoader(str(file_path), encoding="utf-8")
            file_docs = loader.load()
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Failed to load {file_path}: {exc}[/]")
            continue

        for doc in file_docs:
            doc.metadata["source"] = str(file_path)
            doc.metadata["file_type"] = extension

        documents.extend(file_docs)
        loaded_files += 1

    console.print(f"Loaded {len(documents)} documents from {loaded_files} files")
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index

    console.print(f"Split into {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks: list[Document]) -> Chroma:
    embeddings = OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )

    persist_dir = Path(settings.chroma_persist_dir)
    if persist_dir.exists():
        vector_store = Chroma(
            collection_name=settings.collection_name,
            embedding_function=embeddings,
            persist_directory=settings.chroma_persist_dir,
        )
        existing_ids = vector_store.get().get("ids", [])
        if existing_ids:
            vector_store.delete(ids=existing_ids)

    vector_store = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        collection_name=settings.collection_name,
        persist_directory=settings.chroma_persist_dir,
    )
    console.print(f"Stored {len(chunks)} chunks in ChromaDB")
    return vector_store


def run_ingestion() -> Chroma:
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task_id = progress.add_task("Loading documents...", total=None)
        documents = load_documents(settings.data_dir)
        progress.update(task_id, description="Chunking documents...")
        chunks = chunk_documents(documents)
        progress.update(task_id, description="Creating vector store...")
        vector_store = create_vector_store(chunks)
        progress.update(task_id, description="Ingestion complete")

    return vector_store


if __name__ == "__main__":
    run_ingestion()
