import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".log",
    ".py",
    ".html",
    ".pdf",
    ".docx",
}


class IngestState(TypedDict, total=False):
    input_path: str
    memory_file: str
    docs_file: str
    bucket_name: Optional[str]
    cloud_prefix: str
    raw_documents: List[Document]
    chunks: List[Document]
    vector_store: FAISS
    artifact_path: str
    cloud_uri: Optional[str]
    error: Optional[str]
    temp_dir: Optional[str]


def _iter_paths(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    collected: List[Path] = []
    for item in path.rglob("*"):
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
            collected.append(item)
    return collected


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError("Install `pypdf` to read PDF files.") from exc

    reader = PdfReader(str(path))
    text_parts = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(text_parts).strip()


def _read_docx(path: Path) -> str:
    try:
        import docx2txt
    except Exception as exc:
        raise RuntimeError("Install `docx2txt` to read DOCX files.") from exc
    return docx2txt.process(str(path)) or ""


def _read_text_like(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix == ".docx":
        return _read_docx(path)
    return _read_text_like(path)


def node_validate_input(state: IngestState) -> IngestState:
    p = Path(state["input_path"])
    if not p.exists():
        return {"error": f"Input path not found: {p}"}
    return {}


def node_load_documents(state: IngestState) -> IngestState:
    p = Path(state["input_path"])
    files = _iter_paths(p)
    docs: List[Document] = []
    for file_path in files:
        try:
            content = _read_file(file_path)
            if content.strip():
                docs.append(
                    Document(
                        page_content=content,
                        metadata={"source": str(file_path.resolve())},
                    )
                )
        except Exception as exc:
            return {"error": f"Failed reading {file_path}: {exc}"}

    if not docs:
        return {"error": f"No readable supported documents found in: {p}"}
    return {"raw_documents": docs}


def node_chunk_documents(state: IngestState) -> IngestState:
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(state["raw_documents"])
    return {"chunks": chunks}


def node_build_index(state: IngestState) -> IngestState:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    memory_file = state["memory_file"]
    docs_file = state["docs_file"]

    all_chunks = state["chunks"]
    if os.path.exists(memory_file):
        try:
            existing_vs = FAISS.load_local(
                memory_file,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            existing_vs.add_documents(all_chunks)
            vector_store = existing_vs
        except Exception:
            vector_store = FAISS.from_documents(all_chunks, embeddings)
    else:
        vector_store = FAISS.from_documents(all_chunks, embeddings)

    vector_store.save_local(memory_file)

    existing_docs: List[Document] = []
    if os.path.exists(docs_file):
        try:
            with open(docs_file, "rb") as f:
                existing_docs = pickle.load(f)
        except Exception:
            existing_docs = []
    with open(docs_file, "wb") as f:
        pickle.dump(existing_docs + all_chunks, f)

    return {"vector_store": vector_store}


def node_prepare_artifact(state: IngestState) -> IngestState:
    memory_file = Path(state["memory_file"])
    docs_file = Path(state["docs_file"])
    temp_dir = Path(tempfile.mkdtemp(prefix="igris_ingest_"))
    shutil.copytree(memory_file, temp_dir / memory_file.name, dirs_exist_ok=True)
    shutil.copy2(docs_file, temp_dir / docs_file.name)

    artifact_base = temp_dir / "igris_artifacts"
    archive_file = shutil.make_archive(str(artifact_base), "zip", root_dir=temp_dir)
    return {"artifact_path": archive_file, "temp_dir": str(temp_dir)}


def node_upload_to_s3(state: IngestState) -> IngestState:
    bucket_name = state.get("bucket_name")
    if not bucket_name:
        return {"cloud_uri": None}

    try:
        import boto3
    except Exception as exc:
        return {"error": f"S3 upload requested but boto3 is not installed: {exc}"}

    artifact_path = state["artifact_path"]
    key_prefix = state.get("cloud_prefix", "igris")
    key = f"{key_prefix.rstrip('/')}/igris_artifacts.zip"
    s3 = boto3.client("s3")
    try:
        s3.upload_file(artifact_path, bucket_name, key)
    except Exception as exc:
        return {"error": f"S3 upload failed: {exc}"}
    return {"cloud_uri": f"s3://{bucket_name}/{key}"}


def node_cleanup(state: IngestState) -> IngestState:
    temp_dir = state.get("temp_dir")
    if temp_dir and os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
    return {}


def _route_on_error(state: IngestState, ok_target: str) -> str:
    return END if state.get("error") else ok_target


def _route_after_validate(state: IngestState) -> str:
    return _route_on_error(state, "load_documents")


def _route_after_load(state: IngestState) -> str:
    return _route_on_error(state, "chunk_documents")


def _route_after_chunk(state: IngestState) -> str:
    return _route_on_error(state, "build_index")


def _route_after_build(state: IngestState) -> str:
    return _route_on_error(state, "prepare_artifact")


def _route_after_prepare(state: IngestState) -> str:
    return _route_on_error(state, "upload_to_s3")


def _route_after_upload(state: IngestState) -> str:
    return _route_on_error(state, "cleanup")


def build_ingest_graph():
    graph = StateGraph(IngestState)
    graph.add_node("validate_input", node_validate_input)
    graph.add_node("load_documents", node_load_documents)
    graph.add_node("chunk_documents", node_chunk_documents)
    graph.add_node("build_index", node_build_index)
    graph.add_node("prepare_artifact", node_prepare_artifact)
    graph.add_node("upload_to_s3", node_upload_to_s3)
    graph.add_node("cleanup", node_cleanup)

    graph.set_entry_point("validate_input")
    graph.add_conditional_edges(
        "validate_input",
        _route_after_validate,
        {END: END, "load_documents": "load_documents"},
    )
    graph.add_conditional_edges(
        "load_documents",
        _route_after_load,
        {END: END, "chunk_documents": "chunk_documents"},
    )
    graph.add_conditional_edges(
        "chunk_documents",
        _route_after_chunk,
        {END: END, "build_index": "build_index"},
    )
    graph.add_conditional_edges(
        "build_index",
        _route_after_build,
        {END: END, "prepare_artifact": "prepare_artifact"},
    )
    graph.add_conditional_edges(
        "prepare_artifact",
        _route_after_prepare,
        {END: END, "upload_to_s3": "upload_to_s3"},
    )
    graph.add_conditional_edges(
        "upload_to_s3",
        _route_after_upload,
        {END: END, "cleanup": "cleanup"},
    )
    graph.add_edge("cleanup", END)
    return graph.compile()


def run_ingestion_workflow(
    input_path: str,
    memory_file: str,
    docs_file: str,
    bucket_name: Optional[str] = None,
    cloud_prefix: str = "igris",
) -> Dict[str, Any]:
    app = build_ingest_graph()
    result = app.invoke(
        {
            "input_path": input_path,
            "memory_file": memory_file,
            "docs_file": docs_file,
            "bucket_name": bucket_name,
            "cloud_prefix": cloud_prefix,
        }
    )
    return dict(result)
