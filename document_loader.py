import os
import re
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    import docx
except ImportError:
    docx = None


def load_pdf(file_path: str, filename: Optional[str] = None) -> List[Document]:
    """Load PDF using PyPDFLoader with page metadata."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    display_name = filename or os.path.basename(file_path)
    for i, doc in enumerate(docs):
        doc.metadata["source"] = display_name
        doc.metadata["file_type"] = "pdf"
        if "page" not in doc.metadata:
            doc.metadata["page"] = i + 1
        else:
            # 1-index pages for user readability
            doc.metadata["page"] = doc.metadata["page"] + 1
    return docs


def load_docx(file_path: str, filename: Optional[str] = None) -> List[Document]:
    """Load DOCX Word document with paragraph & table extraction."""
    display_name = filename or os.path.basename(file_path)
    if not docx:
        raise ImportError("python-docx is required to load DOCX files.")

    doc = docx.Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

    # Also extract tables if present
    table_texts = []
    for table in doc.tables:
        t_data = []
        for row in table.rows:
            row_data = [cell.text.strip() for cell in row.cells]
            t_data.append(" | ".join(row_data))
        if t_data:
            table_texts.append("\n".join(t_data))

    full_text = "\n\n".join(paragraphs)
    if table_texts:
        full_text += "\n\n--- Tables ---\n\n" + "\n\n".join(table_texts)

    # Estimate pages (approx 500 words per page)
    words = full_text.split()
    estimated_pages = max(1, len(words) // 500)

    # If long, split into simulated page sections
    chunk_size_words = 500
    docs = []
    if len(words) > chunk_size_words:
        for p_idx in range(0, len(words), chunk_size_words):
            page_words = words[p_idx:p_idx + chunk_size_words]
            page_text = " ".join(page_words)
            docs.append(Document(
                page_content=page_text,
                metadata={
                    "source": display_name,
                    "file_type": "docx",
                    "page": (p_idx // chunk_size_words) + 1
                }
            ))
    else:
        docs.append(Document(
            page_content=full_text,
            metadata={
                "source": display_name,
                "file_type": "docx",
                "page": 1
            }
        ))
    return docs


def load_text_or_markdown(file_path: str, filename: Optional[str] = None) -> List[Document]:
    """Load TXT or Markdown files."""
    display_name = filename or os.path.basename(file_path)
    ext = Path(file_path).suffix.lower()
    file_type = "markdown" if ext in [".md", ".markdown"] else "text"

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Split into sections if markdown headers exist, or simulated pages
    words = content.split()
    chunk_size_words = 600
    docs = []
    if len(words) > chunk_size_words:
        for p_idx in range(0, len(words), chunk_size_words):
            page_words = words[p_idx:p_idx + chunk_size_words]
            page_text = " ".join(page_words)
            docs.append(Document(
                page_content=page_text,
                metadata={
                    "source": display_name,
                    "file_type": file_type,
                    "page": (p_idx // chunk_size_words) + 1
                }
            ))
    else:
        docs.append(Document(
            page_content=content,
            metadata={
                "source": display_name,
                "file_type": file_type,
                "page": 1
            }
        ))
    return docs


def load_csv(file_path: str, filename: Optional[str] = None) -> List[Document]:
    """Load CSV or spreadsheet and convert into structured descriptive records."""
    display_name = filename or os.path.basename(file_path)
    df = pd.read_csv(file_path)

    # Generate overview summary
    summary = f"Dataset Overview for {display_name}:\n"
    summary += f"- Total Rows: {len(df)}\n"
    summary += f"- Columns: {', '.join(df.columns.tolist())}\n"
    summary += f"- Statistical Summary:\n{df.describe(include='all').to_string()}\n\n"

    # Break rows into batches of 20
    docs = [
        Document(
            page_content=summary,
            metadata={"source": display_name, "file_type": "csv", "page": 1, "section": "Summary"}
        )
    ]

    batch_size = 20
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i + batch_size]
        batch_text = f"Records {i+1} to {min(i+batch_size, len(df))}:\n" + batch_df.to_string(index=False)
        docs.append(Document(
            page_content=batch_text,
            metadata={
                "source": display_name,
                "file_type": "csv",
                "page": (i // batch_size) + 2,
                "section": f"Rows {i+1}-{min(i+batch_size, len(df))}"
            }
        ))
    return docs


def load_web_url(url: str) -> List[Document]:
    """Scrape and extract text content from a web URL."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers, timeout=12)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script, style, navigation, footer tags
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else url
    # Clean text
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)

    words = clean_text.split()
    docs = []
    chunk_size_words = 500
    if len(words) > chunk_size_words:
        for p_idx in range(0, len(words), chunk_size_words):
            page_words = words[p_idx:p_idx + chunk_size_words]
            page_text = " ".join(page_words)
            docs.append(Document(
                page_content=f"Source: {title} ({url})\n\n{page_text}",
                metadata={
                    "source": title,
                    "url": url,
                    "file_type": "web_url",
                    "page": (p_idx // chunk_size_words) + 1
                }
            ))
    else:
        docs.append(Document(
            page_content=f"Source: {title} ({url})\n\n{clean_text}",
            metadata={
                "source": title,
                "url": url,
                "file_type": "web_url",
                "page": 1
            }
        ))
    return docs


def load_any_document(file_path: str, filename: Optional[str] = None, file_type: Optional[str] = None) -> List[Document]:
    """Universal document loader dispatching to the appropriate parser."""
    fname = filename or os.path.basename(file_path)
    ext = Path(fname).suffix.lower()

    if file_type == "web_url" or file_path.startswith(("http://", "https://")):
        return load_web_url(file_path)
    elif ext == ".pdf":
        return load_pdf(file_path, fname)
    elif ext in [".docx", ".doc"]:
        return load_docx(file_path, fname)
    elif ext == ".csv":
        return load_csv(file_path, fname)
    elif ext in [".txt", ".md", ".markdown", ".json", ".log", ".py", ".js", ".html"]:
        return load_text_or_markdown(file_path, fname)
    else:
        # Fallback to plain text
        return load_text_or_markdown(file_path, fname)


def extract_doc_stats(documents: List[Document]) -> Dict[str, Any]:
    """Calculate rich metadata metrics for a set of loaded documents."""
    total_chars = sum(len(d.page_content) for d in documents)
    all_text = " ".join(d.page_content for d in documents)
    words = all_text.split()
    total_words = len(words)
    est_tokens = int(total_words * 1.3)
    est_read_time_min = max(1, round(total_words / 200))
    pages = max([d.metadata.get("page", 1) for d in documents], default=1)
    file_type = documents[0].metadata.get("file_type", "document") if documents else "document"

    # Extract sample snippet preview (first 300 chars)
    preview = all_text[:300].strip() + ("..." if len(all_text) > 300 else "")

    return {
        "total_chars": total_chars,
        "total_words": total_words,
        "est_tokens": est_tokens,
        "est_reading_time_min": est_read_time_min,
        "page_count": pages,
        "file_type": file_type,
        "preview": preview,
        "raw_doc_count": len(documents)
    }


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """Split documents into semantic chunks with preserved metadata."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx + 1
        chunk.metadata["chunk_char_count"] = len(chunk.page_content)
    return chunks

