import re
from collections import Counter
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

STOPWORDS = set([
    "the", "and", "is", "in", "to", "of", "a", "for", "with", "as", "on", "that", "this",
    "by", "an", "be", "are", "from", "at", "it", "or", "was", "we", "which", "you", "your",
    "can", "has", "have", "not", "but", "our", "all", "their", "more", "also", "into", "other",
    "will", "its", "were", "been", "they", "than", "may", "these", "such", "using", "use",
    "used", "one", "two", "each", "through", "about", "both", "between", "under", "after", "before"
])


def get_all_vectorstore_documents(vectorstore: FAISS) -> List[Document]:
    """Extract all indexed Document objects from FAISS in-memory docstore."""
    docs = []
    try:
        # Access in-memory docstore
        docstore = vectorstore.docstore
        if hasattr(docstore, "_dict"):
            docs = list(docstore._dict.values())
    except Exception:
        pass
    return docs


def compute_keyword_frequencies(documents: List[Document], top_n: int = 15) -> List[Dict[str, Any]]:
    """Calculate top non-stopword keyword frequencies across document chunks."""
    all_text = " ".join([d.page_content for d in documents]).lower()
    words = re.findall(r"\b[a-zA-Z]{4,}\b", all_text)
    filtered = [w for w in words if w not in STOPWORDS]

    counts = Counter(filtered).most_common(top_n)
    return [{"keyword": word.capitalize(), "count": count} for word, count in counts]


def compute_document_analytics(vectorstore: FAISS) -> Dict[str, Any]:
    """Generate comprehensive document analytics, chunk distribution, and top keywords."""
    docs = get_all_vectorstore_documents(vectorstore)
    if not docs:
        return {
            "total_chunks": 0,
            "total_words": 0,
            "avg_chunk_length": 0,
            "min_chunk_length": 0,
            "max_chunk_length": 0,
            "keywords": [],
            "chunks_summary": []
        }

    chunk_lengths = [len(d.page_content) for d in docs]
    all_text = " ".join([d.page_content for d in docs])
    words = all_text.split()

    keywords = compute_keyword_frequencies(docs, top_n=12)

    chunks_summary = []
    for idx, doc in enumerate(docs):
        chunks_summary.append({
            "chunk_id": doc.metadata.get("chunk_id", idx + 1),
            "page": doc.metadata.get("page", 1),
            "char_count": len(doc.page_content),
            "preview": doc.page_content[:180].strip() + ("..." if len(doc.page_content) > 180 else ""),
            "full_text": doc.page_content
        })

    return {
        "total_chunks": len(docs),
        "total_words": len(words),
        "avg_chunk_length": int(sum(chunk_lengths) / len(chunk_lengths)) if chunk_lengths else 0,
        "min_chunk_length": min(chunk_lengths) if chunk_lengths else 0,
        "max_chunk_length": max(chunk_lengths) if chunk_lengths else 0,
        "keywords": keywords,
        "chunks_summary": chunks_summary
    }

