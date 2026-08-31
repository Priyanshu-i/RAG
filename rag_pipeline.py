import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Generator, Union
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from document_loader import load_any_document, split_documents, extract_doc_stats

load_dotenv()

# Valid, Active, and Cost-Efficient Gemini Models currently run by Google
AVAILABLE_MODELS = [
    "gemini-2.5-flash",        # Recommended: fast, highly capable & cost-effective
    "gemini-3.5-flash-lite",   # Ultra cost-effective, lowest latency
    "gemini-3.5-flash",        # Next-gen high-efficiency flash model
    "gemini-flash-latest",     # Google's latest stable flash production model
    "gemini-flash-lite-latest" # Google's latest stable flash-lite production model
]

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"

PERSONA_PROMPTS = {
    "General Assistant": (
        "You are an intelligent, helpful AI research assistant for question-answering tasks. "
        "Use the retrieved document context below to answer the question accurately, clearly, and concisely. "
        "If the document doesn't contain sufficient context to answer, state honestly: 'I cannot find that information in the provided document.' "
        "Always cite relevant sections or facts from the text where applicable."
    ),
    "Executive Brief": (
        "You are an executive Chief of Staff preparing briefing notes for senior leadership. "
        "Analyze the provided document context and deliver high-impact, structured bullet points, key takeaways, and action items. "
        "Be direct, crisp, and highlight business implications, metrics, and risks."
    ),
    "Academic Professor": (
        "You are a distinguished university professor. "
        "Provide thorough, logically structured, and intellectually rigorous answers based on the context. "
        "Cite specific details, explain theoretical foundations, and maintain formal academic precision."
    ),
    "Skeptical Fact-Checker": (
        "You are an investigative fact-checker and auditor with extreme skepticism. "
        "Rely strictly on verifiable evidence in the provided context. "
        "Flag any ambiguities, unsupported claims, contradictions, or missing context. Do not speculate."
    ),
    "ELI5 (Explain Like I'm 5)": (
        "You are a friendly teacher who explains complex topics in delightfully simple terms. "
        "Use vivid everyday analogies, plain English, and zero confusing jargon based on the provided context."
    ),
    "Technical Specialist": (
        "You are a senior principal engineer and domain specialist. "
        "Provide in-depth technical explanations, code snippets, architectural breakdowns, and exact mathematical or structural details based on the context."
    )
}


def extract_text_from_content(content: Any) -> str:
    """Extract plain text string from str, list of parts/dicts, or structured response objects."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    text_parts.append(item["text"])
                elif "content" in item:
                    text_parts.append(str(item["content"]))
            elif hasattr(item, "text"):
                text_parts.append(getattr(item, "text"))
            else:
                text_parts.append(str(item))
        return "".join(text_parts)
    if hasattr(content, "text"):
        return content.text
    return str(content)


def get_api_key(custom_key: Optional[str] = None) -> str:
    """Retrieve Google API key from user input or environment."""
    key = (custom_key or "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    return key


def get_embeddings(api_key: Optional[str] = None) -> GoogleGenerativeAIEmbeddings:
    """Initialize Google Generative AI Embeddings."""
    key = get_api_key(api_key)
    if not key:
        raise ValueError("Google API Key not found. Please set GOOGLE_API_KEY in .env or enter it in the sidebar.")
    return GoogleGenerativeAIEmbeddings(
        model=DEFAULT_EMBEDDING_MODEL,
        google_api_key=key
    )


def get_llm(
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.3,
    api_key: Optional[str] = None,
    streaming: bool = True
) -> ChatGoogleGenerativeAI:
    """Initialize ChatGoogleGenerativeAI instance with validated cost-effective model."""
    key = get_api_key(api_key)
    if not key:
        raise ValueError("Google API Key not found. Please set GOOGLE_API_KEY in .env or enter it in the sidebar.")
    
    # Ensure model is valid or fallback to gemini-2.5-flash
    valid_model = model_name if model_name in AVAILABLE_MODELS else "gemini-2.5-flash"
    
    return ChatGoogleGenerativeAI(
        model=valid_model,
        temperature=temperature,
        google_api_key=key,
        streaming=streaming,
        convert_system_message_to_human=True,
        max_retries=2
    )


def create_vector_store_from_file(
    file_path: str,
    filename: Optional[str] = None,
    file_type: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    api_key: Optional[str] = None
) -> Tuple[FAISS, Dict[str, Any]]:
    """Load, process, chunk, and index any file into a FAISS vectorstore."""
    raw_docs = load_any_document(file_path, filename=filename, file_type=file_type)
    if not raw_docs:
        raise ValueError("No content could be extracted from the document.")

    stats = extract_doc_stats(raw_docs)
    chunks = split_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    stats["chunk_count"] = len(chunks)
    stats["filename"] = filename or os.path.basename(file_path)

    embeddings = get_embeddings(api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, stats


def create_vector_store_from_url(
    web_url: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    api_key: Optional[str] = None
) -> Tuple[FAISS, Dict[str, Any]]:
    """Scrape, split, and index a web URL into a FAISS vectorstore."""
    from document_loader import load_web_url
    raw_docs = load_web_url(web_url)
    if not raw_docs:
        raise ValueError("No content could be extracted from the web URL.")

    stats = extract_doc_stats(raw_docs)
    chunks = split_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    title = raw_docs[0].metadata.get("source", web_url)
    stats["chunk_count"] = len(chunks)
    stats["filename"] = f"🌐 {title[:45]}"
    stats["file_type"] = "web_url"
    stats["url"] = web_url

    embeddings = get_embeddings(api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, stats



def load_vector_store(folder_path: str, api_key: Optional[str] = None) -> FAISS:
    """Load a previously saved FAISS vectorstore from local disk."""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Vector store folder not found: {folder_path}")

    embeddings = get_embeddings(api_key)
    vectorstore = FAISS.load_local(
        folder_path=folder_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


def merge_vector_stores(
    vectorstores: List[FAISS],
    api_key: Optional[str] = None
) -> Optional[FAISS]:
    """Merge multiple FAISS vector stores into a unified library vector store."""
    if not vectorstores:
        return None
    merged = vectorstores[0]
    for vs in vectorstores[1:]:
        merged.merge_from(vs)
    return merged


def reformulate_query_with_history(
    query: str,
    chat_history: List[Dict[str, str]],
    model_name: str = "gemini-2.5-flash",
    api_key: Optional[str] = None
) -> str:
    """Reformulate user query into a standalone question taking conversational history into account."""
    if not chat_history or len(chat_history) < 2:
        return query

    llm = get_llm(model_name=model_name, temperature=0.1, api_key=api_key, streaming=False)

    history_text = ""
    for msg in chat_history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        raw_msg = extract_text_from_content(msg.get("content", ""))
        history_text += f"{role}: {raw_msg}\n"

    reformulate_prompt = (
        "Given the following conversation history and a follow-up question, "
        "rephrase the follow-up question to be a standalone question that captures all needed context "
        "from the history. Do NOT answer the question, only return the rephrased search query.\n\n"
        f"Chat History:\n{history_text}\n"
        f"Follow-up Question: {query}\n\n"
        "Standalone Question:"
    )

    try:
        response = llm.invoke(reformulate_prompt)
        standalone = extract_text_from_content(response.content).strip()
        return standalone if standalone else query
    except Exception:
        return query


def retrieve_relevant_documents(
    vectorstore: FAISS,
    query: str,
    k: int = 5,
    search_type: str = "similarity",
    score_threshold: Optional[float] = None
) -> List[Tuple[Document, float]]:
    """Retrieve relevant documents along with similarity scores."""
    if search_type == "mmr":
        docs = vectorstore.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=max(k * 3, 12),
            lambda_mult=0.7
        )
        return [(doc, 0.88) for doc in docs]
    else:
        docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)
        results = []
        for doc, score in docs_and_scores:
            relevance = max(0.0, min(1.0, 1.0 / (1.0 + float(score))))
            if score_threshold is None or relevance >= score_threshold:
                results.append((doc, round(relevance, 3)))
        return results


def format_context_with_citations(retrieved_docs: List[Tuple[Document, float]]) -> Tuple[str, List[Dict[str, Any]]]:
    """Format retrieved chunks into a prompt context string and extract rich citation metadata."""
    context_parts = []
    sources = []

    for idx, (doc, score) in enumerate(retrieved_docs, start=1):
        source_name = doc.metadata.get("source", "Document")
        page_num = doc.metadata.get("page", 1)
        chunk_id = doc.metadata.get("chunk_id", idx)
        file_type = doc.metadata.get("file_type", "doc")
        url = doc.metadata.get("url", "")

        header = f"[Source {idx}: {source_name} | Page {page_num}]"
        context_parts.append(f"{header}\n{doc.page_content}")

        sources.append({
            "citation_id": idx,
            "source": source_name,
            "page": page_num,
            "chunk_id": chunk_id,
            "file_type": file_type,
            "url": url,
            "relevance_score": score,
            "content_preview": doc.page_content[:250] + ("..." if len(doc.page_content) > 250 else ""),
            "full_content": doc.page_content
        })

    formatted_context = "\n\n---\n\n".join(context_parts)
    return formatted_context, sources


def generate_rag_response_stream(
    vectorstore: FAISS,
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    model_name: str = "gemini-2.5-flash",
    persona: str = "General Assistant",
    temperature: float = 0.3,
    k: int = 5,
    search_type: str = "similarity",
    api_key: Optional[str] = None
) -> Tuple[Generator[str, None, None], List[Dict[str, Any]], str]:
    """Stream answer tokens safely handling structured chunks, citations, and conversational memory."""
    standalone_query = reformulate_query_with_history(
        query=query,
        chat_history=chat_history or [],
        model_name=model_name,
        api_key=api_key
    )

    retrieved_items = retrieve_relevant_documents(
        vectorstore=vectorstore,
        query=standalone_query,
        k=k,
        search_type=search_type
    )

    if not retrieved_items:
        def empty_stream():
            yield "I could not find relevant information in the uploaded document(s) to answer your question."
        return empty_stream(), [], standalone_query

    context_text, citations = format_context_with_citations(retrieved_items)
    base_persona_prompt = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["General Assistant"])

    system_instruction = (
        f"{base_persona_prompt}\n\n"
        "Guidelines:\n"
        "- Base your answer directly on the provided context.\n"
        "- Cite sources inline when helpful (e.g. [Source 1], [Source 2]).\n"
        "- Format your answer with clear markdown headings, lists, or tables where appropriate.\n\n"
        "=== RETRIEVED DOCUMENT CONTEXT ===\n"
        f"{context_text}\n"
        "=================================="
    )

    llm = get_llm(
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        streaming=True
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", "{input}")
    ])

    chain = prompt | llm

    def token_generator():
        for chunk in chain.stream({"input": query}):
            if hasattr(chunk, "content"):
                chunk_text = extract_text_from_content(chunk.content)
                if chunk_text:
                    yield chunk_text
            elif isinstance(chunk, str):
                yield chunk

    return token_generator(), citations, standalone_query