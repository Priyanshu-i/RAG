import json
import re
from typing import Dict, Any, List, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from rag_pipeline import get_llm, retrieve_relevant_documents, extract_text_from_content

SUPPORTED_LANGUAGES = [
    "Spanish", "French", "German", "Hindi", "Japanese",
    "Chinese (Simplified)", "Arabic", "Portuguese", "Italian",
    "Russian", "Korean", "Dutch", "Turkish", "Bengali"
]


def get_representative_context(vectorstore: FAISS, sample_queries: Optional[List[str]] = None, max_chunks: int = 8) -> str:
    """Retrieve representative chunks across broad thematic queries to construct comprehensive context."""
    queries = sample_queries or [
        "introduction overview main topic summary",
        "key findings results conclusions importance",
        "methodology approach structure background",
        "metrics data numbers risks challenges"
    ]

    all_docs = []
    seen_contents = set()

    for q in queries:
        docs_and_scores = retrieve_relevant_documents(vectorstore, q, k=3, search_type="mmr")
        for doc, _ in docs_and_scores:
            cleaned = doc.page_content.strip()
            if cleaned not in seen_contents:
                seen_contents.add(cleaned)
                all_docs.append(doc)
                if len(all_docs) >= max_chunks:
                    break
        if len(all_docs) >= max_chunks:
            break

    context_str = "\n\n---\n\n".join([f"[{d.metadata.get('source', 'Doc')} - Page {d.metadata.get('page', 1)}]\n{d.page_content}" for d in all_docs])
    return context_str


def generate_executive_summary(
    vectorstore: FAISS,
    model_name: str = "gemini-2.5-flash",
    api_key: Optional[str] = None
) -> str:
    """Generate an executive briefing with key takeaways, risks, and action items."""
    context = get_representative_context(vectorstore, max_chunks=8)
    llm = get_llm(model_name=model_name, temperature=0.2, api_key=api_key, streaming=False)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an elite management consultant and executive analyst. "
            "Based on the provided document excerpts, produce a comprehensive, beautifully formatted Executive Brief.\n\n"
            "Structure your output strictly as follows:\n"
            "## 📋 Executive Briefing\n"
            "**Core Purpose / Central Thesis**: (2-3 clear sentences)\n\n"
            "### 🌟 Key Findings & Core Takeaways\n"
            "- (4-6 high-impact bullet points with bold keywords)\n\n"
            "### 📊 Key Data, Metrics & Evidence\n"
            "- (Specific numbers, statistics, percentages, or milestones cited)\n\n"
            "### ⚠️ Risks, Caveats & Limitations\n"
            "- (Critical challenges, bottlenecks, or blind spots mentioned)\n\n"
            "### 🎯 Strategic Action Items & Next Steps\n"
            "- (Actionable recommendations derived from the document)"
        )),
        ("human", "Here are the document excerpts:\n\n{context}\n\nPlease generate the executive brief.")
    ])

    chain = prompt | llm
    response = chain.invoke({"context": context})
    return extract_text_from_content(response.content)


def generate_smart_faqs(
    vectorstore: FAISS,
    model_name: str = "gemini-2.5-flash",
    api_key: Optional[str] = None
) -> List[Dict[str, str]]:
    """Generate top 5 insightful questions with concise answers based on the document."""
    context = get_representative_context(vectorstore, max_chunks=6)
    llm = get_llm(model_name=model_name, temperature=0.3, api_key=api_key, streaming=False)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an AI document curator. Based on the document context, formulate the top 5 most important and insightful questions "
            "that someone reading this document would want answered, along with accurate, concise answers (2-3 sentences each).\n\n"
            "Output your answer ONLY as a valid JSON array of objects with keys 'question' and 'answer'.\n"
            "Do not include markdown code block backticks or extraneous text, just raw JSON:\n"
            '[{"question": "...", "answer": "..."}, ...]'
        )),
        ("human", "Document Excerpts:\n\n{context}")
    ])

    chain = prompt | llm
    response = chain.invoke({"context": context})
    raw_text = extract_text_from_content(response.content).strip()

    # Clean code blocks if present
    raw_text = re.sub(r"^```json\s*", "", raw_text, flags=re.MULTILINE)
    raw_text = re.sub(r"^```\s*", "", raw_text, flags=re.MULTILINE)

    try:
        faqs = json.loads(raw_text)
        if isinstance(faqs, list):
            return faqs
    except Exception:
        pass

    return [
        {"question": "What is the primary topic of this document?", "answer": "The document provides an in-depth analysis of the core concepts, methodologies, and findings presented."},
        {"question": "What are the key conclusions?", "answer": "The author outlines critical insights, recommended practices, and future directions."}
    ]


def generate_interactive_quiz(
    vectorstore: FAISS,
    model_name: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
    num_questions: int = 4
) -> List[Dict[str, Any]]:
    """Generate an interactive multiple choice quiz with answer keys and explanations."""
    context = get_representative_context(vectorstore, max_chunks=7)
    llm = get_llm(model_name=model_name, temperature=0.2, api_key=api_key, streaming=False)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            f"You are an educational assessment expert. Create {num_questions} multiple-choice quiz questions based strictly on the provided document excerpts.\n"
            "Each question must have 4 distinct options (A, B, C, D), exactly one correct answer, and a concise explanation.\n\n"
            "Output your answer ONLY as a valid JSON array of objects with the following keys:\n"
            "- 'id': integer (1, 2, ...)\n"
            "- 'question': string\n"
            "- 'options': array of 4 strings\n"
            "- 'correct_index': integer (0 for first option, 1 for second, 2 for third, 3 for fourth)\n"
            "- 'explanation': string explaining why this is correct based on the text\n\n"
            "Return raw JSON only without markdown formatting."
        )),
        ("human", "Document Excerpts:\n\n{context}")
    ])

    chain = prompt | llm
    response = chain.invoke({"context": context})
    raw_text = extract_text_from_content(response.content).strip()

    raw_text = re.sub(r"^```json\s*", "", raw_text, flags=re.MULTILINE)
    raw_text = re.sub(r"^```\s*", "", raw_text, flags=re.MULTILINE)

    try:
        quiz = json.loads(raw_text)
        if isinstance(quiz, list) and len(quiz) > 0:
            return quiz
    except Exception:
        pass

    return [
        {
            "id": 1,
            "question": "What is the primary subject matter described in the document?",
            "options": ["Theoretical Framework", "Experimental Data Analysis", "Comprehensive Overview", "Policy Guidelines"],
            "correct_index": 2,
            "explanation": "The document provides a comprehensive overview of the discussed subject."
        }
    ]


def extract_entities_and_metrics(
    vectorstore: FAISS,
    model_name: str = "gemini-2.5-flash",
    api_key: Optional[str] = None
) -> str:
    """Extract structured entities, dates, organizations, and metrics from the document."""
    context = get_representative_context(vectorstore, max_chunks=7)
    llm = get_llm(model_name=model_name, temperature=0.1, api_key=api_key, streaming=False)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a knowledge graph and information extraction engine. "
            "Analyze the provided document context and extract key structured information in markdown tables:\n\n"
            "1. **🏢 Key Organizations & People**: Name, Role/Entity Type, Context in Document\n"
            "2. **📅 Key Dates & Milestones**: Date/Timeframe, Event / Significance\n"
            "3. **📊 Key Metrics & Quantitative Data**: Metric Name, Value / Unit, Significance\n"
            "4. **📖 Core Concepts & Terminology**: Term, Concise Definition from Context"
        )),
        ("human", "Document Excerpts:\n\n{context}")
    ])

    chain = prompt | llm
    response = chain.invoke({"context": context})
    return extract_text_from_content(response.content)


def compare_documents(
    vectorstore_a: FAISS,
    doc_name_a: str,
    vectorstore_b: FAISS,
    doc_name_b: str,
    model_name: str = "gemini-2.5-flash",
    api_key: Optional[str] = None
) -> str:
    """Compare two documents side-by-side highlighting overlaps, unique points, and contrasts."""
    context_a = get_representative_context(vectorstore_a, max_chunks=5)
    context_b = get_representative_context(vectorstore_b, max_chunks=5)

    llm = get_llm(model_name=model_name, temperature=0.2, api_key=api_key, streaming=False)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert comparative analyst. Compare and contrast Document A and Document B based on the provided excerpts.\n\n"
            "Structure your comparative report as follows:\n"
            "## ⚖️ Comparative Analysis: {doc_a} vs {doc_b}\n\n"
            "### 🤝 Common Ground & Shared Themes\n"
            "- (Key points both documents agree on or share)\n\n"
            "### ⚔️ Key Differences & Contrasts\n"
            "- (How their perspectives, findings, or methods differ)\n\n"
            "### 🔍 Unique Aspects of {doc_a}\n"
            "- (Specific points only mentioned in Document A)\n\n"
            "### 🔍 Unique Aspects of {doc_b}\n"
            "- (Specific points only mentioned in Document B)\n\n"
            "### 📊 Comparison Summary Table\n"
            "| Dimension | {doc_a} | {doc_b} |\n"
            "|---|---|---|\n"
            "| Primary Focus | ... | ... |\n"
            "| Key Outcome | ... | ... |\n"
            "| Target Audience | ... | ... |"
        )),
        ("human", (
            f"=== DOCUMENT A: {doc_name_a} ===\n{context_a}\n\n"
            f"=== DOCUMENT B: {doc_name_b} ===\n{context_b}\n\n"
            "Please deliver the detailed comparative analysis."
        ))
    ])

    chain = prompt | llm
    response = chain.invoke({
        "doc_a": doc_name_a,
        "doc_b": doc_name_b
    })
    return extract_text_from_content(response.content)


def translate_text(
    text: str,
    target_language: str,
    model_name: str = "gemini-2.5-flash",
    api_key: Optional[str] = None
) -> str:
    """Translate text into target language preserving markdown formatting."""
    llm = get_llm(model_name=model_name, temperature=0.1, api_key=api_key, streaming=False)
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a professional translator. Translate the following text into {target_language}. Maintain all markdown formatting, bullet points, headers, and citations. Provide only the translated text."),
        ("human", "{text}")
    ])
    chain = prompt | llm
    response = chain.invoke({"text": text})
    return extract_text_from_content(response.content)
