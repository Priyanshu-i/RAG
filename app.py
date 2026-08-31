import os
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
import streamlit as st

from langchain_community.vectorstores import FAISS
from rag_pipeline import (
    AVAILABLE_MODELS,
    PERSONA_PROMPTS,
    create_vector_store_from_file,
    create_vector_store_from_url,
    load_vector_store,
    merge_vector_stores,
    generate_rag_response_stream,
    get_api_key
)

from document_loader import load_web_url, extract_doc_stats, split_documents
from ai_studio import (
    SUPPORTED_LANGUAGES,
    generate_executive_summary,
    generate_smart_faqs,
    generate_interactive_quiz,
    extract_entities_and_metrics,
    compare_documents,
    translate_text
)
from analytics import compute_document_analytics
from utils import (
    save_uploaded_file,
    generate_tts_audio,
    format_chat_as_markdown,
    format_chat_as_json,
    delete_document_vectorstore
)

# Page Configuration
st.set_page_config(
    page_title="RAG AI Document Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Glassmorphism & Modern Accents)
st.markdown("""
<style>
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 12px;
        backdrop-filter: blur(8px);
    }
    .badge-chip {
        display: inline-block;
        padding: 3px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        border-radius: 12px;
        background-color: #3b82f6;
        color: white;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .badge-pdf { background-color: #ef4444; }
    .badge-docx { background-color: #2563eb; }
    .badge-txt { background-color: #10b981; }
    .badge-csv { background-color: #f59e0b; }
    .badge-web { background-color: #8b5cf6; }
    
    /* Citation Box */
    .citation-box {
        background: rgba(59, 130, 246, 0.07);
        border-left: 3px solid #3b82f6;
        border-radius: 6px;
        padding: 8px 12px;
        margin: 6px 0;
        font-size: 0.88rem;
    }
    
    /* Persona indicator */
    .persona-badge {
        display: inline-block;
        background: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.3);
        border-radius: 16px;
        padding: 2px 10px;
        font-size: 0.8rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Ensure Required Directories Exist
Path("vectorstores").mkdir(exist_ok=True)
Path("temp").mkdir(exist_ok=True)

# ----------------- SESSION STATE INITIALIZATION -----------------
if "current_doc" not in st.session_state:
    st.session_state.current_doc = None
if "chats" not in st.session_state:
    st.session_state.chats = {}  # {doc_id: [{"role": ..., "content": ..., "citations": ...}]}
if "vector_stores" not in st.session_state:
    st.session_state.vector_stores = {}
if "doc_metadata" not in st.session_state:
    st.session_state.doc_metadata = {}
if "starter_prompt_to_run" not in st.session_state:
    st.session_state.starter_prompt_to_run = None
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = {}  # {doc_id: quiz_json}
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}
if "tts_cache" not in st.session_state:
    st.session_state.tts_cache = {}


def get_doc_id(filename: str) -> str:
    """Generate a clean unique doc ID."""
    return hashlib.md5(filename.encode("utf-8")).hexdigest()[:12]


def load_all_persisted_metadata():
    """Scan vectorstores directory and populate session metadata."""
    v_dir = Path("vectorstores")
    for item in v_dir.iterdir():
        if item.is_dir() and item.name != "_library_all":
            doc_id = item.name
            meta_path = item / "metadata.json"
            if meta_path.exists() and doc_id not in st.session_state.doc_metadata:
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                        st.session_state.doc_metadata[doc_id] = meta
                except Exception:
                    st.session_state.doc_metadata[doc_id] = {
                        "filename": f"Document_{doc_id[:6]}",
                        "file_type": "document",
                        "chunk_count": "Indexed",
                        "page_count": 1
                    }


load_all_persisted_metadata()


# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("🧠 RAG Studio")

    # App Tabs in Sidebar
    tab_docs, tab_engine, tab_studio, tab_analytics = st.tabs(["📚 Docs", "⚙️ Engine", "✨ Studio", "📊 Stats"])

    # ---------- TAB 1: DOCUMENTS ----------
    with tab_docs:
        st.subheader("📥 Ingest Documents")

        # Multi-file batch uploader
        uploaded_files = st.file_uploader(
            "Upload Documents (PDF, DOCX, TXT, MD, CSV)",
            type=["pdf", "docx", "txt", "md", "csv"],
            accept_multiple_files=True
        )

        col_up1, col_up2 = st.columns([1, 1])
        with col_up1:
            process_btn = st.button("⚡ Index Files", type="primary", use_container_width=True)

        if process_btn and uploaded_files:
            for up_file in uploaded_files:
                filename = up_file.name
                doc_id = get_doc_id(filename)
                vs_folder = f"vectorstores/{doc_id}"

                with st.spinner(f"Indexing {filename}..."):
                    try:
                        saved_path = save_uploaded_file(up_file)
                        vs, stats = create_vector_store_from_file(
                            file_path=saved_path,
                            filename=filename,
                            chunk_size=1000,
                            chunk_overlap=200,
                            api_key=st.session_state.get("custom_api_key")
                        )
                        vs.save_local(vs_folder)

                        # Save metadata
                        meta_info = {
                            "doc_id": doc_id,
                            "filename": filename,
                            "file_type": stats.get("file_type", Path(filename).suffix[1:]),
                            "chunk_count": stats.get("chunk_count", 0),
                            "total_words": stats.get("total_words", 0),
                            "page_count": stats.get("page_count", 1),
                            "preview": stats.get("preview", "")
                        }
                        with open(f"{vs_folder}/metadata.json", "w") as f:
                            json.dump(meta_info, f, indent=2)

                        st.session_state.vector_stores[doc_id] = vs
                        st.session_state.doc_metadata[doc_id] = meta_info
                        st.session_state.current_doc = doc_id
                        if doc_id not in st.session_state.chats:
                            st.session_state.chats[doc_id] = []

                        st.success(f"✅ Indexed: {filename}")
                    except Exception as e:
                        st.error(f"Failed to process {filename}: {str(e)}")

        st.divider()

        # Web URL Ingestion
        st.subheader("🌐 Ingest Live Web URL")
        web_url = st.text_input("Enter Article / Webpage URL", placeholder="https://example.com/article")
        if st.button("🔗 Scrape & Ingest URL", use_container_width=True):
            if web_url.strip():
                with st.spinner(f"Scraping & embedding {web_url}..."):
                    try:
                        url_id = hashlib.md5(web_url.strip().encode("utf-8")).hexdigest()[:12]
                        vs_folder = f"vectorstores/{url_id}"

                        vs, stats = create_vector_store_from_url(
                            web_url=web_url.strip(),
                            api_key=st.session_state.get("custom_api_key")
                        )
                        vs.save_local(vs_folder)

                        meta_info = {
                            "doc_id": url_id,
                            "filename": stats.get("filename", f"🌐 {web_url[:35]}"),
                            "file_type": "web_url",
                            "url": web_url.strip(),
                            "chunk_count": stats.get("chunk_count", 0),
                            "total_words": stats.get("total_words", 0),
                            "page_count": stats.get("page_count", 1),
                            "preview": stats.get("preview", "")
                        }
                        with open(f"{vs_folder}/metadata.json", "w") as f:
                            json.dump(meta_info, f, indent=2)

                        st.session_state.vector_stores[url_id] = vs
                        st.session_state.doc_metadata[url_id] = meta_info
                        st.session_state.current_doc = url_id
                        if url_id not in st.session_state.chats:
                            st.session_state.chats[url_id] = []

                        st.success(f"✅ Ingested Webpage: {meta_info['filename']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"URL Ingestion error: {str(e)}")


        st.divider()

        # Document Library Browser
        st.subheader(f"📑 Stored Library ({len(st.session_state.doc_metadata)})")

        if len(st.session_state.doc_metadata) > 1:
            # Multi-document library mode option
            lib_selected = st.session_state.current_doc == "__library__"
            lib_btn_type = "primary" if lib_selected else "secondary"
            if st.button("📚 Chat with Entire Library (All Docs)", type=lib_btn_type, use_container_width=True):
                st.session_state.current_doc = "__library__"
                if "__library__" not in st.session_state.chats:
                    st.session_state.chats["__library__"] = []
                st.rerun()

        for doc_id, meta in list(st.session_state.doc_metadata.items()):
            fname = meta.get("filename", "Unknown Document")
            ftype = meta.get("file_type", "doc").lower()
            chunks = meta.get("chunk_count", 0)

            badge_class = f"badge-{ftype}" if ftype in ["pdf", "docx", "txt", "csv"] else "badge-chip"

            col_a, col_b = st.columns([5, 1])
            with col_a:
                is_active = st.session_state.current_doc == doc_id
                btn_type = "primary" if is_active else "secondary"
                if st.button(f"{fname}", key=f"sel_{doc_id}", type=btn_type, use_container_width=True):
                    st.session_state.current_doc = doc_id
                    # Lazy load vectorstore
                    if doc_id not in st.session_state.vector_stores:
                        vs_path = f"vectorstores/{doc_id}"
                        if os.path.exists(vs_path):
                            with st.spinner(f"Loading {fname}..."):
                                vs = load_vector_store(vs_path, st.session_state.get("custom_api_key"))
                                st.session_state.vector_stores[doc_id] = vs
                    if doc_id not in st.session_state.chats:
                        st.session_state.chats[doc_id] = []
                    st.rerun()
            with col_b:
                if st.button("🗑️", key=f"del_{doc_id}", help="Delete document"):
                    delete_document_vectorstore(doc_id)
                    st.session_state.doc_metadata.pop(doc_id, None)
                    st.session_state.vector_stores.pop(doc_id, None)
                    st.session_state.chats.pop(doc_id, None)
                    if st.session_state.current_doc == doc_id:
                        st.session_state.current_doc = None
                    st.rerun()

    # ---------- TAB 2: RAG ENGINE CONFIG ----------
    with tab_engine:
        st.subheader("⚙️ Model & Retrieval Settings")

        # API Key BYOK
        custom_key = st.text_input(
            "Google API Key (Optional Override)",
            type="password",
            value=st.session_state.get("custom_api_key", ""),
            placeholder="AIzaSy...",
            help="Leave blank to use GOOGLE_API_KEY from .env"
        )
        st.session_state.custom_api_key = custom_key

        key_status = "🟢 Active" if get_api_key(custom_key) else "🔴 Missing"
        st.caption(f"API Key Status: **{key_status}**")

        st.divider()

        # Model Selector
        MODEL_DESCRIPTIONS = {
            "gemini-2.5-flash": "⚡ Gemini 2.5 Flash (Recommended - Fast & Cost-Efficient)",
            "gemini-3.5-flash-lite": "💨 Gemini 3.5 Flash-Lite (Ultra Cost-Efficient & Low Latency)",
            "gemini-3.5-flash": "🚀 Gemini 3.5 Flash (Next-Gen High Efficiency)",
            "gemini-flash-latest": "✨ Gemini Flash Latest (Production Stable)",
            "gemini-flash-lite-latest": "💡 Gemini Flash-Lite Latest (Production Cost-Optimized)"
        }
        model_name = st.selectbox(
            "🧠 Gemini Model",
            AVAILABLE_MODELS,
            index=0,
            format_func=lambda m: MODEL_DESCRIPTIONS.get(m, m)
        )
        st.session_state.model_name = model_name

        # Persona Selector
        persona = st.selectbox(
            "🎭 AI Assistant Persona",
            list(PERSONA_PROMPTS.keys()),
            index=0,
            help="Customizes the tone, reasoning depth, and formatting of answers."
        )
        st.session_state.persona = persona

        # Retrieval Strategy
        retriever_mode = st.radio(
            "🔍 Retrieval Strategy",
            ["similarity", "mmr"],
            format_func=lambda x: "Cosine Similarity (Standard)" if x == "similarity" else "MMR (Maximal Marginal Relevance - Diversity)",
            index=0
        )
        st.session_state.retriever_mode = retriever_mode

        # Top K
        top_k = st.slider("📑 Top-K Chunks to Retrieve", min_value=2, max_value=12, value=5, step=1)
        st.session_state.top_k = top_k

        # Temperature
        temperature = st.slider("🌡️ Creativity / Temperature", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
        st.session_state.temperature = temperature

    # ---------- TAB 3: AI STUDIO ----------
    with tab_studio:
        st.subheader("✨ AI Intelligence Studio")
        if not st.session_state.current_doc or st.session_state.current_doc == "__library__":
            st.info("Select a specific document to use AI Studio tools.")
        else:
            cur_id = st.session_state.current_doc
            cur_meta = st.session_state.doc_metadata.get(cur_id, {})
            cur_fname = cur_meta.get("filename", "Document")

            st.write(f"Active: **{cur_fname}**")

            # Helper to get current vectorstore
            def get_active_vs():
                if cur_id not in st.session_state.vector_stores:
                    st.session_state.vector_stores[cur_id] = load_vector_store(
                        f"vectorstores/{cur_id}",
                        st.session_state.get("custom_api_key")
                    )
                return st.session_state.vector_stores[cur_id]

            # 1. Executive Summary
            if st.button("📋 Generate Executive Brief", use_container_width=True):
                with st.spinner("Analyzing document and distilling executive brief..."):
                    try:
                        vs = get_active_vs()
                        summary_md = generate_executive_summary(
                            vs,
                            model_name=st.session_state.get("model_name", "gemini-2.5-flash"),
                            api_key=st.session_state.get("custom_api_key")
                        )
                        st.session_state[f"summary_{cur_id}"] = summary_md
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            if f"summary_{cur_id}" in st.session_state:
                with st.expander("📋 View Executive Briefing", expanded=True):
                    st.markdown(st.session_state[f"summary_{cur_id}"])

            # 2. Auto FAQ Generator
            if st.button("❓ Generate Smart FAQs", use_container_width=True):
                with st.spinner("Generating document FAQs..."):
                    try:
                        vs = get_active_vs()
                        faqs = generate_smart_faqs(
                            vs,
                            model_name=st.session_state.get("model_name", "gemini-2.5-flash"),
                            api_key=st.session_state.get("custom_api_key")
                        )
                        st.session_state[f"faqs_{cur_id}"] = faqs
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            if f"faqs_{cur_id}" in st.session_state:
                with st.expander("❓ View Smart FAQs", expanded=True):
                    for idx, faq in enumerate(st.session_state[f"faqs_{cur_id}"], 1):
                        st.markdown(f"**Q{idx}: {faq.get('question')}**")
                        st.markdown(f"*{faq.get('answer')}*")
                        if st.button(f"Ask this in chat 💬", key=f"faq_ask_{idx}"):
                            st.session_state.starter_prompt_to_run = faq.get("question")
                            st.rerun()

            # 3. Interactive Quiz
            if st.button("🎯 Generate Interactive Quiz", use_container_width=True):
                with st.spinner("Generating knowledge test questions..."):
                    try:
                        vs = get_active_vs()
                        quiz = generate_interactive_quiz(
                            vs,
                            model_name=st.session_state.get("model_name", "gemini-2.5-flash"),
                            api_key=st.session_state.get("custom_api_key")
                        )
                        st.session_state.quiz_data[cur_id] = quiz
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            # 4. Entity & Metrics Extractor
            if st.button("📊 Extract Entities & Metrics", use_container_width=True):
                with st.spinner("Extracting structured entities, dates & metrics..."):
                    try:
                        vs = get_active_vs()
                        entities = extract_entities_and_metrics(
                            vs,
                            model_name=st.session_state.get("model_name", "gemini-2.5-flash"),
                            api_key=st.session_state.get("custom_api_key")
                        )
                        st.session_state[f"entities_{cur_id}"] = entities
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            if f"entities_{cur_id}" in st.session_state:
                with st.expander("📊 Extracted Entities & Metrics", expanded=True):
                    st.markdown(st.session_state[f"entities_{cur_id}"])

            # 5. Cross Document Comparator
            st.divider()
            st.subheader("⚖️ Compare Two Documents")
            doc_keys = list(st.session_state.doc_metadata.keys())
            if len(doc_keys) >= 2:
                doc_a_id = st.selectbox("Document A", doc_keys, format_func=lambda x: st.session_state.doc_metadata[x].get("filename"), key="comp_a")
                doc_b_id = st.selectbox("Document B", [k for k in doc_keys if k != doc_a_id], format_func=lambda x: st.session_state.doc_metadata[x].get("filename"), key="comp_b")

                if st.button("⚖️ Run Comparative Analysis", use_container_width=True):
                    with st.spinner("Comparing documents..."):
                        try:
                            vs_a = st.session_state.vector_stores.get(doc_a_id) or load_vector_store(f"vectorstores/{doc_a_id}", st.session_state.get("custom_api_key"))
                            vs_b = st.session_state.vector_stores.get(doc_b_id) or load_vector_store(f"vectorstores/{doc_b_id}", st.session_state.get("custom_api_key"))
                            name_a = st.session_state.doc_metadata[doc_a_id]["filename"]
                            name_b = st.session_state.doc_metadata[doc_b_id]["filename"]

                            comp_res = compare_documents(
                                vs_a, name_a, vs_b, name_b,
                                model_name=st.session_state.get("model_name", "gemini-2.5-flash"),
                                api_key=st.session_state.get("custom_api_key")
                            )
                            st.session_state["comparison_result"] = comp_res
                        except Exception as e:
                            st.error(f"Comparison error: {str(e)}")

                if "comparison_result" in st.session_state:
                    with st.expander("⚖️ Comparison Findings", expanded=True):
                        st.markdown(st.session_state["comparison_result"])
            else:
                st.caption("Upload at least 2 documents to enable side-by-side comparison.")

    # ---------- TAB 4: ANALYTICS & CHUNK BROWSER ----------
    with tab_analytics:
        st.subheader("📊 Document Analytics")
        if not st.session_state.current_doc or st.session_state.current_doc == "__library__":
            st.info("Select a document to view analytics and chunk breakdown.")
        else:
            cur_id = st.session_state.current_doc
            vs = st.session_state.vector_stores.get(cur_id)
            if not vs:
                vs_path = f"vectorstores/{cur_id}"
                if os.path.exists(vs_path):
                    vs = load_vector_store(vs_path, st.session_state.get("custom_api_key"))
                    st.session_state.vector_stores[cur_id] = vs

            if vs:
                analytics_data = compute_document_analytics(vs)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Chunks", analytics_data["total_chunks"])
                    st.metric("Avg Chunk Size", f"{analytics_data['avg_chunk_length']} chars")
                with col2:
                    st.metric("Total Words", f"{analytics_data['total_words']:,}")
                    st.metric("Max Chunk", f"{analytics_data['max_chunk_length']} chars")

                st.subheader("🏷️ Top Keywords")
                kw_chips = " ".join([f"<span class='badge-chip'>{kw['keyword']} ({kw['count']})</span>" for kw in analytics_data["keywords"]])
                st.markdown(kw_chips, unsafe_allow_html=True)

                st.subheader("🔍 Chunk Browser")
                for chunk in analytics_data["chunks_summary"][:10]:
                    with st.expander(f"Chunk #{chunk['chunk_id']} (Page {chunk['page']} | {chunk['char_count']} chars)"):
                        st.text(chunk["full_text"])


# ----------------- MAIN CHAT VIEW -----------------
if not st.session_state.current_doc:
    st.markdown("""
    <div style="text-align: center; padding: 40px 20px;">
        <h1>📄 Welcome to Next-Gen RAG AI Studio</h1>
        <p style="font-size: 1.15rem; color: #888;">
            Ingest PDFs, Word Docs, Web URLs, CSVs & Text. Ask questions, explore citations, generate executive briefs, interactive quizzes, and cross-document comparisons.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📚 **Multi-Format Ingestion**\n\nUpload PDFs, DOCX, TXT, CSV, or live Web URLs with full metadata parsing.")
    with col2:
        st.success("🎯 **Grounded Answers & Citations**\n\nEvery answer cites exact document names, page numbers, and similarity relevance scores.")
    with col3:
        st.warning("✨ **AI Intelligence Studio**\n\nOne-click Executive Summaries, Auto-FAQs, Interactive Quizzes & Side-by-Side Document Comparison.")

    st.markdown("---")
    st.markdown("👈 **Get started by uploading a document or entering a URL in the sidebar!**")

else:
    active_doc_id = st.session_state.current_doc

    # Header section
    if active_doc_id == "__library__":
        active_title = "📚 Entire Document Library (Global Cross-Doc Search)"
        doc_count = len(st.session_state.doc_metadata)
        header_sub = f"Searching across all {doc_count} documents in your workspace"
    else:
        meta = st.session_state.doc_metadata.get(active_doc_id, {})
        active_title = meta.get("filename", "Active Document")
        ftype = meta.get("file_type", "doc").upper()
        pages = meta.get("page_count", 1)
        chunks = meta.get("chunk_count", 0)
        header_sub = f"Format: {ftype} | Pages: {pages} | Chunks: {chunks}"

    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.title(f"💬 {active_title}")
        st.caption(f"{header_sub} • Persona: `{st.session_state.get('persona', 'General Assistant')}`")
    with col_h2:
        # Export / Reset Actions
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            chat_msgs = st.session_state.chats.get(active_doc_id, [])
            md_content = format_chat_as_markdown(active_title, chat_msgs)
            st.download_button(
                "📥 Export MD",
                data=md_content,
                file_name=f"chat_export_{active_doc_id[:6]}.md",
                mime="text/markdown",
                use_container_width=True
            )
        with exp_col2:
            if st.button("🧹 Clear Chat", use_container_width=True):
                st.session_state.chats[active_doc_id] = []
                st.rerun()

    # Interactive Quiz Modal/Section if active for this doc
    if active_doc_id in st.session_state.quiz_data:
        quiz_items = st.session_state.quiz_data[active_doc_id]
        with st.expander("🎯 Interactive Knowledge Assessment Quiz", expanded=True):
            st.markdown("Test your understanding of the document:")
            score = 0
            answered_count = 0

            for q in quiz_items:
                q_id = q["id"]
                q_key = f"quiz_{active_doc_id}_{q_id}"
                st.markdown(f"**Q{q_id}: {q['question']}**")
                selected_option = st.radio(
                    f"Choose an option for Q{q_id}:",
                    q["options"],
                    key=q_key,
                    index=None
                )
                if selected_option is not None:
                    answered_count += 1
                    chosen_idx = q["options"].index(selected_option)
                    if chosen_idx == q["correct_index"]:
                        st.success(f"✅ Correct! {q.get('explanation', '')}")
                        score += 1
                    else:
                        correct_txt = q["options"][q["correct_index"]]
                        st.error(f"❌ Incorrect. Correct answer: **{correct_txt}**\n\n*{q.get('explanation', '')}*")

            if answered_count == len(quiz_items):
                percent = int((score / len(quiz_items)) * 100)
                st.info(f"🏆 Final Score: **{score} / {len(quiz_items)} ({percent}%)**")

    # Suggested Starter Prompt Chips
    if len(st.session_state.chats.get(active_doc_id, [])) == 0:
        st.markdown("💡 **Quick Starter Questions:**")
        chip_col1, chip_col2, chip_col3, chip_col4 = st.columns(4)
        with chip_col1:
            if st.button("📌 Summarize main ideas", use_container_width=True):
                st.session_state.starter_prompt_to_run = "Provide a comprehensive summary of the main ideas in this document."
                st.rerun()
        with chip_col2:
            if st.button("📊 Extract key metrics", use_container_width=True):
                st.session_state.starter_prompt_to_run = "Extract all key metrics, statistics, and quantitative findings from this document."
                st.rerun()
        with chip_col3:
            if st.button("⚠️ What are the risks?", use_container_width=True):
                st.session_state.starter_prompt_to_run = "What are the risks, limitations, or challenges mentioned in this document?"
                st.rerun()
        with chip_col4:
            if st.button("🚀 List action items", use_container_width=True):
                st.session_state.starter_prompt_to_run = "List all actionable recommendations and next steps outlined in this document."
                st.rerun()

    # Render Chat Messages
    messages = st.session_state.chats.get(active_doc_id, [])
    for msg_idx, message in enumerate(messages):
        role = message["role"]
        with st.chat_message(role):
            st.markdown(message["content"])

            # Render Citations if assistant message
            if role == "assistant" and message.get("citations"):
                with st.expander(f"📚 Grounded Citations ({len(message['citations'])} sources)", expanded=False):
                    for c in message["citations"]:
                        src = c.get("source", "Doc")
                        page = c.get("page", 1)
                        score = c.get("relevance_score", 0.0)
                        preview = c.get("content_preview", "")
                        st.markdown(f"""
                        <div class="citation-box">
                            <b>📄 {src}</b> — Page {page} | <i>Relevance: {int(score * 100)}%</i><br>
                            <span style="color: #aaa; font-style: italic;">"{preview}"</span>
                        </div>
                        """, unsafe_allow_html=True)

                # Action Bar (TTS & Translation)
                act_c1, act_c2, act_c3 = st.columns([1, 1, 3])
                with act_c1:
                    tts_key = f"tts_{active_doc_id}_{msg_idx}"
                    if st.button("🔊 Listen", key=f"btn_tts_{msg_idx}"):
                        with st.spinner("Generating audio..."):
                            audio_bytes = generate_tts_audio(message["content"])
                            st.session_state.tts_cache[tts_key] = audio_bytes

                    if tts_key in st.session_state.tts_cache and st.session_state.tts_cache[tts_key]:
                        st.audio(st.session_state.tts_cache[tts_key], format="audio/mp3")

                with act_c2:
                    trans_lang = st.selectbox(
                        "Translate",
                        SUPPORTED_LANGUAGES,
                        key=f"trans_sel_{msg_idx}",
                        label_visibility="collapsed"
                    )
                with act_c3:
                    if st.button(f"Translate to {trans_lang}", key=f"btn_trans_{msg_idx}"):
                        with st.spinner(f"Translating to {trans_lang}..."):
                            try:
                                translated = translate_text(
                                    message["content"],
                                    trans_lang,
                                    model_name=st.session_state.get("model_name", "gemini-2.5-flash"),
                                    api_key=st.session_state.get("custom_api_key")
                                )
                                st.info(f"**{trans_lang} Translation:**\n\n{translated}")
                            except Exception as e:
                                st.error(f"Translation failed: {str(e)}")

    # Prompt Execution Handler
    user_query = st.chat_input(f"Ask a question about {active_title}...")
    if st.session_state.starter_prompt_to_run:
        user_query = st.session_state.starter_prompt_to_run
        st.session_state.starter_prompt_to_run = None

    if user_query:
        # Check API key
        active_key = get_api_key(st.session_state.get("custom_api_key"))
        if not active_key:
            st.error("⚠️ Google API Key not found! Please set `GOOGLE_API_KEY` in `.env` or enter it in the '⚙️ Engine' tab.")
        else:
            # 1. Resolve active vectorstore
            target_vs = None
            if active_doc_id == "__library__":
                # Merge all available vectorstores
                all_vs_list = []
                for d_id in st.session_state.doc_metadata.keys():
                    if d_id not in st.session_state.vector_stores:
                        vs_p = f"vectorstores/{d_id}"
                        if os.path.exists(vs_p):
                            st.session_state.vector_stores[d_id] = load_vector_store(vs_p, active_key)
                    if d_id in st.session_state.vector_stores:
                        all_vs_list.append(st.session_state.vector_stores[d_id])

                if all_vs_list:
                    target_vs = merge_vector_stores(all_vs_list, active_key)
            else:
                if active_doc_id not in st.session_state.vector_stores:
                    vs_p = f"vectorstores/{active_doc_id}"
                    if os.path.exists(vs_p):
                        st.session_state.vector_stores[active_doc_id] = load_vector_store(vs_p, active_key)
                target_vs = st.session_state.vector_stores.get(active_doc_id)

            if target_vs is None:
                st.error("Vector store could not be loaded. Please re-index the document.")
            else:
                # Add user message to state
                if active_doc_id not in st.session_state.chats:
                    st.session_state.chats[active_doc_id] = []
                st.session_state.chats[active_doc_id].append({"role": "user", "content": user_query})

                with st.chat_message("user"):
                    st.markdown(user_query)

                # Generate streaming response
                with st.chat_message("assistant"):
                    try:
                        stream_gen, citations, standalone_q = generate_rag_response_stream(
                            vectorstore=target_vs,
                            query=user_query,
                            chat_history=st.session_state.chats[active_doc_id][:-1],
                            model_name=st.session_state.get("model_name", "gemini-2.5-flash"),
                            persona=st.session_state.get("persona", "General Assistant"),
                            temperature=st.session_state.get("temperature", 0.3),
                            k=st.session_state.get("top_k", 5),
                            search_type=st.session_state.get("retriever_mode", "similarity"),
                            api_key=active_key
                        )

                        full_answer = st.write_stream(stream_gen)

                        # Record assistant response with citations
                        st.session_state.chats[active_doc_id].append({
                            "role": "assistant",
                            "content": full_answer,
                            "citations": citations
                        })

                        # Show citations
                        if citations:
                            with st.expander(f"📚 Grounded Citations ({len(citations)} sources)", expanded=False):
                                for c in citations:
                                    src = c.get("source", "Doc")
                                    page = c.get("page", 1)
                                    score = c.get("relevance_score", 0.0)
                                    preview = c.get("content_preview", "")
                                    st.markdown(f"""
                                    <div class="citation-box">
                                        <b>📄 {src}</b> — Page {page} | <i>Relevance: {int(score * 100)}%</i><br>
                                        <span style="color: #aaa; font-style: italic;">"{preview}"</span>
                                    </div>
                                    """, unsafe_allow_html=True)

                        st.rerun()

                    except Exception as e:
                        st.error(f"Generation error: {str(e)}")