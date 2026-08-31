import os
import io
import json
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from gtts import gTTS
except ImportError:
    gTTS = None


def save_uploaded_file(uploaded_file, destination_dir: str = "temp") -> str:
    """Save an uploaded Streamlit file to a local destination directory preserving extension."""
    dest_path = Path(destination_dir)
    dest_path.mkdir(exist_ok=True)

    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, dir=destination_dir, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def cleanup_temp_files(destination_dir: str = "temp") -> None:
    """Safely clear temp files."""
    dest_path = Path(destination_dir)
    if dest_path.exists():
        for file in dest_path.glob("*"):
            try:
                if file.is_file():
                    file.unlink()
            except Exception:
                pass


def delete_document_vectorstore(doc_id: str, vector_dir: str = "vectorstores") -> bool:
    """Delete a document vectorstore directory from disk."""
    folder = Path(vector_dir) / doc_id
    if folder.exists() and folder.is_dir():
        try:
            shutil.rmtree(folder)
            return True
        except Exception:
            return False
    return False


def generate_tts_audio(text: str, lang: str = "en") -> Optional[bytes]:
    """Generate TTS audio MP3 bytes from text using gTTS."""
    if not gTTS or not text.strip():
        return None
    try:
        # Strip markdown symbols for clearer speech
        clean_text = text.replace("#", "").replace("*", "").replace("`", "").strip()
        # Truncate if exceptionally long (gTTS limit)
        if len(clean_text) > 3000:
            clean_text = clean_text[:3000] + "..."

        tts = gTTS(text=clean_text, lang=lang, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception:
        return None


def format_chat_as_markdown(doc_name: str, messages: List[Dict[str, Any]]) -> str:
    """Export chat history as a clean Markdown report with timestamps and citations."""
    md = f"# 📄 Chat History: {doc_name}\n\n"
    md += f"*Exported from RAG Chat App*\n\n---\n\n"

    for msg in messages:
        role = "**User** 👤" if msg.get("role") == "user" else "**AI Assistant** 🤖"
        content = msg.get("content", "")
        md += f"### {role}\n\n{content}\n\n"

        citations = msg.get("citations", [])
        if citations:
            md += "#### 📚 Sources Cited:\n"
            for c in citations:
                source = c.get("source", "Doc")
                page = c.get("page", 1)
                score = c.get("relevance_score", "N/A")
                preview = c.get("content_preview", "")
                md += f"- **[{source} - Page {page}]** (Relevance: {score})\n  > *\"{preview}\"*\n"
            md += "\n"

        md += "---\n\n"
    return md


def format_chat_as_json(doc_name: str, messages: List[Dict[str, Any]]) -> str:
    """Export chat history as JSON."""
    data = {
        "document_name": doc_name,
        "message_count": len(messages),
        "messages": messages
    }
    return json.dumps(data, indent=2)