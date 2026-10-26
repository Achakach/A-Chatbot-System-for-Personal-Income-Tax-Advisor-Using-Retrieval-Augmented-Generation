# 📋 Handover Summary: Legal & Tax Document RAG Pipeline

**Project:** Tax Law & Personal Income Tax RAG System  
**Audience:** Antigravity Engineering Team  
**Status:** Pipeline Functional (PoC Completed)  

---

## 1. Executive Summary

ระบบ RAG (Retrieval-Augmented Generation) นี้ถูกพัฒนาขึ้นเพื่อรองรับการสืบค้นและตอบคำถามจาก **เอกสารกฎหมายและประกาศเกี่ยวกับภาษีเงินได้บุคคลธรรมดา** 

ความท้าทายหลักของเอกสารประเภทนี้คือ **ตารางอัตราภาษี/ค่าลดหย่อน** และ **โครงสร้างลำดับชั้นของมาตรากฎหมาย (Hierarchy)** สถาปัตยกรรมนี้จึงเลือกใช้ **`pymupdf4llm`** เพื่อแปลง PDF เป็น Markdown ก่อนทำการ Chunking ซึ่งช่วยรักษาโครงสร้างตารางและหัวข้อกฎหมายได้อย่างแม่นยำ ต่างจากการอ่าน PDF แบบ Plain text ทั่วไป

---

## 2. Architecture & Design Choices

| Component | Technology / Method | Justification |
|---|---|---|
| **Document Parser** | `pymupdf4llm` | แปลง PDF เป็น Markdown พร้อมรักษารูปแบบตาราง (Markdown tables) และหัวข้อระดับต่างๆ (`#`, `##`, `###`) โดยยังคงความเร็วระดับ C-engine |
| **Chunking Strategy** | `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter` | แบ่งตามโครงสร้างหัวข้อกฎหมายก่อน แล้วจึงตัดย่อย (700 chars, 80 overlap) เพื่อคงบริบทเงื่อนไขและข้อยกเว้นของมาตรา |
| **Vector Database** | `ChromaDB` (Persistent) | บันทึกข้อมูลลงดิสก์โลคอล (`./chroma_tax_db`) ติดตั้งง่าย ไม่ต้องมี DB infrastructure ภายนอก |
| **Embedding Engine** | `sentence-transformers` (`BAAI/bge-m3`) | รันเวกเตอร์ภาษาไทย/พหุภาษาขนาด 1024 มิติ รองรับ Multi-lingual, Multi-functionality และประมวลผลกฎหมายได้อย่างแม่นยำสูง |
| **LLM Gateway** | **OpenRouter** (`openai` SDK) | ยืดหยุ่นในการสลับโมเดล เช่น `google/gemini-2.5-flash` หรือ `qwen/qwen-2.5-72b-instruct` โดยใช้ endpoint เดียวกัน |
| **Configuration** | `.env` + `python-dotenv` | จัดการ Environment variables และ API keys อย่างปลอดภัย |

---

## 3. Project Directory Structure

```text
tax-rag-project/
├── .env                     # จัดเก็บ OPENROUTER_API_KEY, AI_MODEL, EMBEDDING_MODEL
├── .env.example             # ตัวอย่าง Configuration template
├── .gitignore               # ละเว้น .env, chroma_tax_db/, chat_history.json, __pycache__/, .venv/
├── requirements.txt         # Dependencies ทั้งหมด
├── HANDOVER_ANTIGRAVITY.md  # เอกสารสรุปการส่งมอบงานฉบับนี้
├── app_rag_openrouter.py    # สคริปต์หลักสำหรับ Ingestion เอกสาร PDF สู่ ChromaDB
├── app_chat_streamlit.py    # เว็บแอปสนทนาโต้ตอบ (Streamlit Web Chat UI)
├── evaluate_rag.py          # สคริปต์ประเมินผล RAG (Latency, Hit Rate, Faithfulness, Hallucination)
├── rag_eval_report.json     # รายงานผล Benchmark ล่าสุด
└── pdf_docs/                # โฟลเดอร์วางไฟล์ PDF กฎหมาย/ภาษี
```

---

## 4. Dependencies (`requirements.txt`)

```text
# PDF Processing
pymupdf>=1.24.0
pymupdf4llm>=0.0.10

# Vector Database & Local Embeddings
chromadb>=0.5.0
sentence-transformers>=2.7.0
torchvision>=0.18.0

# LLM Client (OpenRouter / OpenAI compatible)
openai>=1.30.0

# Chunking utilities
langchain-text-splitters>=0.2.0

# Environment Management
python-dotenv>=1.0.0

# Web Chat Interface
streamlit>=1.35.0
```

---

## 5. Core Codebase (`app_rag_openrouter.py`)

```python
import os
import glob
from dotenv import load_dotenv
import pymupdf4llm
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from openai import OpenAI

# 1. Load Configurations
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
AI_MODEL = os.getenv("AI_MODEL", "google/gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

if not OPENROUTER_API_KEY:
    raise ValueError("❌ ไม่พบ OPENROUTER_API_KEY ในไฟล์ .env")

# 2. OpenRouter Client Setup
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# 3. Vector DB Setup (ChromaDB)
chroma_client = chromadb.PersistentClient(path="./chroma_tax_db")
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)
collection = chroma_client.get_or_create_collection(
    name="tax_law_collection",
    embedding_function=embed_fn
)

def ingest_pdfs(pdf_folder: str = "./pdf_docs"):
    """
    อ่านไฟล์ PDF ทั้งหมดในโฟลเดอร์ แปลงเป็น Markdown, ทำ Chunking และบันทึกลง ChromaDB
    """
    pdf_files = glob.glob(f"{pdf_folder}/*.pdf")
    if not pdf_files:
        print(f"⚠️ ไม่พบไฟล์ PDF ในโฟลเดอร์ {pdf_folder}")
        return

    print(f"📄 กำลังเริ่มประมวลผล PDF {len(pdf_files)} ฉบับ...")
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=80)

    docs, metadatas, ids = [], [], []
    chunk_counter = 0

    for file_path in pdf_files:
        file_name = os.path.basename(file_path)
        print(f"  -> แปลงเนื้อหา: {file_name}")
        md_text = pymupdf4llm.to_markdown(file_path)

        splits = md_splitter.split_text(md_text)
        final_splits = text_splitter.split_documents(splits)

        for doc in final_splits:
            meta = doc.metadata.copy()
            meta["source_file"] = file_name
            docs.append(doc.page_content)
            metadatas.append(meta)
            ids.append(f"{file_name}_{chunk_counter}")
            chunk_counter += 1

    if docs:
        batch_size = 64
        for i in range(0, len(docs), batch_size):
            collection.add(
                documents=docs[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )
        print(f"✅ บันทึกข้อมูลเรียบร้อยทั้งหมด: {len(docs)} chunks")

def ask_tax_question(query: str, top_k: int = 4) -> str:
    """
    ค้นหา Chunks ที่เกี่ยวข้องและส่งให้โมเดลตอบผ่าน OpenRouter
    """
    results = collection.query(query_texts=[query], n_results=top_k)
    retrieved_docs = results["documents"][0]
    retrieved_metas = results["metadatas"][0]

    if not retrieved_docs:
        return "ไม่พบข้อมูลที่เกี่ยวข้องในเอกสาร"

    context_parts = []
    for doc, meta in zip(retrieved_docs, retrieved_metas):
        source = meta.get("source_file", "เอกสารอ้างอิง")
        context_parts.append(f"[{source}]:\n{doc}")
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "คุณคือที่ปรึกษาผู้เชี่ยวชาญด้านกฎหมายภาษีอากร "
        "หน้าที่ของคุณคือตอบคำถามตาม 'ข้อมูลอ้างอิง' ที่ให้มาเท่านั้น "
        "หากไม่มีข้อมูลให้ระบุว่า 'ไม่พบข้อมูลในเอกสาร' ห้ามคาดเดาหรือแต่งคำตอบเอง"
    )

    response = client.chat.completions.create(
        extra_headers={"HTTP-Referer": "http://localhost:3000", "X-Title": "Tax-RAG"},
        model=AI_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"ข้อมูลอ้างอิง:\n{context}\n\nคำถาม: {query}"}
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    os.makedirs("./pdf_docs", exist_ok=True)
    
    # รันการย่อยเอกสาร (Uncomment เมื่อต้องการ Re-index)
    # ingest_pdfs("./pdf_docs")
    
    # ตัวอย่างการทดสอบถามคำถาม
    sample_q = "สรุปเงื่อนไขและเพดานการหักลดหย่อนภาษีคู่สมรส"
    print(ask_tax_question(sample_q))
```

---

## 6. Recommended Next Steps for Antigravity

1. **Embedding Model Upgrade (High Priority for Thai Language):**
   * โมเดล `all-MiniLM-L6-v2` มีข้อจำกัดด้านภาษาไทย แนะนำให้อัปเกรดเป็น:
     * `BAAI/bge-m3` (รองรับ Multi-lingual, Chunking ขนาดใหญ่ และทำ Dense/Sparse hybrid ได้)
     * หรือ `intfloat/multilingual-e5-large`
2. **Handling Scanned PDF:**
   * โค้ดปัจจุบันรองรับ Digital Text PDF หากมีเอกสารฉบับสแกน (รูปภาพ) ควรวาง Ingestion fallback ไปที่ **Docling** หรือโมเดล OCR เช่น Tesseract / Surun
3. **Hybrid Search (BM25 + Dense Vector):**
   * ในการค้นหากฎหมาย การจับ Keyword ตรงตัว (เช่น "มาตรา 47(1)(ช)") สำคัญมาก การผสาน BM25 เข้ากับ Cosine Similarity จะเพิ่ม Hit-rate ของข้อกฎหมายได้อย่างชัดเจน
4. **API & Interface Wrapping:**
   * พัฒนา REST API ครอบฟังก์ชัน `ask_tax_question` ด้วย **FastAPI** เพื่อให้ Frontend หรือ Microservices อื่นๆ เรียกใช้งานได้สะดวก
