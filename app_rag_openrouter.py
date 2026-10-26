import os
import glob
from dotenv import load_dotenv
import pymupdf4llm
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# 1. Load Configurations
load_dotenv(override=True)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# 2. Vector DB Setup (ChromaDB)
chroma_client = chromadb.PersistentClient(path="./chroma_tax_db")

# ไม่ fallback ไปโมเดลอื่นแบบเงียบๆ เพราะมิติเวกเตอร์จะไม่ตรงกับฐานข้อมูล
try:
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
except Exception as e:
    raise RuntimeError(f"❌ โหลด Embedding Model '{EMBEDDING_MODEL}' ไม่สำเร็จ: {e}") from e

# ใช้ cosine distance เพื่อให้คะแนนความเกี่ยวข้อง = 1 - distance
COLLECTION_METADATA = {"hnsw:space": "cosine"}

# Collection ที่มีอยู่แล้วเปลี่ยน distance function ไม่ได้ จึงกำหนด metadata เฉพาะตอนสร้างใหม่
try:
    collection = chroma_client.get_collection(
        name="tax_law_collection",
        embedding_function=embed_fn
    )
except Exception:
    collection = chroma_client.create_collection(
        name="tax_law_collection",
        embedding_function=embed_fn,
        metadata=COLLECTION_METADATA
    )

SUPPORTED_PATTERNS = ("*.pdf", "*.md")

def load_page_texts(file_path: str) -> list[tuple[int | None, str]]:
    """
    คืนรายการ (เลขหน้า, ข้อความ Markdown) ของไฟล์
    - PDF: แปลงทีละหน้าด้วย pymupdf4llm
    - Markdown: ไม่มีเลขหน้า จึงคืนข้อความทั้งไฟล์เป็นก้อนเดียว (เลขหน้า = None)
    """
    if file_path.lower().endswith(".md"):
        with open(file_path, "r", encoding="utf-8-sig") as f:
            return [(None, f.read())]

    # page_chunks=True คืนค่า 1 dict ต่อ 1 หน้า เรียงตามลำดับหน้า
    pages = pymupdf4llm.to_markdown(file_path, page_chunks=True)
    return [(page_num, page.get("text", "")) for page_num, page in enumerate(pages, 1)]

def ingest_pdfs(pdf_folder: str = "./pdf_docs", reset: bool = True):
    """
    อ่านไฟล์ PDF และ Markdown ทั้งหมดในโฟลเดอร์, ทำ Chunking และบันทึกลง ChromaDB (PDF มีเลขหน้ากำกับ)
    :param reset: หากเป็น True จะลบ Collection เก่าทิ้งก่อน เพื่อป้องกันปัญหามิติเวกเตอร์ไม่ตรงกันเมื่อเปลี่ยนโมเดล
    """
    global collection

    if reset:
        try:
            chroma_client.delete_collection("tax_law_collection")
            print("🗑️ ล้าง Collection เดิมใน ChromaDB เรียบร้อยแล้ว เพื่อเริ่มสร้างใหม่ให้ตรงกับโมเดลปัจจุบัน")
        except Exception:
            pass
        collection = chroma_client.create_collection(
            name="tax_law_collection",
            embedding_function=embed_fn,
            metadata=COLLECTION_METADATA
        )

    doc_files = sorted(
        path for pattern in SUPPORTED_PATTERNS for path in glob.glob(f"{pdf_folder}/{pattern}")
    )
    if not doc_files:
        print(f"⚠️ ไม่พบไฟล์ PDF หรือ Markdown ในโฟลเดอร์ {pdf_folder}")
        return

    print(f"📄 กำลังเริ่มประมวลผลเอกสาร {len(doc_files)} ฉบับด้วยโมเดล: {EMBEDDING_MODEL}...")
    # strip_headers=False: เก็บบรรทัดหัวข้อไว้ในเนื้อหา chunk เพื่อให้รู้ว่าข้อความอยู่ในหัวข้อใด
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")],
        strip_headers=False
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=80)

    docs, metadatas, ids = [], [], []
    chunks_per_file = {}

    for idx, file_path in enumerate(doc_files, 1):
        file_name = os.path.basename(file_path)
        print(f"  [{idx}/{len(doc_files)}] -> แปลงเนื้อหา: {file_name}")
        file_chunks = 0

        for page_num, page_text in load_page_texts(file_path):
            if not page_text.strip():
                continue

            splits = md_splitter.split_text(page_text)
            final_splits = text_splitter.split_documents(splits)

            for chunk_idx, doc in enumerate(final_splits):
                meta = doc.metadata.copy()
                meta["source_file"] = file_name
                # ChromaDB ไม่รับค่า None ใน metadata จึงใส่ page เฉพาะไฟล์ที่มีเลขหน้า
                if page_num is not None:
                    meta["page"] = page_num
                    ids.append(f"{file_name}_p{page_num}_{chunk_idx}")
                else:
                    ids.append(f"{file_name}_{chunk_idx}")
                docs.append(doc.page_content)
                metadatas.append(meta)
                file_chunks += 1

        chunks_per_file[file_name] = file_chunks

    if docs:
        print(f"💾 กำลังคำนวณ Embeddings ({EMBEDDING_MODEL}) และบันทึก {len(docs)} chunks ลง ChromaDB...")
        batch_size = 64
        for i in range(0, len(docs), batch_size):
            collection.upsert(
                documents=docs[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )
            print(f"  -> บันทึกแล้ว {min(i + batch_size, len(docs))}/{len(docs)} chunks...")
        print(f"✅ บันทึกข้อมูลเรียบร้อยทั้งหมด: {len(docs)} chunks ลง ChromaDB")

    print("\n📊 จำนวน Chunks ต่อไฟล์:")
    for file_name, count in chunks_per_file.items():
        if count > 0:
            warn = ""
        elif file_name.lower().endswith(".md"):
            warn = "  ⚠️ ไม่มีข้อความ (ไฟล์ว่าง)"
        else:
            warn = "  ⚠️ ไม่มีข้อความ (อาจเป็นไฟล์สแกน ต้องใช้ OCR)"
        print(f"  {count:>5}  {file_name}{warn}")

if __name__ == "__main__":
    os.makedirs("./pdf_docs", exist_ok=True)
    print("=" * 60)
    print(f"🚀 เริ่มต้น Ingest เอกสาร PDF/Markdown ด้วยโมเดล: {EMBEDDING_MODEL}")
    print("=" * 60)
    ingest_pdfs("./pdf_docs", reset=True)
    print("=" * 60)
    print(f"🎉 เสร็จสิ้นเรียบร้อย! รวมมีข้อมูลทั้งหมดในฐานข้อมูล {collection.count()} chunks")
    print("=" * 60)
