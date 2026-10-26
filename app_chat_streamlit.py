import os
import re
import json
import hmac
import streamlit as st
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI, AuthenticationError, RateLimitError

# Persistent Chat History Storage
HISTORY_FILE = "./chat_history.json"

def load_chat_history():
    if PERSIST_CHAT_HISTORY and os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return data
        except Exception:
            pass
    return [
        {
            "role": "assistant",
            "content": "สวัสดีครับ! ผมคือผู้ช่วยตอบคำถามด้านกฎหมายและภาษีเงินได้บุคคลธรรมดา (Thai PIT RAG) อ้างอิงจากประมวลรัษฎากร กฎกระทรวง และคู่มือทางการของกรมสรรพากร คุณสามารถสอบถามเรื่องอัตราภาษี ค่าลดหย่อน หรือเงื่อนไขต่างๆ ได้เลยครับ ⚖️"
        }
    ]

def save_chat_history(messages):
    # ปิดการบันทึกลงไฟล์เมื่อ deploy ให้หลายคนใช้ ประวัติจะอยู่เฉพาะใน session ของผู้ใช้แต่ละคน
    if not PERSIST_CHAT_HISTORY:
        return
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# 1. Page Configuration
st.set_page_config(
    page_title="AI ที่ปรึกษาภาษีเงินได้บุคคลธรรมดา (Thai Tax RAG)",
    page_icon="⚖️",
    layout="wide"
)

# ตารางกว้างในส่วนเอกสารอ้างอิงให้เลื่อนดูในกรอบของตัวเอง แทนการดันทั้งหน้าให้เลื่อนแนวนอน
st.markdown(
    "<style>[data-testid='stExpander'] [data-testid='stMarkdownContainer'] { overflow-x: auto; }</style>",
    unsafe_allow_html=True,
)

# 2. Load Environment Variables
load_dotenv(override=True)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
# fp16 ใช้หน่วยความจำราว 1.9GB แทน 3GB+ (จำเป็นบน Streamlit Community Cloud ที่จำกัด ~2.7GB)
# ผลการค้นหาเหมือน fp32 (ทดสอบแล้ว: top-4 ตรงกันทุกคำถาม คะแนนต่างกันไม่เกิน 0.0001)
EMBEDDING_DTYPE = os.getenv("EMBEDDING_DTYPE", "float16")
# branch หลักของ bge-m3 มีแค่ pytorch_model.bin ซึ่งต้องโหลดเป็น fp32 ก่อนแปลง (หน่วยความจำพุ่ง ~3.7GB)
# จึงใช้ revision ที่มี model.safetensors (refs/pr/130) เพื่อโหลดเป็น fp16 ได้โดยตรง (peak ~2.5GB)
DEFAULT_EMBEDDING_REVISIONS = {"BAAI/bge-m3": "9a0624b896d81da7492a910ffa53731274b6cf3d"}
EMBEDDING_REVISION = os.getenv("EMBEDDING_REVISION") or DEFAULT_EMBEDDING_REVISIONS.get(EMBEDDING_MODEL)
# ตั้งเป็น false เมื่อ deploy ให้คนอื่นใช้ เพื่อไม่ให้ทุกคนเห็นประวัติแชตของกันและกัน
PERSIST_CHAT_HISTORY = os.getenv("PERSIST_CHAT_HISTORY", "true").strip().lower() in ("1", "true", "yes")
# หากตั้งค่า ผู้ใช้ต้องใส่รหัสผ่านก่อนใช้งาน (ป้องกันคนอื่นใช้โควตา API ของเรา)
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

if APP_PASSWORD and not st.session_state.get("authenticated"):
    st.title("⚖️ ระบบสืบค้นและตอบคำถามกฎหมายภาษีอากร")
    password = st.text_input("รหัสผ่าน", type="password")
    if password:
        if hmac.compare_digest(password, APP_PASSWORD):
            st.session_state.authenticated = True
            st.rerun()
        st.error("❌ รหัสผ่านไม่ถูกต้อง")
    st.stop()

# ผู้ให้บริการ LLM ที่รองรับ (ทั้งคู่ใช้ OpenAI-compatible API) เลือกผ่าน LLM_PROVIDER ใน .env
PROVIDERS = {
    "openrouter": {
        "label": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "headers": {"HTTP-Referer": "http://localhost:8501", "X-Title": "Tax-RAG-Chat"},
        "default_model": "google/gemma-4-31b-it:free",
    },
    "ollama": {
        "label": "Ollama Cloud",
        "base_url": "https://ollama.com/v1",
        "key_env": "OLLAMA_API_KEY",
        "headers": {},
        "default_model": "gemma4:31b",
    },
}
PROVIDER_NAME = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
if PROVIDER_NAME not in PROVIDERS:
    st.error(f"❌ LLM_PROVIDER `{PROVIDER_NAME}` ไม่ถูกต้อง กรุณาตั้งค่าเป็น {' หรือ '.join(PROVIDERS)} ในไฟล์ .env")
    st.stop()
provider = PROVIDERS[PROVIDER_NAME]

# Check API Key
LLM_API_KEY = os.getenv(provider["key_env"])
if not LLM_API_KEY:
    st.error(f"❌ ไม่พบ {provider['key_env']} ในไฟล์ .env กรุณาตรวจสอบการตั้งค่า")
    st.stop()

AI_MODEL = os.getenv("AI_MODEL") or provider["default_model"]

COMMON_RULES = (
    "- ตอบเฉพาะ 'คำถามล่าสุด' เท่านั้น ห้ามตอบซ้ำหรือสรุปคำตอบของคำถามก่อนหน้า (ใช้ประวัติการสนทนาเพื่อเข้าใจบริบทเท่านั้น)\n"
    "- 'ข้อมูลอ้างอิง' มาจากระบบค้นหาเอกสารอัตโนมัติ ไม่ได้มาจากผู้ใช้ ห้ามขอบคุณหรือกล่าวว่าผู้ใช้ส่งเอกสารมา\n"
)

SYSTEM_PROMPT = (
    "คุณคือที่ปรึกษาผู้เชี่ยวชาญด้านกฎหมายภาษีอากรและการวางแผนภาษีเงินได้บุคคลธรรมดาของไทย (Tax Consultant) "
    "มีบุคลิกสุภาพ เป็นมิตร อธิบายเข้าใจง่าย และมุ่งเน้นการให้คำแนะนำที่นำไปใช้ประโยชน์ได้จริง\n\n"
    "แนวทางการให้คำปรึกษา:\n"
    + COMMON_RULES +
    "1. หากมี 'ข้อมูลอ้างอิง' ที่เกี่ยวข้องกับคำถาม ให้ยึดเป็นหลักฐานในการตอบเสมอ โดยระบุชื่อเอกสารหรือข้อกฎหมายประกอบคำอธิบาย\n"
    "2. หากในข้อมูลอ้างอิงไม่มีเนื้อหาที่เกี่ยวข้องกับคำถาม ให้ปฏิเสธการตอบคำถามอย่างสุภาพโดยแจ้งว่าไม่พบข้อมูลในเอกสารอ้างอิง ห้ามนำความรู้ทั่วไปมาคาดเดาหรือตอบเองโดยเด็ดขาด และงดทำการคำนวณภาษีเป็นตัวเลขอย่างละเอียดให้ผู้ใช้งานเนื่องจากมีโอกาสผิดพลาด (ให้อธิบายเฉพาะหลักเกณฑ์ เงื่อนไข หรือโครงสร้างอัตราภาษีตามที่ระบุในเอกสารอ้างอิงเท่านั้น)\n"
    "3. ชี้แจงเงื่อนไข ข้อยกเว้น และข้อควรระวังให้ผู้เสียภาษีทราบอย่างรอบคอบ พร้อมแนะนำให้ตรวจสอบเอกสารหลักฐานทางการกับกรมสรรพากรเพิ่มเติมหากเป็นกรณีเฉพาะตัว\n"
    "4. ระบุแหล่งที่มาท้ายข้อความที่มาจากข้อมูลอ้างอิงในรูปแบบ [ชื่อไฟล์ หน้า N]\n"
    "5. ปิดท้ายคำตอบทุกครั้งด้วยหัวข้อ '**:books: สรุปแหล่งอ้างอิง**' แล้วแสดงเป็นรายการหัวข้อย่อย:\n"
    "   - เอกสารที่ใช้ตอบจริงเท่านั้น ไฟล์ละ 1 บรรทัด ไม่ซ้ำกัน พร้อมเลขหน้าที่ใช้ เช่น '- 08_PND90_Guide_2568.pdf หน้า 7, 9' "
    "(ไฟล์ที่ไม่มีเลขหน้าให้ใส่แค่ชื่อไฟล์) ตามด้วยสรุปสั้นๆ ว่าใช้ข้อมูลเรื่องอะไรจากไฟล์นั้น\n"
    "   - หากไม่พบข้อมูลในเอกสารอ้างอิงที่เกี่ยวข้อง ให้เขียนว่า '- ไม่พบเอกสารอ้างอิงที่เกี่ยวข้อง'"
)

REWRITE_PROMPT = (
    "หน้าที่ของคุณคือเขียนคำถามล่าสุดของผู้ใช้ใหม่ให้เป็นคำค้นหาที่สมบูรณ์ในตัวเอง (standalone) "
    "สำหรับค้นหาเอกสารกฎหมายภาษีเงินได้บุคคลธรรมดา โดยเติมบริบทที่จำเป็นจากบทสนทนาก่อนหน้า "
    "เช่น หัวข้อ ประเภทเงินได้ ประเภทค่าลดหย่อน หรือเลขมาตรา\n"
    "ตอบกลับเป็นคำค้นหาภาษาไทยเพียงบรรทัดเดียว ห้ามตอบคำถาม ห้ามอธิบายเพิ่ม "
    "หากคำถามสมบูรณ์ในตัวเองอยู่แล้ว ให้ตอบคำถามเดิมกลับมา"
)

# 3. Cache ChromaDB & Embedding Model to prevent reloading on rerun
@st.cache_resource
def get_vector_collection():
    chroma_client = chromadb.PersistentClient(path="./chroma_tax_db")
    # ไม่ fallback ไปโมเดลอื่นแบบเงียบๆ เพราะมิติเวกเตอร์จะไม่ตรงกับฐานข้อมูล
    model_options = {"model_kwargs": {"dtype": EMBEDDING_DTYPE}}
    if EMBEDDING_REVISION:
        model_options["revision"] = EMBEDDING_REVISION
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        **model_options
    )
    collection = chroma_client.get_or_create_collection(
        name="tax_law_collection",
        embedding_function=embed_fn
    )
    return collection

# Cache แยกตามผู้ให้บริการและ API Key เพื่อให้แก้ไข key ใน .env แล้วมีผลทันทีโดยไม่ต้องรีสตาร์ทแอป
@st.cache_resource
def get_llm_client(base_url: str, api_key: str):
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

try:
    collection = get_vector_collection()
except Exception as e:
    st.error(
        f"❌ โหลด Embedding Model `{EMBEDDING_MODEL}` ไม่สำเร็จ: {e}\n\n"
        "กรุณาตรวจสอบค่า EMBEDDING_MODEL ในไฟล์ .env และการติดตั้ง sentence-transformers"
    )
    st.stop()

client = get_llm_client(provider["base_url"], LLM_API_KEY)

def get_distance_space() -> str:
    """อ่านชนิด distance ของ Collection (cosine / l2 / ip)"""
    try:
        hnsw = (collection.configuration or {}).get("hnsw") or {}
        if hnsw.get("space"):
            return hnsw["space"]
    except Exception:
        pass
    return (collection.metadata or {}).get("hnsw:space", "l2")

def to_relevance(distance: float) -> float:
    """แปลง distance เป็นคะแนนความเกี่ยวข้อง 0-1 (embedding ของ bge-m3 ถูก normalize แล้ว)"""
    if get_distance_space() in ("cosine", "ip"):
        score = 1 - distance
    else:
        # squared L2 ของเวกเตอร์ที่ normalize แล้ว = 2 - 2*cos
        score = 1 - distance / 2
    return max(0.0, min(1.0, score))

def source_label(src: dict) -> str:
    if src.get("page"):
        return f"{src['file']} หน้า {src['page']}"
    return src["file"]

def retrieve(query: str, top_k: int) -> list[dict]:
    """ค้นหา Chunks ที่เกี่ยวข้อง คืนค่าเป็นรายการ {file, page, score, text}"""
    results = collection.query(query_texts=[query], n_results=top_k)
    docs = results["documents"][0] if results.get("documents") else []
    metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
    dists = results["distances"][0] if results.get("distances") else [None] * len(docs)

    sources = []
    for doc, meta, dist in zip(docs, metas, dists):
        meta = meta or {}
        sources.append({
            "file": meta.get("source_file", "เอกสารอ้างอิง"),
            "page": meta.get("page"),
            "score": to_relevance(dist) if dist is not None else None,
            "text": doc,
        })
    return sources

def rewrite_query(history: list[dict], question: str, model: str) -> str:
    """เขียนคำถามต่อเนื่องใหม่ให้เป็นคำค้นหาที่สมบูรณ์ในตัวเอง (คืนคำถามเดิมหากไม่มีบริบทหรือเกิดข้อผิดพลาด)"""
    turns = [
        m for m in history
        if m["role"] in ("user", "assistant") and not m.get("error")
    ][-4:]
    if not any(m["role"] == "user" for m in turns):
        return question

    transcript = "\n".join(
        f"{'ผู้ใช้' if m['role'] == 'user' else 'ผู้ช่วย'}: {m['content'][:500]}"
        for m in turns
    )
    try:
        response = client.chat.completions.create(
            extra_headers=provider["headers"],
            model=model,
            temperature=0.0,
            max_tokens=300,
            messages=[
                {"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": f"บทสนทนาก่อนหน้า:\n{transcript}\n\nคำถามล่าสุด: {question}\n\nคำค้นหา:"},
            ],
        )
        lines = (response.choices[0].message.content or "").strip().splitlines()
        rewritten = lines[0].strip() if lines else ""
        return rewritten or question
    except Exception:
        return question

TABLE_SEPARATOR = re.compile(r"^\|\s*:?-{3,}")

def to_markdown_excerpt(text: str) -> str:
    """เตรียมข้อความของท่อนเอกสารให้แสดงเป็น Markdown ในส่วนเอกสารอ้างอิง"""
    # หัวข้อ (#) ใหญ่เกินไปในกล่องอ้างอิง จึงแสดงเป็นตัวหนาแทน
    text = re.sub(
        r"(?m)^#{1,6}\s*(.+)$",
        lambda m: f"**{m.group(1).replace('**', '').strip()}**",
        text,
    )
    # ตัวหนาที่มีช่องว่างติดเครื่องหมาย (เช่น "**300,000 **") Markdown จะไม่แสดงเป็นตัวหนา
    text = re.sub(r"\*\*[ \t]*([^*\n]+?)[ \t]*\*\*", r"**\1**", text)
    # เครื่องหมาย $ ใน Streamlit markdown คือการเริ่มสูตร LaTeX
    text = text.replace("$", "\\$")

    # ตารางที่ถูกตัดแบ่งกลางท่อนจะไม่มีแถวหัวตาราง จึงไม่แสดงเป็นตาราง
    # เติมแถวหัวว่างให้ และเว้นบรรทัดก่อนตารางเพื่อให้ Markdown แยกตารางออกจากย่อหน้า
    out, block = [], []

    def flush_table():
        if not block:
            return
        if out and out[-1].strip():
            out.append("")
        if not any(TABLE_SEPARATOR.match(line.strip()) for line in block):
            cols = max(line.strip().strip("|").count("|") + 1 for line in block)
            out.append("|" + " |" * cols)
            out.append("|" + "---|" * cols)
        out.extend(block)
        block.clear()

    for line in text.split("\n"):
        if line.strip().startswith("|"):
            block.append(line)
        else:
            flush_table()
            out.append(line)
    flush_table()
    return "\n".join(out)

def render_sources(sources: list[dict]):
    """แสดงเอกสารอ้างอิงจัดกลุ่มตามไฟล์ พร้อมเลขหน้าและคะแนนความเกี่ยวข้อง"""
    grouped: dict[str, list[dict]] = {}
    for src in sources:
        grouped.setdefault(src["file"], []).append(src)

    with st.expander(f"📄 ดูเอกสารอ้างอิง ({len(grouped)} ฉบับ, {len(sources)} ท่อน)"):
        for file_name, chunks in grouped.items():
            st.markdown(f"**📌 เอกสาร:** `{file_name}`")
            for src in chunks:
                details = []
                if src.get("page"):
                    details.append(f"หน้า {src['page']}")
                if src.get("score") is not None:
                    details.append(f"ความเกี่ยวข้อง {src['score']:.0%}")
                if details:
                    st.caption(" · ".join(details))
                with st.container(border=True):
                    st.markdown(to_markdown_excerpt(src["text"]))
            st.markdown("---")

# 4. Sidebar Controls
with st.sidebar:
    st.title("⚙️ ตั้งค่าระบบ")
    st.markdown("---")

    # ChromaDB Status
    total_chunks = collection.count()
    st.subheader("📊 สถานะฐานข้อมูล")
    st.metric(label="จำนวน Chunks ในระบบ", value=f"{total_chunks:,} Chunks")

    if total_chunks == 0:
        st.warning("⚠️ ฐานข้อมูลยังว่างเปล่า! กรุณารัน `python app_rag_openrouter.py` เพื่อ Ingest เอกสาร PDF ก่อน")
    else:
        st.success("✅ ฐานข้อมูลพร้อมใช้งาน")

    st.markdown("---")
    st.subheader("🤖 การตั้งค่าโมเดล")

    top_k = st.slider(
        "จำนวน Chunks ที่ใช้ค้นหา (Top-K):",
        min_value=1,
        max_value=10,
        value=4,
        help="จำนวนท่อนข้อความกฎหมายที่ดึงมาใช้อ้างอิง"
    )

    temperature = st.slider(
        "Temperature (ความแน่นอน):",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="ค่าต่ำ (เช่น 0.0) ตอบตรงและคงที่ ค่าสูงขึ้นจะอธิบายยืดหยุ่นและเป็นธรรมชาติมากขึ้น"
    )

    st.markdown("---")
    if st.button("🗑️ ล้างประวัติการสนทนา", use_container_width=True):
        if PERSIST_CHAT_HISTORY and os.path.exists(HISTORY_FILE):
            try:
                os.remove(HISTORY_FILE)
            except Exception:
                pass
        st.session_state.messages = load_chat_history()
        st.rerun()

# 5. Initialize Chat History (Loaded from disk to persist across refresh)
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# 6. Main UI Header
st.title("⚖️ ระบบสืบค้นและตอบคำถามกฎหมายภาษีเงินได้บุคคลธรรมดา")
st.caption(f"ขับเคลื่อนด้วย RAG (ChromaDB + {provider['label']} · {AI_MODEL})")

# รูปประจำตัวในแชต (ใช้ Material icon ของ Streamlit)
AVATARS = {"assistant": ":material/balance:", "user": ":material/person:"}

# 7. Render Conversation History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=AVATARS.get(msg["role"])):
        if msg.get("error"):
            st.error(msg["content"])
        else:
            st.markdown(msg["content"])
        if msg.get("search_query"):
            st.caption(f"🔎 ค้นหาด้วย: {msg['search_query']}")
        if msg.get("sources"):
            render_sources(msg["sources"])

# 8. User Input & Query Execution
if user_query := st.chat_input("พิมพ์คำถามภาษี เช่น 'สรุปเงื่อนไขและเพดานการหักลดหย่อนภาษีคู่สมรส'..."):
    # Append & display User message
    st.session_state.messages.append({"role": "user", "content": user_query})
    save_chat_history(st.session_state.messages)
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(user_query)

    # ประวัติก่อนคำถามปัจจุบัน (ไม่รวมข้อความแจ้งข้อผิดพลาด)
    prior_messages = [
        m for m in st.session_state.messages[:-1]
        if m["role"] in ("user", "assistant") and not m.get("error")
    ]

    # Assistant Response
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        if total_chunks == 0:
            st.info("💡 หมายเหตุ: ฐานข้อมูล ChromaDB ยังไม่มีเอกสาร (กำลังให้คำปรึกษาตามหลักการภาษีทั่วไป)")

        search_query = user_query
        sources_data = []
        if total_chunks > 0:
            with st.spinner("🔍 กำลังค้นหาข้อกฎหมายและเอกสารที่เกี่ยวข้อง..."):
                # เขียนคำถามต่อเนื่องใหม่ให้สมบูรณ์ในตัวเองก่อนค้นหา (คำถามแรกจะใช้คำถามเดิม)
                search_query = rewrite_query(prior_messages, user_query, AI_MODEL)
                try:
                    sources_data = retrieve(search_query, top_k)
                except Exception as e:
                    st.warning(
                        f"⚠️ ค้นหาเอกสารไม่สำเร็จ: {e}\n\n"
                        "หาก Embedding Model ไม่ตรงกับฐานข้อมูล ให้รัน `python app_rag_openrouter.py` เพื่อสร้างดัชนีใหม่"
                    )

        # Format Context with file name and page
        context_parts = [f"[{source_label(src)}]:\n{src['text']}" for src in sources_data]
        context = "\n\n---\n\n".join(context_parts) if context_parts else "ไม่มีเอกสารอ้างอิงที่ตรงตัวโดยตรง"

        # Format conversation history for context (last 4 turns)
        llm_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for past_msg in prior_messages[-4:]:
            llm_messages.append({"role": past_msg["role"], "content": past_msg["content"]})

        # Current Turn with Context
        llm_messages.append({
            "role": "user",
            "content": f"ข้อมูลอ้างอิงจากคลังเอกสาร:\n{context}\n\nคำถามหรือข้อปรึกษา: {user_query}"
        })

        # Stream response from the selected provider
        def generate_stream():
            stream = client.chat.completions.create(
                extra_headers=provider["headers"],
                model=AI_MODEL,
                temperature=temperature,
                messages=llm_messages,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        # Stream output live to UI
        error_text = None
        full_response = ""
        try:
            full_response = st.write_stream(generate_stream())
        except AuthenticationError:
            error_text = (
                f"⚠️ {provider['label']} ปฏิเสธ API Key (401) กรุณาตรวจสอบ {provider['key_env']} ในไฟล์ .env "
                "แล้วส่งคำถามใหม่อีกครั้ง (ไม่ต้องรีสตาร์ทแอป)"
            )
        except RateLimitError:
            error_text = (
                f"⚠️ {provider['label']} แจ้งว่าใช้งานโมเดล `{AI_MODEL}` เกินโควตา (429) "
                "กรุณารอสักครู่แล้วลองใหม่อีกครั้ง"
            )
        except Exception as e:
            error_text = f"⚠️ เรียกโมเดล `{AI_MODEL}` ไม่สำเร็จ: {e}"

        if not error_text and not str(full_response).strip():
            error_text = (
                "⚠️ โมเดลไม่ได้ส่งคำตอบกลับมา (อาจเป็นโมเดลแบบ reasoning) "
                "กรุณาลองใหม่ หรือเปลี่ยน AI_MODEL ในไฟล์ .env"
            )

        if error_text:
            # บันทึกข้อผิดพลาดไว้ในประวัติ แต่จะไม่ถูกส่งให้โมเดลในรอบถัดไป
            st.error(error_text)
            st.session_state.messages.append({"role": "assistant", "content": error_text, "error": True})
        else:
            if search_query != user_query:
                st.caption(f"🔎 ค้นหาด้วย: {search_query}")
            if sources_data:
                render_sources(sources_data)
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources_data,
                "search_query": search_query if search_query != user_query else None,
            })
        save_chat_history(st.session_state.messages)
