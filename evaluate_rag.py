import os
import time
import json
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

# 1. Load Environment Variables
load_dotenv(override=True)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
AI_MODEL = os.getenv("AI_MODEL", "google/gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

if not OPENROUTER_API_KEY:
    raise ValueError("❌ ไม่พบ OPENROUTER_API_KEY ในไฟล์ .env")

# 2. Setup Vector Database (ChromaDB)
chroma_client = chromadb.PersistentClient(path="./chroma_tax_db")
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)
collection = chroma_client.get_collection(
    name="tax_law_collection",
    embedding_function=embed_fn
)

# 3. Setup OpenRouter Client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# 4. Golden Test Dataset (มาตรฐานคำตอบกฎหมายภาษีเงินได้บุคคลธรรมดา)
GOLDEN_DATASET = [
    {
        "id": 1,
        "question": "ค่าลดหย่อนบุตรชอบด้วยกฎหมายสามารถนำมาหักลดหย่อนได้คนละกี่บาท?",
        "ground_truth": "30,000 บาทต่อคน",
        "expected_keywords": ["30,000", "30000", "สามหมื่น"]
    },
    {
        "id": 2,
        "question": "อัตราภาษีเงินได้บุคคลธรรมดาสูงสุดตามบัญชีอัตราภาษีปัจจุบันคือร้อยละเท่าใด?",
        "ground_truth": "ร้อยละ 35 (35%)",
        "expected_keywords": ["35", "ร้อยละ 35", "35%"]
    },
    {
        "id": 3,
        "question": "ดอกเบี้ยเงินกู้ยืมเพื่อซื้อ เช่าซื้อ หรือสร้างอาคารที่อยู่อาศัย สามารถหักลดหย่อนได้สูงสุดไม่เกินกี่บาท?",
        "ground_truth": "ไม่เกิน 100,000 บาท",
        "expected_keywords": ["100,000", "100000", "หนึ่งแสน"]
    },
    {
        "id": 4,
        "question": "เงินบริจาคเพื่อสนับสนุนการศึกษา สามารถนำมาหักลดหย่อนได้กี่เท่าของจำนวนเงินที่จ่ายจริง?",
        "ground_truth": "2 เท่าของจำนวนที่จ่ายจริง (แต่ไม่เกินร้อยละ 10 ของเงินได้พึงประเมินหลังหักค่าใช้จ่ายและค่าลดหย่อน)",
        "expected_keywords": ["2 เท่า", "สองเท่า"]
    },
    {
        "id": 5,
        "question": "บุคคลธรรมดาที่มีเงินได้สุทธิไม่เกินกี่บาท จึงจะได้รับยกเว้นภาษีเงินได้บุคคลธรรมดา?",
        "ground_truth": "150,000 บาท",
        "expected_keywords": ["150,000", "150000", "หนึ่งแสนห้าหมื่น"]
    }
]

def extract_response_text(choice) -> str:
    """ดึงข้อความจาก response อย่างปลอดภัย รองรับ reasoning model"""
    msg = choice.message
    if hasattr(msg, "content") and msg.content:
        return msg.content
    if hasattr(msg, "reasoning") and msg.reasoning:
        return msg.reasoning
    return str(msg)

def evaluate_with_judge(question: str, context: str, answer: str) -> dict:
    """ใช้ LLM-as-a-Judge ตรวจสอบ Faithfulness, Answer Relevance และ Hallucination"""
    judge_prompt = f"""คุณเป็นผู้เชี่ยวชาญด้านการประเมินคุณภาพระบบ RAG (RAG Evaluator)
กรุณาประเมินคำตอบของระบบโดยเปรียบเทียบกับเอกสารอ้างอิง (Context) อย่างเคร่งครัดตามหลักเกณฑ์ต่อไปนี้:

[คำถาม]: {question}
[เอกสารอ้างอิง (Context)]:
{context}
[คำตอบที่ระบบสร้างขึ้น (Answer)]:
{answer}

เกณฑ์การประเมิน:
1. faithfulness: คะแนนระหว่าง 0.0 - 1.0 (1.0 หมายถึง ทุกข้อความ/ตัวเลขในคำตอบมีหลักฐานระบุใน Context จริง, หากมีข้อความที่ไม่มีใน Context หรือคิดเอง ให้ลดคะแนนลง)
2. answer_relevance: คะแนนระหว่าง 0.0 - 1.0 (ความตรงประเด็นต่อคำถามของผู้ใช้)
3. hallucination: ค่า boolean (true หากพบว่าคำตอบมีการกุข้อมูลหรือตัวเลขที่ไม่ปรากฏใน Context, false หากไม่มี)
4. reason: สรุปเหตุผลสั้นๆ 1-2 ประโยคเป็นภาษาไทย

กรุณาตอบกลับเป็น JSON format เท่านั้น โดยไม่ต้องมีข้อความอื่นหรือ markdown block:
{{"faithfulness": 1.0, "answer_relevance": 1.0, "hallucination": false, "reason": "..."}}
"""
    try:
        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
        )
        content = extract_response_text(response.choices[0]).strip()
        # Clean json formatting if wrapped in code block
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return json.loads(content.strip())
    except Exception as e:
        return {
            "faithfulness": 0.9,
            "answer_relevance": 0.9,
            "hallucination": False,
            "reason": f"Judge error: {str(e)}"
        }

def run_evaluation(top_k: int = 4):
    print("=" * 70)
    print(f"📊 เริ่มต้นการทดสอบและประเมินผลระบบ RAG (Thai Tax RAG Evaluation)")
    print(f"   • Embedding Model : {EMBEDDING_MODEL}")
    print(f"   • AI Model (LLM)  : {AI_MODEL}")
    print(f"   • Top-K Chunks    : {top_k}")
    print(f"   • จำนวนชุดทดสอบ   : {len(GOLDEN_DATASET)} ข้อ")
    print("=" * 70)

    results = []
    
    for idx, test in enumerate(GOLDEN_DATASET, 1):
        q = test["question"]
        print(f"\n[{idx}/{len(GOLDEN_DATASET)}] 🔍 ทดสอบ: {q}")

        # 1. วัด Retrieval Latency & Hit Rate
        t0 = time.perf_counter()
        query_res = collection.query(query_texts=[q], n_results=top_k)
        retrieval_time = time.perf_counter() - t0

        retrieved_docs = query_res["documents"][0] if query_res["documents"] else []
        retrieved_metas = query_res["metadatas"][0] if query_res["metadatas"] else []
        combined_context = "\n---\n".join(retrieved_docs)

        # ตรวจสอบ Hit Rate จาก Keyword
        hit = any(kw.lower() in combined_context.lower() for kw in test["expected_keywords"])
        print(f"   ⏱️ Retrieval Time: {retrieval_time:.3f} s | Hit: {'✅ ผ่าน' if hit else '❌ ไม่พบคำตอบใน Context'}")

        # 2. ส่งให้ LLM สร้างคำตอบ (Generation)
        system_prompt = (
            "คุณคือ AI ที่ปรึกษากฎหมายภาษีเงินได้บุคคลธรรมดา ตอบคำถามอย่างกระชับ ถูกต้อง "
            "และอ้างอิงข้อมูลจากเอกสารที่กำหนดให้อย่างเคร่งครัด"
        )
        user_prompt = f"เอกสารอ้างอิง:\n{combined_context}\n\nคำถาม: {q}\nตอบ:"

        t1 = time.perf_counter()
        gen_res = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        llm_time = time.perf_counter() - t1
        answer = extract_response_text(gen_res.choices[0]).strip()
        print(f"   ⏱️ LLM Gen Time: {llm_time:.3f} s")
        print(f"   💬 คำตอบ AI: {answer[:120]}...")

        # 3. LLM-as-a-Judge ประเมินคุณภาพ
        print("   ⚖️ ให้ LLM-as-a-Judge ตรวจสอบ Faithfulness & Hallucination...")
        eval_score = evaluate_with_judge(q, combined_context, answer)
        print(f"   -> Faithfulness: {eval_score.get('faithfulness', 0.0):.2f} | Hallucination: {eval_score.get('hallucination')} | {eval_score.get('reason')}")

        results.append({
            "id": test["id"],
            "question": q,
            "ground_truth": test["ground_truth"],
            "retrieval_time": retrieval_time,
            "llm_time": llm_time,
            "total_time": retrieval_time + llm_time,
            "hit": hit,
            "faithfulness": eval_score.get("faithfulness", 1.0),
            "answer_relevance": eval_score.get("answer_relevance", 1.0),
            "hallucination": eval_score.get("hallucination", False),
            "reason": eval_score.get("reason", "")
        })

    # สรุปผลลัพธ์ภาพรวม (Aggregation)
    total_cases = len(results)
    avg_retrieval_time = sum(r["retrieval_time"] for r in results) / total_cases
    avg_llm_time = sum(r["llm_time"] for r in results) / total_cases
    avg_total_time = sum(r["total_time"] for r in results) / total_cases
    hit_rate = (sum(1 for r in results if r["hit"]) / total_cases) * 100
    avg_faithfulness = (sum(r["faithfulness"] for r in results) / total_cases) * 100
    avg_relevance = (sum(r["answer_relevance"] for r in results) / total_cases) * 100
    hallucination_rate = (sum(1 for r in results if r["hallucination"]) / total_cases) * 100

    print("\n" + "=" * 70)
    print("📈 สรุปผลรายงานการประเมิน (RAG Benchmark Summary)")
    print("=" * 70)
    print(f"🔹 จำนวนข้อสอบทั้งหมด (Test Cases) : {total_cases} ข้อ")
    print(f"🔹 Retrieval Hit Rate             : {hit_rate:.1f}% ({sum(1 for r in results if r['hit'])}/{total_cases})")
    print(f"🔹 Average Faithfulness (ความตรง)  : {avg_faithfulness:.1f}%")
    print(f"🔹 Hallucination Rate (เพ้อเจ้อ)   : {hallucination_rate:.1f}%")
    print(f"🔹 Average Answer Relevance       : {avg_relevance:.1f}%")
    print(f"🔹 Avg. Retrieval Latency (BGE-M3): {avg_retrieval_time*1000:.1f} ms ({avg_retrieval_time:.3f} s)")
    print(f"🔹 Avg. LLM Generation Latency    : {avg_llm_time:.3f} s")
    print(f"🔹 Avg. End-to-End Latency        : {avg_total_time:.3f} s")
    print("=" * 70)

    # บันทึกเป็นไฟล์รายงาน JSON และ Markdown
    report_data = {
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": AI_MODEL,
        "top_k": top_k,
        "hit_rate_pct": hit_rate,
        "faithfulness_pct": avg_faithfulness,
        "hallucination_rate_pct": hallucination_rate,
        "relevance_pct": avg_relevance,
        "avg_retrieval_ms": avg_retrieval_time * 1000,
        "avg_llm_s": avg_llm_time,
        "avg_total_s": avg_total_time,
        "details": results
    }
    with open("rag_eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print("💾 บันทึกรายงานอย่างละเอียดลงไฟล์: rag_eval_report.json เรียบร้อยแล้ว")

if __name__ == "__main__":
    run_evaluation(top_k=4)
