import os
import re
import glob
import numpy as np
import pandas as pd
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from pythainlp.util import normalize
from google import genai

# ==========================================
# 1. ตั้งค่าหน้าตาและหัวข้อของ Streamlit App
# ==========================================
st.set_page_config(
    page_title="ระบบตอบคำถามภาษีเงินได้บุคคลธรรมดา",
    page_icon="💰",
    layout="wide"
)

st.title("💰 ระบบที่ปรึกษาภาษีเงินได้บุคคลธรรมดา (RAG Assistant)")
st.markdown("ค้นหาข้อมูลและตอบคำถามจากคลังเอกสารความรู้ภาษีเงินได้บุคคลธรรมดาอย่างแม่นยำ")

# ==========================================
# 2. การจัดการ API Key
# ==========================================
# ดึง API Key จาก Streamlit Secrets หรือช่องกรอกใน Sidebar
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = st.sidebar.text_input("กรุณากรอก Gemini API Key", type="password")

if not api_key:
    st.info("💡 กรุณากรอก Gemini API Key ที่แถบด้านข้าง (Sidebar) หรือตั้งค่าใน Secrets บน Streamlit Cloud เพื่อเริ่มใช้งาน")
    st.stop()

# สร้าง Client สำหรับ Gemini API
gemini_client = genai.Client(api_key=api_key)

# ==========================================
# 3. เตรียมระบบ RAG (Cache ไว้นานตลอดการเปิดแอป)
# ==========================================
@st.cache_resource
def load_and_prepare_rag():
    """โหลดเอกสาร ทำ Cleaning, Chunking, Embedding และสร้าง FAISS Index"""
    
    def clean_text(text: str) -> str:
        """ทำความสะอาดข้อความภาษาไทย"""
        text = normalize(text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        return text.strip()

    # โหลดไฟล์ .txt จากโฟลเดอร์ data/
    file_paths = glob.glob("data/*.txt")
    chunks = []
    chunk_size = 1000
    overlap = 150

    if not file_paths:
        st.error("❌ ไม่พบไฟล์เอกสาร .txt ในโฟลเดอร์ data/ กรุณาตรวจสอบโฟลเดอร์โครงการ")
        st.stop()

    for path in sorted(file_paths):
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = clean_text(f.read())
                source = os.path.basename(path)
                
                start = 0
                while start < len(text):
                    end = start + chunk_size
                    chunk_str = text[start:end]
                    chunks.append({
                        "chunk_id": len(chunks),
                        "text": chunk_str,
                        "source": source
                    })
                    start += (chunk_size - overlap)
        except Exception as e:
            st.warning(f"ไม่สามารถอ่านไฟล์ {path} ได้: {e}")

    # สร้าง Embedding และ FAISS Index
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embedding_model.encode(chunk_texts, convert_to_numpy=True).astype("float32")
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return embedding_model, index, chunks, clean_text

# โหลดระบบ RAG
with st.spinner("กำลังเตรียมคลังข้อมูลและสร้าง Vector Database..."):
    embedding_model, index, chunks, clean_text = load_and_prepare_rag()

# ==========================================
# 4. ฟังก์ชันค้นหาเอกสารและตอบคำถามด้วย LLM
# ==========================================
PROMPT_TEMPLATE = """คุณคือ Thai Personal Income Tax Assistant

หน้าที่ของคุณคือการตอบคำถามเกี่ยวกับภาษีเงินได้บุคคลธรรมดา ให้ใช้เฉพาะข้อมูลจาก Context ที่ระบบจัดเตรียมให้เท่านั้น

กฎข้อบังคับ:
1. ห้ามใช้ความรู้ภายนอก Context และห้ามคาดเดาข้อมูลเด็ดขาด
2. หาก Context ไม่มีข้อมูลที่สามารถตอบคำถามได้ ให้ตอบว่า "ไม่พบข้อมูลในเอกสารความรู้ที่ระบบมี"
3. ตอบเป็นภาษาไทยด้วยถ้อยคำที่เข้าใจง่าย สรุปชัดเจน และระบุแหล่งอ้างอิงท้ายคำตอบ

Context:
{context}

คำถาม:
{question}
"""

def search_documents(query: str, top_k: int = 3, distance_threshold: float = 25.0):
    """ค้นหาข้อมูล chunk ที่ใกล้เคียงที่สุดจาก FAISS"""
    cleaned_q = clean_text(query)
    query_vector = embedding_model.encode([cleaned_q], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_vector, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if dist <= distance_threshold:
            item = chunks[idx].copy()
            item["distance"] = float(dist)
            results.append(item)
    return results

def ask_rag(question: str):
    """รวมขั้นตอนการค้นหาและการเจนคำตอบจาก Gemini"""
    retrieved_chunks = search_documents(question)
    
    if not retrieved_chunks:
        return "ไม่พบข้อมูลในเอกสารความรู้ที่ระบบมี", []

    context_text = "\n\n---\n\n".join([c["text"] for c in retrieved_chunks])
    sources = sorted(list(set([c["source"] for c in retrieved_chunks])))
    
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=question)

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text, sources
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการเชื่อมต่อกับ Gemini API: {str(e)}", sources

# ==========================================
# 5. ส่วนแสดงผล UI และ Chat Interface
# ==========================================

# แถบ Sidebar แสดงสถานะระบบ
st.sidebar.header("📊 สถานะระบบ")
st.sidebar.write(f"- จำนวนเอกสาร chunk ทั้งหมด: `{len(chunks)}` ชิ้น")
st.sidebar.write(f"- โมเดล Embedding: `paraphrase-multilingual-MiniLM-L12-v2`")
st.sidebar.write(f"- โมเดล LLM: `gemini-2.5-flash`")

if st.sidebar.button("ล้างประวัติการสนทนา"):
    st.session_state.messages = []
    st.rerun()

# จัดเก็บประวัติการคุยลง Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# แสดงประวัติการแชทเดิม
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# กล่องรับข้อความจากผู้ใช้งาน
if user_input := st.chat_input("สอบถามเรื่องภาษีเงินได้บุคคลธรรมดา (เช่น ค่าลดหย่อนบุตรมีอะไรบ้าง?)..."):
    # บันทึกคำถามผู้ใช้
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ประมวลผลและสร้างคำตอบ
    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาข้อมูลในคลังเอกสารและวิเคราะห์คำตอบ..."):
            answer, sources = ask_rag(user_input)
            
            # รวมคำตอบและแหล่งอ้างอิง
            full_response = answer
            if sources:
                full_response += "\n\n---\n📚 **เอกสารอ้างอิงที่ใช้:**\n" + "\n".join([f"- `{src}`" for src in sources])
            
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})