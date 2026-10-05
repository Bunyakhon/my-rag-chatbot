import os
import re
import glob
import requests
import numpy as np
import pandas as pd
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from pythainlp.util import normalize

# ==========================================
# 1. ตั้งค่าหน้าตา Streamlit App & Custom CSS
# ==========================================
st.set_page_config(
    page_title="ระบบตอบคำถามภาษีเงินได้บุคคลธรรมดา",
    page_icon="💰",
    layout="wide"
)

# ตกแต่ง UI ด้วย Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        border-color: #2563EB;
        color: #2563EB;
        background-color: #EFF6FF;
    }
    .status-card {
        padding: 1rem;
        border-radius: 8px;
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">💰 ระบบที่ปรึกษาภาษีเงินได้บุคคลธรรมดา (RAG Assistant)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">ค้นหาข้อมูลและตอบคำถามจากคลังเอกสารความรู้ภาษีเงินได้บุคคลธรรมดาอย่างแม่นยำด้วยปัญญาประดิษฐ์</div>', unsafe_allow_html=True)
st.divider()

# ==========================================
# 2. การจัดการ OpenRouter API Key และการเช็กสถานะ API
# ==========================================
if "OPENROUTER_API_KEY" in st.secrets:
    api_key = st.secrets["OPENROUTER_API_KEY"]
else:
    api_key = st.sidebar.text_input("🔑 กรุณากรอก OpenRouter API Key", type="password")

if not api_key:
    st.info("💡 กรุณากรอก OpenRouter API Key ที่ Sidebar หรือตั้งค่าใน Secrets บน Streamlit Cloud เพื่อเริ่มต้นใช้งาน")
    st.stop()

# ฟังก์ชันตรวจสอบสถานะ API ให้แสดงแถบเขียว/แดงอย่างแม่นยำ
def check_api_status(key: str) -> tuple[bool, str]:
    url = "https://openrouter.ai/api/v1/auth/key"
    headers = {"Authorization": f"Bearer {key}"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res_data = res.json()
        if res.status_code == 200 and "data" in res_data:
            label = res_data.get("data", {}).get("label", "Connected")
            return True, f"พร้อมใช้งาน ({label})"
        else:
            err = res_data.get("error", {}).get("message", res.text)
            return False, f"ขัดข้อง: {res.status_code} - {err}"
    except Exception as e:
        return False, f"ขัดข้อง: {str(e)}"

api_online, api_status_msg = check_api_status(api_key)

# แสดงแถบสถานะ API บน Sidebar
st.sidebar.header("⚙️ การตั้งค่า & สถานะ")
if api_online:
    st.sidebar.success(f"🟢 **OpenRouter API:** {api_status_msg}")
else:
    st.sidebar.error(f"🔴 **OpenRouter API:** {api_status_msg}")

# ==========================================
# 3. เตรียมระบบ RAG (Cache ไว้นานตลอดการเปิดแอป)
# ==========================================
@st.cache_resource
def load_and_prepare_rag():
    def clean_text(text: str) -> str:
        text = normalize(text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        return text.strip()

    file_paths = glob.glob("data/*.txt")
    chunks = []
    chunk_size = 1000
    overlap = 150

    if not file_paths:
        st.error("ขัดข้อง: ไม่พบไฟล์เอกสาร .txt ในโฟลเดอร์ data/")
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
            st.error(f"ขัดข้อง: ไม่สามารถอ่านไฟล์ {path} ได้ ({str(e)})")

    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embedding_model.encode(chunk_texts, convert_to_numpy=True).astype("float32")
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return embedding_model, index, chunks, clean_text

embedding_model, index, chunks, clean_text = load_and_prepare_rag()

# แสดงข้อมูลคลังเอกสารใน Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📚 คลังข้อมูลความรู้")
st.sidebar.info(f"จำนวนชิ้นส่วนข้อมูล (Chunks): **{len(chunks)}** ชิ้น")

if st.sidebar.button("🗑️ ล้างประวัติการสนทนา", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# ==========================================
# 4. ฟังก์ชัน RAG & เรียกใช้งาน OpenRouter API
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

def ask_openrouter(prompt: str) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "google/gemini-2.5-flash",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2000
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"]
        else:
            err_msg = response.json().get("error", {}).get("message", response.text)
            return f"ขัดข้อง: {response.status_code} - {err_msg}"
            
    except Exception as e:
        return f"ขัดข้อง: {str(e)}"

def ask_rag(question: str):
    retrieved_chunks = search_documents(question)
    
    if not retrieved_chunks:
        return "ไม่พบข้อมูลในเอกสารความรู้ที่ระบบมี", []

    context_text = "\n\n---\n\n".join([c["text"] for c in retrieved_chunks])
    sources = sorted(list(set([c["source"] for c in retrieved_chunks])))
    
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=question)
    answer = ask_openrouter(prompt)
    return answer, sources

# ==========================================
# 5. ตัวอย่างคำถาม 5 ข้อที่คลิกได้เลย (Quick Sample Questions)
# ==========================================
st.subheader("💡 ตัวอย่างคำถามทดสอบ")
st.caption("คลิกเลือกคำถามด้านล่างเพื่อทดสอบการตอบคำถามของระบบได้ทันที")

sample_question = None

# แถวที่ 1: คำถามที่มีข้อมูลในเอกสาร
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📌 เงินได้พึงประเมินคืออะไร?", use_container_width=True):
        sample_question = "เงินได้พึงประเมินคืออะไร?"
with col2:
    if st.button("📌 ค่าลดหย่อนภาษีส่วนตัวได้เท่าไหร่?", use_container_width=True):
        sample_question = "ค่าลดหย่อนภาษีส่วนตัวได้เท่าไหร่?"
with col3:
    if st.button("📌 ใครบ้างที่มีหน้าที่ยื่น ภ.ง.ด.94?", use_container_width=True):
        sample_question = "ใครบ้างที่มีหน้าที่ยื่น ภ.ง.ด.94?"

# แถวที่ 2: คำถามที่ไม่มีข้อมูลในคลังเอกสาร
col4, col5 = st.columns(2)
with col4:
    if st.button("❓ ภาษีมรดกมีวิธีคำนวณอย่างไร?", use_container_width=True):
        sample_question = "ภาษีมรดกมีวิธีคำนวณอย่างไร?"
with col5:
    if st.button("❓ ภาษีสำหรับธุรกิจเฉพาะมีอะไรบ้าง?", use_container_width=True):
        sample_question = "ภาษีสำหรับธุรกิจเฉพาะมีอะไรบ้าง?"

st.markdown("---")

# ==========================================
# 6. ส่วนแสดงผล UI (Chat Interface)
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# แสดงประวัติการสนทนา
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ตัวแปรรับคำถาม
user_input = st.chat_input("พิมพ์คำถามภาษีเงินได้บุคคลธรรมดาที่นี่...")

if sample_question:
    user_input = sample_question

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🔍 กำลังค้นหาข้อมูลและประมวลผลคำตอบ..."):
            answer, sources = ask_rag(user_input)
            
            if answer.startswith("ขัดข้อง:"):
                full_response = answer
            else:
                full_response = answer
                if sources:
                    full_response += "\n\n---\n📚 **เอกสารอ้างอิงที่ใช้:**\n" + "\n".join([f"- `{src}`" for src in sources])
            
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})