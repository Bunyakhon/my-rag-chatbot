import os
import requests
import streamlit as st
from google import genai
from openai import OpenAI

# 1. ดึง API Keys จาก Streamlit Secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY")

# 2. สร้าง Clients
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
) if OPENROUTER_API_KEY else None

PROMPT_TEMPLATE = """คุณเป็นผู้เชี่ยวชาญด้านภาษีอากร ตอบคำถามโดยใช้อ้างอิงจากบริบทที่กำหนดให้เท่านั้น 
หากไม่พบคำตอบในบริบท ให้ตอบว่า "ไม่พบข้อมูลในเอกสารความรู้ที่ระบบมี"

บริบท:
{context}

คำถาม: {question}
คำตอบ:"""

# ----------------------------------------------------
# ฟังก์ชันตรวจสอบสถานะการเชื่อมต่อ API (Health Check)
# ----------------------------------------------------
def check_gemini_connection():
    """ตรวจสอบว่า Gemini API พร้อมใช้งานหรือไม่"""
    if not GEMINI_API_KEY:
        return False, "ไม่พบ GEMINI_API_KEY ใน Secrets"
    try:
        # ทดลองยิง Ping ข้อความสั้นๆ
        gemini_client.models.generate_content(
            model="gemini-3.8-flash",
            contents="ping"
        )
        return True, "พร้อมใช้งาน (gemini-3.8-flash)"
    except Exception as e:
        return False, f"ขัดข้อง/ติด Quota ({str(e)})"

def check_openrouter_connection():
    """ตรวจสอบว่า OpenRouter API พร้อมใช้งานหรือไม่"""
    if not OPENROUTER_API_KEY:
        return False, "ไม่พบ OPENROUTER_API_KEY ใน Secrets"
    try:
        free_models = get_active_openrouter_free_models()
        if not free_models:
            return False, "ไม่พบโมเดลฟรีที่ active ในขณะนี้"
        return True, f"พร้อมใช้งาน ({len(free_models)} โมเดลฟรี)"
    except Exception as e:
        return False, f"ขัดข้อง ({str(e)})"

# 3. ดึงรายชื่อโมเดลฟรีจาก OpenRouter แบบไดนามิก
@st.cache_data(ttl=3600)
def get_active_openrouter_free_models():
    try:
        response = requests.get("https://openrouter.ai/api/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json().get("data", [])
            return [m["id"] for m in models if m["id"].endswith(":free")]
    except Exception:
        pass
    return ["google/gemini-2.0-flash-exp:free", "meta-llama/llama-3.1-8b-instruct:free"]

# 4. ฟังก์ชันค้นหาและตอบคำถาม
def ask_rag(question, search_documents_func):
    retrieved_chunks = search_documents_func(question, top_k=3, distance_threshold=25.0)
    
    if not retrieved_chunks:
        return {
            "answer": "ไม่พบข้อมูลในเอกสารความรู้ที่ระบบมี",
            "sources": [],
            "results": []
        }
    
    context_text = "\n\n---\n\n".join([c["text"] for c in retrieved_chunks])
    sources = list(set([c["source"] for c in retrieved_chunks]))
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=question)
    
    # 1. พยายามเรียก Gemini API ก่อน
    if gemini_client:
        try:
            response = gemini_client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )
            return {
                "answer": response.text, 
                "sources": sources, 
                "results": retrieved_chunks
            }
        except Exception:
            # ติด Error/Quota ให้ข้ามไปใช้ OpenRouter
            pass

    # 2. Fallback สลับไปใช้ OpenRouter Free Models
    free_models = get_active_openrouter_free_models()
    if openrouter_client:
        for model_name in free_models[:5]:
            try:
                response = openrouter_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=15
                )
                answer_text = response.choices[0].message.content
                return {
                    "answer": f"*(ตอบโดย OpenRouter - {model_name})*\n\n{answer_text}",
                    "sources": sources,
                    "results": retrieved_chunks
                }
            except Exception:
                continue

    return {
        "answer": "❌ ไม่สามารถดึงคำตอบจากระบบได้ในขณะนี้ (ทั้งสองระบบไม่พร้อมใช้งาน) กรุณาเว้นช่วงแล้วลองใหม่อีกครั้ง",
        "sources": sources,
        "results": retrieved_chunks
    }