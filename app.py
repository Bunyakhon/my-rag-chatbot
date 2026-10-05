import streamlit as st
from rag_engine import (
    ask_rag, 
    check_gemini_connection, 
    check_openrouter_connection
)

# ----------------------------------------------------
# ฟังก์ชัน Vector Search ของคุณ (ตัวอย่าง Mockup)
# ให้แทนที่ส่วนนี้ด้วยฟังก์ชัน search_documents ตัวจริงของคุณ
# ----------------------------------------------------
def search_documents(query, top_k=3, distance_threshold=25.0):
    # ตัวอย่างคืนค่าจำลอง (ให้ใส่โค้ด Vector Search/ChromaDB เดิมของคุณตรงนี้)
    return [
        {"text": "เงินได้พึงประเมิน คือ เงินได้ของบุคคลใดๆ ที่เกิดขึ้นในปีภาษี...", "source": "02_เงินได้พึงประเมิน.txt"},
        {"text": "การหักค่าลดหย่อนส่วนตัว สามารถหักได้ 60,000 บาท...", "source": "05_ค่าลดหย่อนภาษี.txt"}
    ]

# ----------------------------------------------------
# UI Configuration
# ----------------------------------------------------
st.set_page_config(page_title="TAX RAG Assistant", page_icon="💰", layout="centered")

st.title("💰 ระบบที่ปรึกษาภาษีเงินได้บุคคลธรรมดา (RAG Assistant)")
st.caption("ค้นหาข้อมูลและตอบคำถามจากคลังเอกสารความรู้ภาษีเงินได้บุคคลธรรมดาอย่างแม่นยำ")

# ----------------------------------------------------
# Sidebar: เช็คสถานะการเชื่อมต่อ API
# ----------------------------------------------------
with st.sidebar:
    st.header("⚙️ สถานะการเชื่อมต่อ API")
    
    if st.button("🔄 ตรวจสอบการเชื่อมต่อใหม่"):
        st.cache_data.clear()

    # ตรวจสอบสถานะ Gemini
    gemini_ok, gemini_msg = check_gemini_connection()
    if gemini_ok:
        st.success(f"🟢 **Gemini API:**\n{gemini_msg}")
    else:
        st.error(f"🔴 **Gemini API:**\n{gemini_msg}")

    # ตรวจสอบสถานะ OpenRouter
    openrouter_ok, openrouter_msg = check_openrouter_connection()
    if openrouter_ok:
        st.success(f"🟢 **OpenRouter API (Backup):**\n{openrouter_msg}")
    else:
        st.warning(f"🟡 **OpenRouter API (Backup):**\n{openrouter_msg}")

    st.markdown("---")
    st.info("💡 ระบบจะเรียกใช้ **Gemini** เป็นหลัก และจะสลับไปใช้ **OpenRouter** อัตโนมัติเมื่อติด Quota/Error")

# ----------------------------------------------------
# Main Interface: Chat / Query
# ----------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# แสดงประวัติแชท
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            st.caption(f"📚 **เอกสารอ้างอิงที่ใช้:** {', '.join(message['sources'])}")

# กล่องรับคำถาม
if prompt := st.chat_input("พิมพ์คำถามภาษี เช่น ค่าลดหย่อนภาษีส่วนตัวได้เท่าไหร่?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาข้อมูลและประมวลผลคำตอบ..."):
            res = ask_rag(prompt, search_documents)
            answer = res["answer"]
            sources = res.get("sources", [])
            
            st.markdown(answer)
            if sources:
                st.caption(f"📚 **เอกสารอ้างอิงที่ใช้:** {', '.join(sources)}")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": sources
            })