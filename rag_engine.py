import streamlit as st
from rag_engine import ask_rag

# Dummy หรือนำฟังก์ชัน search_documents จากโค้ดใน Colab มาวางตรงนี้
def search_documents(query, top_k=3, distance_threshold=25.0):
    # ใส่โค้ดค้นหา Vector Search เดิมของคุณที่นี่
    return []

st.set_page_config(page_title="TAX RAG Assistant", page_icon="🇹🇭", layout="centered")

st.title("🇹🇭 ระบบผู้ช่วยตอบคำถามภาษีอากร (RAG System)")
st.write("ค้นหาและตอบคำถามภาษีเงินได้บุคคลธรรมดาอ้างอิงจากเอกสารประมวลรัษฎากร")

# Sidebar สำหรับแสดงสถานะ
with st.sidebar:
    st.header("⚙️ การตั้งค่าระบบ")
    st.info("ระบบเปิดใช้งาน Auto-Fallback เมื่อติด API Quota")

# ช่องแชทสำหรับผู้ใช้งาน
if "messages" not in st.session_state:
    st.session_state.messages = []

# แสดงประวัติการสนทนา
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            st.caption(f"📚 **อ้างอิง:** {', '.join(message['sources'])}")

# รับคำถามใหม่
if prompt := st.chat_input("พิมพ์คำถามภาษีของคุณที่นี่..."):
    # บันทึกคำถามผู้ใช้
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ประมวลผลคำตอบ
    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาเอกสารและประมวลผลคำตอบ..."):
            res = ask_rag(prompt, search_documents)
            answer = res["answer"]
            sources = res.get("sources", [])
            
            st.markdown(answer)
            if sources:
                st.caption(f"📚 **อ้างอิง:** {', '.join(sources)}")
            
            # บันทึกคำตอบลง Session
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": sources
            })