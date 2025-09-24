import re

import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Diyet Chatbox", page_icon="🍏")
st.title("🍏 Yapay Zeka Destekli Diyet Chatbox (Flan-T5 Base)")

# Hugging Face FLAN-T5 tabanlı model
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

generator = load_model()

# Kullanıcı bilgileri
age = st.number_input("Yaşınız", min_value=10, max_value=100, step=1)
gender = st.selectbox("Cinsiyetiniz", ["Kadın", "Erkek"])
height = st.number_input("Boyunuz (cm)", min_value=100, max_value=250)
weight = st.number_input("Kilonuz (kg)", min_value=30, max_value=200)
goal = st.selectbox("Hedefiniz", ["Kilo Vermek", "Kilo Almak", "Kiloyu Korumak"])
activity = st.selectbox("Aktivite Seviyesi", [
    "Düşük (masa başı iş)",
    "Orta (hafif hareketli)",
    "Yüksek (sporcu vb.)"
])

if st.button("Haftalık Diyet Önerisi Al"):
    prompt = f"""
Sen profesyonel bir diyetisyensin. 
Kullanıcı bilgileri: Yaş: {age}, Cinsiyet: {gender}, Boy: {height} cm, Kilo: {weight} kg,
Hedef: {goal}, Aktivite: {activity}.

Görev: Bu kişiye uygun 7 günlük bir diyet programı hazırla. 
- Her gün için Kahvaltı, Ara Öğün, Öğle, Ara Öğün, Akşam başlıklarını yaz.
- Türk mutfağından yiyecekler öner.
- Yaklaşık kalori bilgisini ekle.
"""

    with st.spinner("Diyet önerisi hazırlanıyor..."):
        result = generator(prompt, max_length=512, do_sample=True)

    text_output = result[0]["generated_text"]

    # Sonucu daha okunabilir hale getir
    st.subheader("📝 Haftalık Diyet Önerisi")
    day_heading_pattern = re.compile(r"^\s*(\d+\.?\s*)?(gün|day)\b", re.IGNORECASE)

    for line in text_output.split("\n"):
        if line.strip():
            if day_heading_pattern.match(line):
                st.markdown(f"### 📅 {line}")
            else:
                st.markdown(f"- {line}")
