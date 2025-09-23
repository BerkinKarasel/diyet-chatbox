import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Diyet Chatbox", page_icon="🍏")
st.title("🍏 Yapay Zeka Destekli Diyet Chatbox (Web Tabanlı)")

# Hugging Face'ten küçük bir model (Türkçe destekli BloomZ)
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="bigscience/bloomz-560m")

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
    Kullanıcı bilgileri:
    Yaş: {age}, Cinsiyet: {gender}, Boy: {height}, Kilo: {weight},
    Hedef: {goal}, Aktivite: {activity}.

    Bu kişiye uygun 1 haftalık detaylı diyet listesi hazırla.
    Kahvaltı, ara öğün, öğle, akşam yemeklerini belirt.
    """
    with st.spinner("Diyet önerisi hazırlanıyor..."):
        result = generator(prompt, max_length=300, do_sample=True, temperature=0.7)
        st.subheader("📝 Haftalık Diyet Önerisi")
        st.write(result[0]["generated_text"])
