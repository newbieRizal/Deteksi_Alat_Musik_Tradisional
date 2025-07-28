import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import time
import os  # untuk cek file audio lokal

# Konfigurasi halaman
st.set_page_config(
    page_title="Blitar Musical Instrument Classifier",
    page_icon="🎵",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main { max-width: 800px; padding: 2rem; }
    .header { text-align: center; margin-bottom: 2rem; }
    .result-card {
        background: white; border-radius: 10px; padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 1rem 0;
    }
    .confidence-meter {
        height: 10px; background: #e0e0e0; border-radius: 5px; margin: 0.5rem 0;
    }
    .confidence-level {
        height: 100%; border-radius: 5px; background: #4CAF50;
    }
    .upload-area {
        border: 2px dashed #4a6fa5; border-radius: 10px;
        padding: 2rem; text-align: center; margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model dengan caching
@st.cache_resource
def load_model_cached():
    try:
        model = load_model('Model_CNN_Baru_Final.keras')
        st.success("✅ Model berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {str(e)}")
        return None

# Fungsi prediksi
def predict_image(model, img):
    try:
        img = img.resize((225, 225))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        start_time = time.time()
        predictions = model.predict(img_array, verbose=0)
        pred_time = time.time() - start_time

        pred_class = np.argmax(predictions[0])
        confidence = float(predictions[0][pred_class])
        return pred_class, confidence, pred_time
    except Exception as e:
        st.error(f"❌ Error saat prediksi: {str(e)}")
        return None, None, None

# Daftar nama kelas dan data fakta
class_names = ["Balungan", "Bonang", "Gambang", "Kendang", "Rebab", "Slentho"]
fun_facts = {
    "Balungan": {
        "fact": "Nada dasar dalam musik gamelan, biasanya dimainkan oleh saron atau slenthem. Dalam musik gamelan Jawa, balungan merujuk pada melodi kerangka atau garis nada utama dari sebuah komposisi. Menariknya, meskipun alat musik seperti saron atau slenthem memainkan bagian ini, para pemain gamelan yang berpengalaman sering kali mendengar balungan hanya di dalam pikiran mereka, bahkan saat tidak ada alat musik yang secara eksplisit memainkannya.",
        "video": "https://youtu.be/Uf3YdU0NsXg?si=8pt8-X6ZUvYB0-fD",
        "audio": "data_suara/Balungan.mp3"
    },
    "Bonang": {
        "fact": "Bonang terdiri dari gong kecil yang ditata mendatar dan dipukul dengan stik empuk. Alat ini memiliki peran penting dalam ansambel gamelan sebagai pembawa melodi dan pencipta pola ritmis yang kompleks. Bonang sering menjadi pusat perhatian dalam pertunjukan gamelan karena suaranya yang khas dan perannya yang vital.",
        "video": "https://youtu.be/NcvBzIJ2tK8?si=Ui1f7p3mnvXm5WEq",
        "audio": "data_suara/Bonang.mp3"
    },
    "Gambang": {
        "fact": "Gambang adalah instrumen dari bilah kayu dimainkan dengan dua pemukul. Dengan rentang nada yang luas, gambang mampu memainkan melodi dengan kecepatan tinggi. Suaranya yang cerah dan jernih membuat gambang sering digunakan untuk mengiringi lagu-lagu tradisional Jawa.",
        "video": "https://www.youtube.com/watch?v=OpmMJNH2bG4",
        "audio": "data_suara/Gambang.mp3"
    },
    "Kendang": {
        "fact": "Kendang adalah gendang dua sisi yang berfungsi sebagai pengatur tempo dalam gamelan. Pemain kendang menggunakan teknik khusus dengan jari dan telapak tangan untuk menghasilkan berbagai macam suara. Kendang dianggap sebagai 'jantung' dari ansambel gamelan karena mengendalikan dinamika dan transisi dalam musik.",
        "video": "https://www.youtube.com/watch?v=dygpeUtvn1g",
        "audio": "data_suara/Kendang.mp3"
    },
    "Rebab": {
        "fact": "Rebab adalah alat musik gesek tradisional berbentuk hati yang berasal dari Timur Tengah. Dalam gamelan, rebab berfungsi sebagai pembawa melodi utama dan sering dianggap sebagai 'suara manusia' dalam ansambel karena kemampuannya mengekspresikan emosi melalui teknik gesek yang bervariasi.",
        "video": "https://www.youtube.com/watch?v=_m5W7QnH2Gc",
        "audio": "data_suara/Rebab.mp3"
    },
    "Slentho": {
        "fact": "Slentho mirip dengan saron, namun memiliki bunyi lebih rendah dan khas. Alat ini biasanya terbuat dari logam perunggu atau besi dan dimainkan dengan pemukul kayu. Slentho memberikan dasar harmonis dalam ansambel gamelan dan sering bekerja sama dengan demung untuk menciptakan tekstur suara yang kaya.",
        "video": "https://www.youtube.com/watch?v=9vn_-YxdYJk",
        "audio": "data_suara/Slentho.mp3"
    }
}

# Fungsi utama aplikasi
def main():
    st.markdown("""
    <div class="header">
        <h1>🎷 Blitar Musical Instrument Classifier</h1>
        <p>Identifikasi alat musik tradisional dari Blitar</p>
    </div>
    """, unsafe_allow_html=True)

    model = load_model_cached()
    if model is None:
        return

    st.markdown("""
    <div class="upload-area">
        <h4>📤 Upload Gambar Alat Musik</h4>
        <p>Format yang didukung: JPG, JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(" ", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file:
        col1, col2 = st.columns(2)

        with col1:
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, caption="Gambar yang diunggah", use_container_width=True)

        with col2:
            if st.button("🔍 Analisis Gambar", use_container_width=True):
                with st.spinner("🔎 Menganalisis..."):
                    pred_class, confidence, pred_time = predict_image(model, img)

                if pred_class is not None and 0 <= pred_class < len(class_names):
                    instrument = class_names[pred_class]
                    info = fun_facts.get(instrument, {})

                    st.markdown(f"""
                    <div class="result-card" style="background: #000; color: white; padding: 10px; border-left: 5px solid #4CAF50;">
                        <h2>{instrument}</h2>
                        <p>Kepercayaan: {confidence:.1%}</p>
                        <div style="background-color: #444; height: 10px; width: 100%; border-radius: 5px;">
                            <div style="background-color: #4CAF50; height: 100%; width: {confidence*100}%; border-radius: 5px;"></div>
                        </div>
                        <p><small>Waktu analisis: {pred_time:.2f} detik</small></p>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("ℹ️ Fakta Menarik & Multimedia"):
                        st.write(f"📌 {info.get('fact', 'Fakta tidak tersedia.')}")
                        
                        # Video
                        video_url = info.get('video')
                        if video_url:
                            st.markdown(f"📺 **[Tonton Video di YouTube]({video_url})**")
                        else:
                            st.info("📺 Video tidak tersedia.")

                        # Audio lokal
                        audio_path = info.get("audio")
                        if audio_path and os.path.exists(audio_path):
                            with open(audio_path, "rb") as f:
                                st.audio(f.read(), format="audio/mp3")
                        else:
                            st.warning("🔇 Maaf, audio tidak tersedia.")

if __name__ == "__main__":
    main()
