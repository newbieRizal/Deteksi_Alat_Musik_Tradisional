import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import time
import os
from streamlit_lottie import st_lottie
import requests
import json

# Konfigurasi halaman
st.set_page_config(
    page_title="Blitar Musical Instrument Classifier",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load Lottie animations
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_music = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_jcikwtux.json")
lottie_upload = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_yo4ytbyz.json")
lottie_success = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_7yomjwvk.json")

# Custom CSS with animations and better styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main { 
        max-width: 900px; 
        padding: 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .header { 
        text-align: center; 
        margin-bottom: 2rem;
        animation: fadeIn 1s ease-in-out;
    }
    
    .header h1 {
        color: #2c3e50;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .header p {
        color: #7f8c8d;
        font-size: 1.1rem;
    }
    
    .result-card {
        background: white; 
        border-radius: 15px; 
        padding: 1.5rem;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1); 
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: slideUp 0.5s ease-out;
    }
    
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.15);
    }
    
    .confidence-meter {
        height: 12px; 
        background: #e0e0e0; 
        border-radius: 10px; 
        margin: 0.8rem 0;
        overflow: hidden;
    }
    
    .confidence-level {
        height: 100%; 
        border-radius: 10px; 
        background: linear-gradient(90deg, #4CAF50 0%, #8BC34A 100%);
        transition: width 1s ease-in-out;
    }
    
    .upload-area {
        border: 2px dashed #4a6fa5; 
        border-radius: 15px;
        padding: 2rem; 
        text-align: center; 
        margin-bottom: 2rem;
        background: rgba(255,255,255,0.7);
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        background: rgba(255,255,255,0.9);
        border-color: #3498db;
    }
    
    .instrument-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .instrument-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 50px;
        font-size: 1rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 14px rgba(0,0,0,0.15);
        background: linear-gradient(135deg, #43A047 0%, #1B5E20 100%);
    }
    
    .tab-content {
        padding: 1rem 0;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideUp {
        from { 
            opacity: 0;
            transform: translateY(20px);
        }
        to { 
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
        color: #4CAF50;
    }
    
    .feature-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load model dengan caching
@st.cache_resource
def load_model_cached():
    try:
        model = load_model('Model_CNN_Baru_Final.keras')
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
        "video": "https://www.youtube.com/watch?v=5kLDV8PnHaI",
        "audio": "data_suara/Balungan.mp3",
        "icon": "🎼"
    },
    "Bonang": {
        "fact": "Bonang terdiri dari gong kecil yang ditata mendatar dan dipukul dengan stik empuk. Alat ini memiliki peran penting dalam ansambel gamelan sebagai pembawa melodi dan pencipta pola ritmis yang kompleks. Bonang sering menjadi pusat perhatian dalam pertunjukan gamelan karena suaranya yang khas dan perannya yang vital.",
        "video": "https://www.youtube.com/watch?v=U4SvK8QaJr8",
        "audio": "data_suara/Bonang.mp3",
        "icon": "🥁"
    },
    "Gambang": {
        "fact": "Gambang adalah instrumen dari bilah kayu dimainkan dengan dua pemukul. Dengan rentang nada yang luas, gambang mampu memainkan melodi dengan kecepatan tinggi. Suaranya yang cerah dan jernih membuat gambang sering digunakan untuk mengiringi lagu-lagu tradisional Jawa.",
        "video": "https://www.youtube.com/watch?v=OpmMJNH2bG4",
        "audio": "data_suara/Gambang.mp3",
        "icon": "🎹"
    },
    "Kendang": {
        "fact": "Kendang adalah gendang dua sisi yang berfungsi sebagai pengatur tempo dalam gamelan. Pemain kendang menggunakan teknik khusus dengan jari dan telapak tangan untuk menghasilkan berbagai macam suara. Kendang dianggap sebagai 'jantung' dari ansambel gamelan karena mengendalikan dinamika dan transisi dalam musik.",
        "video": "https://www.youtube.com/watch?v=dygpeUtvn1g",
        "audio": "data_suara/Kendang.mp3",
        "icon": "🪘"
    },
    "Rebab": {
        "fact": "Rebab adalah alat musik gesek tradisional berbentuk hati yang berasal dari Timur Tengah. Dalam gamelan, rebab berfungsi sebagai pembawa melodi utama dan sering dianggap sebagai 'suara manusia' dalam ansambel karena kemampuannya mengekspresikan emosi melalui teknik gesek yang bervariasi.",
        "video": "https://www.youtube.com/watch?v=_m5W7QnH2Gc",
        "audio": "data_suara/Rebab.mp3",
        "icon": "🎻"
    },
    "Slentho": {
        "fact": "Slentho mirip dengan saron, namun memiliki bunyi lebih rendah dan khas. Alat ini biasanya terbuat dari logam perunggu atau besi dan dimainkan dengan pemukul kayu. Slentho memberikan dasar harmonis dalam ansambel gamelan dan sering bekerja sama dengan demung untuk menciptakan tekstur suara yang kaya.",
        "video": "https://www.youtube.com/watch?v=9vn_-YxdYJk",
        "audio": "data_suara/Slentho.mp3",
        "icon": "🔔"
    }
}

# Fungsi utama aplikasi
def main():
    # Header dengan animasi
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("""
            <div class="header">
                <h1>🎷 Blitar Musical Instrument Classifier</h1>
                <p>Identifikasi alat musik tradisional dari Blitar dengan teknologi AI</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if lottie_music:
                st_lottie(lottie_music, height=100, key="music")
    
    # Tabs untuk navigasi
    tab1, tab2, tab3 = st.tabs(["🔍 Identifikasi Alat Musik", "📚 Ensiklopedia Alat Musik", "ℹ️ Tentang Aplikasi"])
    
    with tab1:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h3 style="color: #2c3e50;">Unggah gambar alat musik untuk diidentifikasi</h3>
            <p style="color: #7f8c8d;">Format yang didukung: JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)
        
        model = load_model_cached()
        if model is None:
            return
        
        # Upload area dengan animasi
        with st.container():
            st.markdown("""
            <div class="upload-area pulse">
                <h4>📤 Tarik dan Lepas Gambar di Sini</h4>
                <p>Atau klik untuk memilih file</p>
            </div>
            """, unsafe_allow_html=True)
            
            if lottie_upload:
                st_lottie(lottie_upload, height=150, key="upload")
        
        uploaded_file = st.file_uploader(" ", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        if uploaded_file:
            col1, col2 = st.columns(2)

            with col1:
                img = Image.open(uploaded_file).convert('RGB')
                st.image(img, caption="Gambar yang diunggah", use_container_width=True)
                
            with col2:
                if st.button("🔍 Analisis Gambar", use_container_width=True, key="analyze"):
                    with st.spinner("🔎 Menganalisis gambar..."):
                        pred_class, confidence, pred_time = predict_image(model, img)

                    if pred_class is not None and 0 <= pred_class < len(class_names):
                        instrument = class_names[pred_class]
                        info = fun_facts.get(instrument, {})
                        
                        # Animasi sukses
                        if lottie_success:
                            st_lottie(lottie_success, height=100, key="success")

                        st.markdown(f"""
                        <div class="result-card">
                            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                                <span style="font-size: 2rem; margin-right: 1rem;">{info.get('icon', '🎵')}</span>
                                <h2 style="margin: 0; color: #2c3e50;">{instrument}</h2>
                            </div>
                            <p style="color: #7f8c8d;">Kepercayaan model:</p>
                            <div class="confidence-meter">
                                <div class="confidence-level" style="width: {confidence*100}%"></div>
                            </div>
                            <p style="text-align: right; color: #7f8c8d; font-size: 0.9rem;">{confidence:.1%}</p>
                            <p style="color: #7f8c8d; font-size: 0.9rem;">Waktu analisis: {pred_time:.2f} detik</p>
                        </div>
                        """, unsafe_allow_html=True)

                        with st.expander(f"🎵 Detail tentang {instrument}", expanded=True):
                            st.markdown(f"""
                            <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                                <h4 style="color: #2c3e50; margin-top: 0;">Fakta Menarik</h4>
                                <p style="color: #34495e;">{info.get('fact', 'Fakta tidak tersedia.')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Multimedia section
                            st.markdown("### 🎬 Multimedia")
                            col_media1, col_media2 = st.columns(2)
                            
                            with col_media1:
                                # Video
                                video_url = info.get('video')
                                if video_url:
                                    st.markdown(f"""
                                    <a href="{video_url}" target="_blank" style="text-decoration: none;">
                                        <div style="background: #4CAF50; color: white; padding: 0.5rem 1rem; border-radius: 5px; text-align: center; margin-bottom: 1rem;">
                                            Tonton Video di YouTube
                                        </div>
                                    </a>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.info("Video tidak tersedia.")
                            
                            with col_media2:
                                # Audio lokal
                                audio_path = info.get("audio")
                                if audio_path and os.path.exists(audio_path):
                                    with open(audio_path, "rb") as f:
                                        audio_bytes = f.read()
                                    st.audio(audio_bytes, format="audio/mp3")
                                else:
                                    st.warning("Audio tidak tersedia.")
    
    with tab2:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="color: #2c3e50;">📚 Ensiklopedia Alat Musik Blitar</h2>
            <p style="color: #7f8c8d;">Pelajari tentang berbagai alat musik tradisional dari Blitar</p>
        </div>
        """, unsafe_allow_html=True)
        
        for instrument in class_names:
            info = fun_facts.get(instrument, {})
            with st.container():
                st.markdown(f"""
                <div class="instrument-card">
                    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                        <span style="font-size: 2rem; margin-right: 1rem;">{info.get('icon', '🎵')}</span>
                        <h3 style="margin: 0; color: #2c3e50;">{instrument}</h3>
                    </div>
                    <p style="color: #34495e;">{info.get('fact', 'Fakta tidak tersedia.')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Multimedia for encyclopedia
                col_media1, col_media2 = st.columns(2)
                with col_media1:
                    video_url = info.get('video')
                    if video_url:
                        st.markdown(f"[🎥 Tonton Video]({video_url})", unsafe_allow_html=True)
                with col_media2:
                    audio_path = info.get("audio")
                    if audio_path and os.path.exists(audio_path):
                        with open(audio_path, "rb") as f:
                            st.audio(f.read(), format="audio/mp3")
                st.markdown("---")
    
    with tab3:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="color: #2c3e50;">ℹ️ Tentang Aplikasi Ini</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: white; border-radius: 15px; padding: 1.5rem; box-shadow: 0 5px 15px rgba(0,0,0,0.05); margin-bottom: 2rem;">
            <p style="color: #34495e;">Aplikasi ini menggunakan teknologi <strong>Convolutional Neural Network (CNN)</strong> untuk mengidentifikasi alat musik tradisional dari Blitar. Model AI telah dilatih dengan ratusan gambar untuk memberikan hasil yang akurat.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Fitur aplikasi
        st.markdown("### ✨ Fitur Aplikasi")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🔍</div>
                <h4>Identifikasi Cepat</h4>
                <p>Analisis gambar dalam hitungan detik</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🎵</div>
                <h4>Audio Sample</h4>
                <p>Dengarkan suara asli alat musik</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">📚</div>
                <h4>Ensiklopedia</h4>
                <p>Pelajari tentang alat musik tradisional</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Tim pengembang
        st.markdown("""
        <div style="margin-top: 2rem;">
            <h4>👨‍💻 Tim Pengembang</h4>
            <p>Aplikasi ini dikembangkan oleh tim yang peduli dengan pelestarian budaya Indonesia, khususnya alat musik tradisional dari Blitar.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()