import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load model dan class names
model = load_model("Model_CNN_Baru_Final.h5")
class_names = ['Balungan', 'Bonang', 'Kendang', 'Slentho']

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Klasifikasi Alat Musik Tradisional",
    page_icon=":musical_note:",
    layout="centered"
)

# CSS tambahan untuk tampilan
st.markdown("""
    <style>
    .title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #4CAF50;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #777;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Judul Aplikasi
st.markdown('<div class="title">Klasifikasi Alat Musik Tradisional Jawa Timur 🎶</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload gambar alat musik dan model akan memprediksi jenisnya</div>', unsafe_allow_html=True)

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar alat musik...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    # Preprocessing gambar
    image_resized = image.resize((225, 225))
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Prediksi
    predictions = model.predict(image_array)
    predicted_index = np.argmax(predictions[0])
    predicted_instrument = class_names[predicted_index]
    confidence = float(np.max(predictions[0]))

    # Penilaian keyakinan
    if confidence > 0.75:
        level_text = "Prediksi sangat yakin"
        level_color = "#4CAF50"  # hijau
    elif confidence > 0.5:
        level_text = "Prediksi cukup yakin"
        level_color = "#FFC107"  # kuning
    else:
        level_text = "Prediksi kurang yakin"
        level_color = "#F44336"  # merah

    # Tampilkan hasil prediksi
    st.markdown(f"""
    <div style="background-color: #f9f9f9; border-radius: 12px; padding: 1.5rem; 
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1); margin-top: 2rem;
                color: var(--text-color);">
        <h3 style="text-align: center; color: var(--primary-color);">🎉 Hasil Prediksi</h3>
        
        <div style="text-align: center; font-size: 2rem; font-weight: bold; margin: 1rem 0;">
            {predicted_instrument}
        </div>
        
        <div style="text-align: center; font-size: 1rem; margin-bottom: 1rem;">
            Tingkat Keyakinan: <strong>{int(confidence * 100)}%</strong>
        </div>
        
        <div style="height: 20px; width: 100%; background-color: #e0e0e0; 
                    border-radius: 10px; overflow: hidden;">
            <div style="height: 100%; width: {confidence * 100}%; background-color: {level_color};"></div>
        </div>
        
        <p style="text-align: center; margin-top: 1rem; font-style: italic; color: {level_color};">
            {level_text}
        </p>
    </div>
""", unsafe_allow_html=True)

