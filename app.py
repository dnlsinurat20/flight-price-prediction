import streamlit as st
import pandas as pd
import joblib

# 1. Konfigurasi Halaman Dasar
st.set_page_config(page_title="Prediksi Harga Tiket Pesawat", layout="centered")

# 2. Memuat Arsitektur Preprocessing dan Model (Di-cache agar efisien memori)
@st.cache_resource
def load_model_artifacts():
    preprocessor = joblib.load('preprocessor_jalur1.joblib')
    model = joblib.load('model_rf_lite_jalur1.joblib') # PASTIKAN INI MENGGUNAKAN FILE LITE
    return preprocessor, model

preprocessor, model = load_model_artifacts()

# 3. Merancang Antarmuka Pengguna (UI)
st.title("Sistem Prediksi Harga Tiket Pesawat ✈️")
st.markdown("Masukkan parameter penerbangan di bawah ini. Mesin akan mengeksekusi arsitektur Random Forest (Tuned) untuk memprediksi harga tiket secara akurat.")

# Membagi form input menjadi dua kolom untuk visual yang profesional
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Rute & Maskapai")
        airline = st.selectbox("Maskapai", ['Vistara', 'Air_India', 'Indigo', 'GO_FIRST', 'AirAsia', 'SpiceJet'])
        flight_class = st.selectbox("Kelas Penerbangan", ['Economy', 'Business'])
        source_city = st.selectbox("Kota Asal", ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'])
        destination_city = st.selectbox("Kota Tujuan", ['Mumbai', 'Delhi', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'])
        
    with col2:
        st.subheader("Data Waktu & Transit")
        departure_time = st.selectbox("Waktu Keberangkatan", ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'])
        arrival_time = st.selectbox("Waktu Kedatangan", ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'])
        stops = st.selectbox("Jumlah Transit", ['zero', 'one', 'two_or_more'])
        
    st.subheader("Data Durasi Matematis")
    duration = st.number_input("Durasi Penerbangan (Jam)", min_value=0.5, max_value=50.0, value=2.5, step=0.1)
    days_left = st.number_input("Sisa Hari (Menuju Keberangkatan)", min_value=1, max_value=50, value=15, step=1)
    
    # Tombol Eksekusi
    submit = st.form_submit_button("Kalkulasi Prediksi Harga")

# 4. Logika Pemrosesan saat Tombol Ditekan
if submit:
    if source_city == destination_city:
        st.error("Logika tidak valid: Kota Asal dan Kota Tujuan tidak boleh sama.")
    else:
        # Menyusun data input menjadi format matriks satu baris yang dikenali Pipeline
        input_data = pd.DataFrame({
            'airline': [airline],
            'source_city': [source_city],
            'departure_time': [departure_time],
            'stops': [stops],
            'arrival_time': [arrival_time],
            'destination_city': [destination_city],
            'class': [flight_class],
            'duration': [duration],
            'days_left': [days_left]
        })
        
        try:
            # Transformasi format teks ke angka (One-Hot, Ordinal, Scaling)
            processed_data = preprocessor.transform(input_data)
            
            # Prediksi matematis
            predicted_price = model.predict(processed_data)[0]
            
            # Menampilkan Hasil
            st.success(f"### Estimasi Harga Tiket: **Rp {predicted_price:,.2f}**")
            st.info("Prediksi dihitung menggunakan algoritma Random Forest dengan R2 Score 0.9861 dan MAE 1116.11.")
            
        except Exception as e:
            st.error(f"Terjadi kesalahan komputasi internal: {e}")