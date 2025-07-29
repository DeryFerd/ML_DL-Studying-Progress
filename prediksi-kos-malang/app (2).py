import streamlit as st
import pandas as pd
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide", page_title="Analisis Harga Kos Malang")

@st.cache_data
def load_data_and_model():
    df = pd.read_csv('data_kos_malang_bersih.csv')
    model = joblib.load('model_prediksi_harga_kos.pkl')
    with open('training_columns.pkl', 'rb') as f:
        training_columns = pickle.load(f)
    return df, model, training_columns

df, model, training_columns = load_data_and_model()

st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["🏠 Analisis Pasar", "🧮 Kalkulator Harga"])

# ======================================================================
# HALAMAN 1: ANALISIS PASAR (KODE LENGKAP)
# ======================================================================
if page == "🏠 Analisis Pasar":
    st.title("🏠 Analisis Pasar Kos di Malang")
    st.markdown(f"Hasil analisis dari **{len(df)}** data kos unik yang berhasil di-scrape dan dibersihkan.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Harga Rata-rata", f"Rp {int(df['Harga per Bulan'].mean()):,}")
    col2.metric("Harga Termurah", f"Rp {int(df['Harga per Bulan'].min()):,}")
    col3.metric("Harga Termahal", f"Rp {int(df['Harga per Bulan'].max()):,}")
    
    st.divider()

    fig_col1, fig_col2 = st.columns(2)

    with fig_col1:
        st.subheader("Rata-rata Harga per Kecamatan")
        harga_per_lokasi = df.groupby('Lokasi')['Harga per Bulan'].mean().sort_values(ascending=True)
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.barplot(x=harga_per_lokasi.values, y=harga_per_lokasi.index, ax=ax1, palette='viridis')
        ax1.set_xlabel("Rata-rata Harga")
        ax1.set_ylabel("Kecamatan")
        st.pyplot(fig1)

    with fig_col2:
        st.subheader("Jumlah Kos per Tipe")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.countplot(y=df['Tipe'], order=df['Tipe'].value_counts().index, ax=ax2, palette='plasma')
        ax2.set_xlabel("Jumlah Kos")
        ax2.set_ylabel("Tipe")
        st.pyplot(fig2)

    st.subheader("Lihat Semua Data")
    st.dataframe(df)

# ======================================================================
# HALAMAN 2: KALKULATOR HARGA (KODE LENGKAP)
# ======================================================================
elif page == "🧮 Kalkulator Harga":
    st.title("🧮 Kalkulator Estimasi Harga Kos")
    st.markdown("Masukkan spesifikasi kos untuk mendapatkan prediksi harga dari model Machine Learning kami.")
    
    lokasi_list = sorted(df['Lokasi'].unique())
    tipe_list = sorted(df['Tipe'].unique())
    all_facilities_series = df['Fasilitas'].str.split(',').explode()
    fasilitas_options = sorted(all_facilities_series.str.strip().replace('', pd.NA).dropna().unique())

    with st.form("prediction_form"):
        lokasi = st.selectbox("Pilih Lokasi/Kecamatan:", lokasi_list)
        tipe = st.selectbox("Pilih Tipe Kos:", tipe_list)
        fasilitas_terpilih = st.multiselect("Pilih Fasilitas yang Tersedia:", fasilitas_options)
        submitted = st.form_submit_button("Prediksi Harga")

        if submitted:
            input_df = pd.DataFrame(columns=training_columns)
            input_df.loc[0] = 0
            input_df.loc[0, 'Lokasi'] = lokasi
            input_df.loc[0, 'Tipe'] = tipe
            for fas in fasilitas_terpilih:
                col_name = f'Fasilitas_{fas}'
                if col_name in input_df.columns:
                    input_df.loc[0, col_name] = 1
            
            prediksi = model.predict(input_df)[0]
            
            st.success(f"Estimasi Harga Kos per Bulan: **Rp {int(prediksi):,}**")
