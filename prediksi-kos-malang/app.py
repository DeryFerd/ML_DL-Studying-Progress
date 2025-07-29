import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path # <-- Tambahkan ini

st.set_page_config(layout="wide", page_title="Analisis Harga Kos Malang")

# --- FUNGSI UTAMA YANG MELAKUKAN SEMUANYA ---
@st.cache_data
def train_model_and_get_data():
    # --- BAGIAN PATH YANG DIPERBAIKI ---
    BASE_DIR = Path(__file__).resolve().parent
    CSV_PATH = BASE_DIR / "data_kos_malang_bersih.csv"
    df = pd.read_csv(CSV_PATH)
    # -----------------------------------
    
    df['Fasilitas'] = df['Fasilitas'].fillna('')

    # Lakukan Feature Engineering
    fasilitas_dummies = df['Fasilitas'].str.get_dummies(sep=r'\s*,\s*')
    if '' in fasilitas_dummies.columns:
        fasilitas_dummies = fasilitas_dummies.drop(columns=[''])
    fasilitas_dummies.columns = fasilitas_dummies.columns.str.strip()
    fasilitas_dummies = fasilitas_dummies.groupby(level=0, axis=1).sum()
    fasilitas_dummies = fasilitas_dummies.add_prefix('Fasilitas_')
    df_engineered = pd.concat([df, fasilitas_dummies], axis=1)

    # Siapkan data untuk training
    categorical_features = ['Lokasi', 'Tipe']
    numerical_features = list(fasilitas_dummies.columns)
    features = categorical_features + numerical_features
    target = 'Harga per Bulan'

    X = df_engineered[features]
    y = df_engineered[target]

    # Definisikan preprocessor dan model pipeline
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
        remainder='passthrough'
    )
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Latih model dengan SEMUA data
    model.fit(X, y)
    
    return df, model, features

# Jalankan fungsi utama
df, model, training_columns = train_model_and_get_data()

# --- Sisa aplikasi sama persis ---
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["🏠 Analisis Pasar", "🧮 Kalkulator Harga", "🧠 Analisis Model"])

if page == "🏠 Analisis Pasar":
    st.title("🏠 Analisis Pasar Kos di Malang")
    st.markdown(f"Hasil analisis dari **{len(df)}** data kos unik.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Harga Rata-rata", f"Rp {int(df['Harga per Bulan'].mean()):,}")
    col2.metric("Harga Termurah", f"Rp {int(df['Harga per Bulan'].min()):,}")
    col3.metric("Harga Termahal", f"Rp {int(df['Harga per Bulan'].max()):,}")
    st.divider()
    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        st.subheader("Rata-rata Harga per Kecamatan")
        harga_per_lokasi = df.groupby('Lokasi')['Harga per Bulan'].mean().sort_values(ascending=True)
        fig, ax = plt.subplots()
        sns.barplot(x=harga_per_lokasi.values, y=harga_per_lokasi.index, ax=ax, palette='viridis')
        st.pyplot(fig)
    with fig_col2:
        st.subheader("Jumlah Kos per Tipe")
        fig, ax = plt.subplots()
        sns.countplot(y=df['Tipe'], order=df['Tipe'].value_counts().index, ax=ax, palette='plasma')
        st.pyplot(fig)
    st.subheader("Lihat Semua Data")
    st.dataframe(df)

elif page == "🧮 Kalkulator Harga":
    st.title("🧮 Kalkulator Estimasi Harga Kos")
    
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

# --- HALAMAN BARU: ANALISIS MODEL ---
elif page == "🧠 Analisis Model":
    st.title("🧠 Analisis Model Machine Learning")
    st.markdown("Fitur apa yang dianggap paling penting oleh model dalam memprediksi harga?")

    # Ambil nama fitur setelah di-encode oleh preprocessor
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    
    # Ambil nilai importance dari regressor
    importances = model.named_steps['regressor'].feature_importances_

    # Buat DataFrame
    feature_importance_df = pd.DataFrame({
        'Fitur': feature_names,
        'Pentingnya': importances
    }).sort_values(by='Pentingnya', ascending=False)

    st.subheader("Tingkat Kepentingan Fitur")
    
    # Tampilkan 15 fitur paling penting
    top_features = feature_importance_df.head(15)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='Pentingnya', y='Fitur', data=top_features, palette='rocket', ax=ax)
    ax.set_title("Top 15 Fitur Paling Berpengaruh")
    st.pyplot(fig)

    with st.expander("Lihat semua tingkat kepentingan fitur"):
        st.dataframe(feature_importance_df)
