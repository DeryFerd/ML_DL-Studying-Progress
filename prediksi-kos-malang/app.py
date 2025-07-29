import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(layout="wide", page_title="Analisis Harga Kos Malang")

# --- FUNGSI UTAMA YANG MELAKUKAN SEMUANYA ---
@st.cache_data
def train_model_and_get_data():
    BASE_DIR = Path(__file__).resolve().parent
    CSV_PATH = BASE_DIR / "data_kos_malang_bersih.csv"
    df = pd.read_csv(CSV_PATH)
    
    # --- PERBAIKAN 2: NORMALISASI LOKASI ---
    # Hapus kata "Kecamatan " dan spasi di awal/akhir
    df['Lokasi'] = df['Lokasi'].str.replace('Kecamatan ', '', regex=False).str.strip()
    # ----------------------------------------
    
    df['Fasilitas'] = df['Fasilitas'].fillna('')

    # Feature Engineering Fasilitas
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

    # Definisikan pipeline
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
        remainder='passthrough'
    )
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Latih model
    model.fit(X, y)
    
    return df, model, features

# Jalankan fungsi utama
df, model, training_columns = train_model_and_get_data()

st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["🏠 Analisis Pasar", "🧮 Kalkulator Harga", "🧠 Analisis Model"])

if page == "🏠 Analisis Pasar":
    # ... (Bagian ini tidak berubah) ...
    st.title("🏠 Analisis Pasar Kos di Malang")
    # ... (dst)

elif page == "🧮 Kalkulator Harga":
    # ... (Bagian ini tidak berubah) ...
    st.title("🧮 Kalkulator Estimasi Harga Kos")
    # ... (dst)

elif page == "🧠 Analisis Model":
    st.title("🧠 Analisis Model Machine Learning")
    st.markdown("Fitur apa yang dianggap paling penting oleh model dalam memprediksi harga?")

    feature_names_raw = model.named_steps['preprocessor'].get_feature_names_out()
    importances = model.named_steps['regressor'].feature_importances_

    # --- PERBAIKAN 1: MEMBERSIHKAN NAMA FITUR ---
    cleaned_names = []
    for name in feature_names_raw:
        if 'remainder__' in name:
            clean_name = name.split('__')[1]
        elif 'cat__' in name:
            clean_name = name.split('__')[1].replace('_', ' = ')
        else:
            clean_name = name
        cleaned_names.append(clean_name)
    # -------------------------------------------

    feature_importance_df = pd.DataFrame({
        'Fitur': cleaned_names, # Gunakan nama yang sudah bersih
        'Pentingnya': importances
    }).sort_values(by='Pentingnya', ascending=False).head(15)

    st.subheader("Top 15 Fitur Paling Berpengaruh")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='Pentingnya', y='Fitur', data=feature_importance_df, palette='rocket', ax=ax)
    st.pyplot(fig)
