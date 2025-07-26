
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# --- DEFINISI FUNGSI ---

@st.cache_data
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"File '{filepath}' tidak ditemukan.")
        return None
    
    df.rename(columns={
        'artist(s)_name': 'artists_name', 'danceability_%': 'danceability_pct',
        'valence_%': 'valence_pct', 'energy_%': 'energy_pct', 
        'acousticness_%': 'acousticness_pct', 'instrumentalness_%': 'instrumentalness_pct',
        'liveness_%': 'liveness_pct', 'speechiness_%': 'speechiness_pct'
    }, inplace=True, errors='ignore')

    df['artist_genres'].fillna('', inplace=True)
    df.dropna(subset=['track_name', 'artists_name'], inplace=True)
    return df

def calculate_similarity(df, genre_weight):
    df_copy = df.copy()
    audio_features_list = ['danceability_pct', 'valence_pct', 'energy_pct', 'acousticness_pct', 'instrumentalness_pct', 'liveness_pct', 'speechiness_pct', 'bpm']
    
    df_copy.dropna(subset=audio_features_list, inplace=True)
    
    scaler = MinMaxScaler()
    audio_features_scaled = scaler.fit_transform(df_copy[audio_features_list])
    df_audio = pd.DataFrame(audio_features_scaled, index=df_copy.index, columns=audio_features_list)

    df_genres = df_copy['artist_genres'].str.get_dummies(sep=', ')
    df_genres_weighted = df_genres * genre_weight
    df_combined = pd.concat([df_audio, df_genres_weighted], axis=1)
    
    return cosine_similarity(df_combined), df_copy.index

def get_recommendations(song_title, df, similarity_matrix, valid_indices, num_recommendations=5):
    original_df_index = df.index[df['track_name'] == song_title].tolist()
    if not original_df_index:
        return pd.DataFrame()
    
    try:
        matrix_index = valid_indices.get_loc(original_df_index[0])
    except KeyError:
        return pd.DataFrame()

    similarity_scores = list(enumerate(similarity_matrix[matrix_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    top_matrix_indices = [i[0] for i in similarity_scores[1:num_recommendations+1]]
    top_original_indices = [valid_indices[i] for i in top_matrix_indices]
    
    return df.loc[top_original_indices][['track_name', 'artists_name', 'released_year']]

# --- FUNGSI CHART YANG DIPERBARUI ---
def plot_radar_chart(df, selected_song):
    """
    Membuat radar chart statis yang rapi.
    """
    audio_features_list = ['danceability_pct', 'valence_pct', 'energy_pct', 'acousticness_pct', 'instrumentalness_pct', 'liveness_pct', 'speechiness_pct']
    song_data = df.loc[df['track_name'] == selected_song, audio_features_list].iloc[0]
    
    values = list(song_data.values)
    
    fig = px.line_polar(
        r=values, 
        theta=audio_features_list, 
        line_close=True,
        range_r=[0, 100],
        title="Profil Audio Lagu"
    )
    fig.update_traces(fill='toself')

    # --- PERBAIKAN TAMPILAN CHART ---
    fig.update_layout(
        font_size=14,
        margin=dict(l=80, r=80, t=100, b=80),
        # --- PERBAIKAN DI SINI ---
        # Matikan semua jenis drag pada layout utama
        dragmode=False 
    )
    return fig


# --- UI STREAMLIT ---
st.set_page_config(page_title="Music Recommender", layout="wide", initial_sidebar_state="expanded")

df = load_data('spotify-2023-final-with-genres.csv') 

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Kontrol")
    if df is not None:
        song_list = sorted(df['track_name'].unique())
        selected_song = st.selectbox("Pilih sebuah lagu:", options=song_list)
        
        genre_weight = st.slider(
            "Bobot Pengaruh Genre:",
            min_value=0.0, max_value=1.0, value=0.5, step=0.1,
            help="0.0: Murni kemiripan audio. 1.0: Sangat dipengaruhi genre."
        )
        
        search_button = st.button("Dapatkan Rekomendasi")
    else:
        selected_song = None
        st.warning("Data tidak termuat.")

# --- HALAMAN UTAMA ---
st.title("🎵 Music Recommender System")

if df is not None and selected_song:
    col1, col2 = st.columns([1, 2]) 

    with col1: 
        song_details = df[df['track_name'] == selected_song].iloc[0]
        st.subheader(song_details['track_name'])
        st.write(f"**Oleh:** *{song_details['artists_name']}*")

        if pd.notna(song_details['album_art_url']):
            st.image(song_details['album_art_url'], use_container_width=True)
        
        # Konfigurasi di sini memastikan tidak ada zoom scroll dan toolbar
        st.plotly_chart(
            plot_radar_chart(df, selected_song), 
            use_container_width=True, 
            config={'displayModeBar': False, 'scrollZoom': False}
        )

    with col2:
        st.subheader("Rekomendasi")
        st.write(f"**Genre:** {song_details['artist_genres'] if song_details['artist_genres'] else 'Tidak diketahui'}")
        
        if search_button:
            with st.spinner("Menganalisis kemiripan..."):
                similarity_matrix, valid_indices = calculate_similarity(df, genre_weight)
                recommendations = get_recommendations(selected_song, df, similarity_matrix, valid_indices, num_recommendations=7)
                
                st.write("**Lagu lain yang mungkin kamu suka:**")
                if not recommendations.empty:
                    st.dataframe(recommendations)
else:
    st.error("Gagal memuat data. Pastikan nama file CSV sudah benar.")
