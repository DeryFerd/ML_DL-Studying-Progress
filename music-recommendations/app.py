import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# --- FUNCTION DEFINITIONS ---

@st.cache_data
def load_data(filepath):
    """
    Loads and prepares data from the clean CSV file.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"File '{filepath}' not found.")
        st.info("Please ensure you have run the API enrichment script first.")
        return None
    
    # Rename columns for consistency (good practice)
    df.rename(columns={
        'artist(s)_name': 'artists_name', 'danceability_%': 'danceability_pct',
        'valence_%': 'valence_pct', 'energy_%': 'energy_pct', 
        'acousticness_%': 'acousticness_pct', 'instrumentalness_%': 'instrumentalness_pct',
        'liveness_%': 'liveness_pct', 'speechiness_%': 'speechiness_pct'
    }, inplace=True, errors='ignore')

    # Standard data cleaning
    df['artist_genres'].fillna('', inplace=True)
    df.dropna(subset=['track_name', 'artists_name'], inplace=True)
    return df

def calculate_similarity(df, genre_weight):
    """
    Calculates the similarity matrix based on a combination of audio and genre features.
    """
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
    """
    Provides song recommendations based on the input song title.
    """
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

def plot_radar_chart(df, selected_song):
    """
    Creates and displays a clean, non-interactive radar chart for the song's audio features.
    """
    audio_features_list = ['danceability_pct', 'valence_pct', 'energy_pct', 'acousticness_pct', 'instrumentalness_pct', 'liveness_pct', 'speechiness_pct']
    song_data = df.loc[df['track_name'] == selected_song, audio_features_list].iloc[0]
    
    values = list(song_data.values)
    
    fig = px.line_polar(
        r=values, 
        theta=audio_features_list, 
        line_close=True,
        range_r=[0, 100],
        title="Song Audio Profile"
    )
    fig.update_traces(fill='toself')

    # --- Chart Layout Improvements ---
    fig.update_layout(
        font_size=14,
        margin=dict(l=80, r=80, t=100, b=80),
        # Disable all dragging interactions on the layout
        dragmode=False 
    )
    return fig


# --- STREAMLIT UI ---
st.set_page_config(page_title="Music Recommender", layout="wide", initial_sidebar_state="expanded")

# --- LOAD DATA ---
# Make sure to use the final, clean CSV file from your API enrichment script
df = load_data('spotify-2023-final-with-genres.csv') 

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Controls")
    if df is not None:
        song_list = sorted(df['track_name'].unique())
        selected_song = st.selectbox("Select a song:", options=song_list)
        
        genre_weight = st.slider(
            "Genre Influence Weight:",
            min_value=0.0, max_value=1.0, value=0.5, step=0.1,
            help="0.0: Purely audio similarity. 1.0: Heavily influenced by genre."
        )
        
        search_button = st.button("Get Recommendations")
    else:
        selected_song = None
        st.warning("Data could not be loaded.")

# --- MAIN PAGE ---
st.title("🎵 Music Recommender System")

if df is not None and selected_song:
    # Create columns for layout
    col1, col2 = st.columns([1, 2]) 

    with col1: # Column for album art and chart
        song_details = df[df['track_name'] == selected_song].iloc[0]
        st.subheader(song_details['track_name'])
        st.write(f"**By:** *{song_details['artists_name']}*")

        # Display album art
        if pd.notna(song_details['album_art_url']):
            st.image(song_details['album_art_url'], use_container_width=True)
        
        # Display radar chart, disabling the mode bar and scroll-to-zoom
        st.plotly_chart(
            plot_radar_chart(df, selected_song), 
            use_container_width=True, 
            config={'displayModeBar': False, 'scrollZoom': False}
        )

    with col2: # Column for details and recommendations
        st.subheader("Recommendations")
        st.write(f"**Genre(s):** {song_details['artist_genres'] if song_details['artist_genres'] else 'Unknown'}")
        
        if search_button:
            with st.spinner("Analyzing similarities..."):
                similarity_matrix, valid_indices = calculate_similarity(df, genre_weight)
                recommendations = get_recommendations(selected_song, df, similarity_matrix, valid_indices, num_recommendations=7)
                
                st.write("**Other songs you might like:**")
                if not recommendations.empty:
                    st.dataframe(recommendations)
                else:
                    st.warning("Could not find any recommendations.")
else:
    st.error("Failed to load data. Please ensure the CSV file name is correct.")

# --- FOOTER ---
st.divider()
st.markdown(
    """
    <div style="text-align: center;">
        <p>Created with <b>Streamlit</b> | An AI/ML Portfolio Project</p>
    </div>
    """,
    unsafe_allow_html=True
)
