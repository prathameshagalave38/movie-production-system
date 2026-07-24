import streamlit as st
import pickle
import pandas as pd
import difflib

# 1. Page Configuration
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)

# 2. Load Model Artifacts with Caching
@st.cache_resource
def load_data():
    movies = pickle.load(open('artifacts/movies.pkl', 'rb'))
    similarity = pickle.load(open('artifacts/similarity.pkl', 'rb'))
    return movies, similarity

try:
    movies, similarity = load_data()
except FileNotFoundError:
    # Fallback if artifacts folder isn't used
    movies = pickle.load(open('movies.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))

# 3. Custom Recommendation Logic
def recommend(movie_name, top_n=10):
    clean_query = str(movie_name).lower().strip()
    all_titles = movies['clean_title'].tolist()
    
    matched_titles = difflib.get_close_matches(clean_query, all_titles, n=1, cutoff=0.3)
    
    if not matched_titles:
        return None, f"No close matches found for '{movie_name}'."
    
    best_match = matched_titles[0]
    idx = movies[movies['clean_title'] == best_match].index[0]
    actual_title = movies.loc[idx, 'title']
    
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    recommended_movies = []
    for score_idx, score in sim_scores:
        recommended_movies.append({
            'title': movies.loc[score_idx, 'title'],
            'genres': movies.loc[score_idx, 'genres'],
            'score': f"{score * 100:.1f}%"
        })
        
    return actual_title, pd.DataFrame(recommended_movies)

# 4. Streamlit UI Layout
st.title("🎬 Content-Based Movie Recommendation System")
st.markdown("Discover movies similar to your favorites using Natural Language Processing and Cosine Similarity.")
st.divider()

# Sidebar Info
st.sidebar.header("About System")
st.sidebar.info(
    "This system uses **TF-IDF Vectorization** on movie genre attributes "
    "and calculates similarity using **Cosine Similarity**."
)

# Main Search Input
movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown:",
    movie_list
)

num_recommendations = st.slider("Number of recommendations:", min_value=5, max_value=20, value=10)

if st.button("Get Recommendations 🚀", type="primary"):
    with st.spinner('Finding recommendations...'):
        matched_title, recommendations = recommend(selected_movie, top_n=num_recommendations)
        
        if recommendations is None:
            st.error(matched_title)
        else:
            st.success(f"Showing top recommendations similar to: **{matched_title}**")
            
            # Display results in structured UI grid
            for idx, row in recommendations.iterrows():
                with st.container():
                    col1, col2, col3 = st.columns([3, 3, 2])
                    with col1:
                        st.subheader(f"{idx + 1}. {row['title']}")
                    with col2:
                        st.write(f"**Genres:** {row['genres']}")
                    with col3:
                        st.metric(label="Match Score", value=row['score'])
                    st.divider()