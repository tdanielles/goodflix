import streamlit as st
import pickle
import pandas as pd
import re
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)


# Find id given title
def find(movie_title):
    movie_title = clean_title(movie_title)
    query_vec = vectorizer.transform([movie_title])

    similarity = cosine_similarity(query_vec, tfidf).flatten()

    index = np.argpartition(similarity, -5)[-5:]
    result = movies.iloc[index][::-1][:1]

    return result.movieId.item()


def recommend(movie_title):
    movie_id = find(movie_title)

    # finding recommendations from users similar to us
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]

    # adjusting so we only have recommendations where over 10% of users recommended that movie
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
    similar_user_recs = similar_user_recs[similar_user_recs > .10]

    # finding how common the recommendation is among all users
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_users_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())

    # creating recommendation score
    rec_percentages = pd.concat([similar_user_recs, all_users_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]

    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]

    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")


movies_dict = pickle.load(open("movies_dict.pkl", "rb"))
movies = pd.DataFrame(movies_dict)

ratings = pickle.load(open("ratings.pkl", "rb"))

vectorizer = TfidfVectorizer(ngram_range=(1, 2))

tfidf = vectorizer.fit_transform(movies["clean_title"])

# Main program
st.markdown(f'<h1 style="color:#E50914;font-size:70px;margin-bottom:40px;'
            f'text-align:center; letter-spacing:5px;">{"GOODFLIX"}</h1>', unsafe_allow_html=True)

selected_movie_name = st.selectbox(
    "Choose a movie you recently watched",
    movies["title"].values
)

if st.button("Recommend similar movies!"):
    recommendations = recommend(selected_movie_name)
    recommendations = recommendations.title.head()

    if len(recommendations) == 0:
        st.write("Sorry, we couldn't find any match :(")
    else:
        for i in recommendations:
            st.markdown(f'<p style="color:#FFFFFF;font-size:16px;">{i}</p>', unsafe_allow_html=True)
