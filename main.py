#!/usr/bin/env python
# coding: utf-8

# #### 🔹 Import Necessary Libraries

# In[145]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[146]:


movies = pd.read_csv('tmdb_5000_movies.csv')
movies = movies[['id','title','overview','genres', 'homepage']]
movies["overview"]= movies["overview"].fillna("")
movies.head()


# In[147]:


import ast

def clean_genres(genre_str):
    genres_list = ast.literal_eval(genre_str)  # Convert JSON string to list
    return " ".join([genre["name"] for genre in genres_list])

movies['genres'] = movies['genres'].apply(clean_genres)
movies.head()


# In[148]:


movies["Content"] = movies["title"] +" "+ movies["genres"] +" "+ movies["overview"]
movies.head()


# In[149]:


tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["Content"])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# In[150]:


def recommend(movie_title):
    try:
        idx_list = movies[movies["title"] == movie_title].index.tolist()
        if not idx_list:
            return ["Movie not found"]

        idx = idx_list[0]
        scores = list(enumerate(cosine_sim[idx]))  
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]  

        recommendations = [
            {"title": movies.iloc[i[0]]["title"], "homepage": movies.iloc[i[0]]["homepage"]}
            for i in scores
        ]
        return recommendations   
    except Exception as e:
        return [f"Error: {str(e)}"]


# In[151]:


print(recommend("friends"))


# In[152]:


import streamlit as st

st.title("Movie Recommendation System 🎬")

movie_name = st.text_input("Enter a movie name:")
if st.button("Get Recommendations"):
    recommendations = recommend(movie_name)

    if isinstance(recommendations, list) and "Movie not found" in recommendations:
        st.error("Movie not found. Try another title.")
    else:
        for movie in recommendations:
            st.subheader(movie["title"])
            st.markdown(f"[Visit Homepage]({movie['homepage']})", unsafe_allow_html=True)

