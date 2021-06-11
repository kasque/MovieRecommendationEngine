import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_title(index):
    return df[df.index == index]["title"].values[0]

def get_index(title):
    return df[df.title == title]["index"].values[0]

df=pd.read_csv("movie_dataset.csv")
#print(df.columns)

#selecting features
features = ['keywords', 'cast', 'genres', 'director']

#replacing all the NA values with empty string
for feature in features:
    df[feature] = df[feature].fillna('')

#combining all the features
def combined_features(row):
    try:
        return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director']
    except:
        print("error: ", row)

df["combined_features"] = df.apply(combined_features,axis=1)

#creating count matrix
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])

#calculating cosine similarity
similarity = cosine_similarity(count_matrix)

#taking input
movie_input = input("Enter the movie user likes:  ")

movie_index= get_index(movie_input)
similar_movies = list(enumerate(similarity[movie_index]))

#list the similar movies in descending order of their similarity value
sorted_list = sorted(similar_movies, key = lambda x:x[1], reverse = True)

#printing first 10 similar movies
i=0
for movie in sorted_list:
    print(get_title(movie[0]))
    i=i+1
    if i>20:
        break
