import pandas as pd
import numpy as np
import requests
import tarfile
import regex as re
import string
import spacy
import torch

from wordcloud import WordCloud
from textwrap import wrap
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

import nltk
import json
from collections import Counter

#check if MovieSummaries.tar.gz exists, download and decompress
try:
    open('MovieSummaries.tar.gz')
except FileNotFoundError:
    print("Downloading MovieSummaries.tar.gz, decompressing, and saving to ./Data")
    url = 'http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz'
    r = requests.get(url)
    with open('MovieSummaries.tar.gz', 'wb') as f:
        f.write(r.content)
    # decompress downloaded data
    with tarfile.open('MovieSummaries.tar.gz') as f:
        f.extractall('./Data')
        

# read data
# some files are saved as .txt but all files are in tsv-format (tab separated)
df_character_metadata = pd.read_csv('Data/MovieSummaries/character.metadata.tsv', delimiter='\t', header=None)
df_movie_metadata = pd.read_csv('Data/MovieSummaries/movie.metadata.tsv', delimiter='\t', header=None)
df_name_cluster = pd.read_csv('Data/MovieSummaries/name.clusters.txt', delimiter='\t', header=None)
df_plot_summary = pd.read_csv('Data/MovieSummaries/plot_summaries.txt', delimiter='\t', header=None)
df_tvtrope_cluster = pd.read_csv('Data/MovieSummaries/tvtropes.clusters.txt', delimiter='\t', header=None)

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove numbers
    text = re.sub(r'\w*\d\w*', '', text)
    # Remove punctuation
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    # Remove extra spaces
    text = re.sub(r' +', ' ', text)
    #remove leading and trailing spaces
    text = text.strip()
    return text

# try if cuda is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(torch.cuda.get_device_name(0))
    import spacy_transformers
    spacy.require_gpu() #or spacy.prefer_gpu()
else:
    device = torch.device('cpu')  


#test if spacy en_core_web_xx exists, if not download it
#English pipeline optimized for CPU. Components: tok2vec, tagger, parser, senter, ner, attribute_ruler, lemmatizer.
#https://spacy.io/models/en

try:
    lemmatizer = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
except:
    print('Downloading en_core_web_sm')
    !python -m spacy download en_core_web_sm
    lemmatizer = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    

#if plot_summaries.csv exists, read it, else create it
try:
    df_plot_summary = pd.read_csv('Data/MovieSummaries/plot_summaries_trf.csv').iloc[: , 1:]
except FileNotFoundError:
    print('Creating plot_summaries.csv')
    # preprocess the text
    df_plot_summary[1] = df_plot_summary[1].apply(preprocess_text)
    # Lemmatization with stopwords removal
    df_plot_summary[1] = df_plot_summary[1].apply(lambda x: ' '.join([token.lemma_ for token in list(lemmatizer(x)) if (token.is_stop==False)]))
    # save the preprocessed data
    df_plot_summary.to_csv('Data/MovieSummaries/plot_summaries.csv')

df_plot_summary.head()

df_plot_summary

# create document term matrix
cv = CountVectorizer(analyzer='word')
data = cv.fit_transform(df_plot_summary['1'])  # returns sparse DTM

# Extract one row (film) from the DTM and expand to all rows (sparse -> dense)
df_dtm = pd.DataFrame(data.getrow(1).toarray(), columns=cv.get_feature_names_out())

# Importing wordcloud for plotting word clouds and textwrap for wrapping longer text


# Function for generating word clouds
def generate_wordcloud(data,title):
  wc = WordCloud(width=400, height=330, max_words=150,colormap="Dark2").generate_from_frequencies(data)
  plt.figure(figsize=(10,8))
  plt.imshow(wc, interpolation='bilinear')
  plt.axis("off")
  plt.title('\n'.join(wrap(title,60)),fontsize=13)
  plt.show()


generate_wordcloud(df_dtm.iloc[0], 'Wordcloud')

df_plot_summary['1'].str.len().hist(bins=30)

df_plot_summary['1'].str.split().map(lambda x: len(x)).hist(bins=30)

genre_df = df_movie_metadata[[0,2,8]]
genre_df['genre'] = genre_df[8].apply(lambda x: list(json.loads(x).values()))
genre_df['genre_id'] = genre_df[8].apply(lambda x: list(json.loads(x).keys()))

genre_df.columns = ['movie_id','title','genre_json','genre','genre_id']

#remove genre_json and genre_id
genre_df = genre_df.drop(columns=['genre_json','genre_id'])
genre_df.head()

flattened_list = [item for sublist in genre_df['genre'] for item in sublist]

# Step 2: Count the occurrences of each element
element_counts = Counter(flattened_list)
print(len(element_counts))
sorted(element_counts.items(), key=lambda x:x[1], reverse=True)

#remove all with <= 10 occurences -> reduce from 363 to 
element_counts = {k: v for k, v in element_counts.items() if v >= 10}
print(len(element_counts))
sorted(element_counts.items(), key=lambda x:x[1], reverse=True)


for i in np.arange(0, len(genre_df)):
    genres = genre_df['genre'].iloc[i]
    if len(genres) > 0:
        counts = np.zeros(len(genres))
        idx = 0
        
        for genre in genres:
            counts[idx] = element_counts[genre]
            idx += 1

        print(genres[np.argmax(counts)])
    else:
        print('None')


# make a cooccurence matrix for the genres in genred_df column 'genre' and plot it
from itertools import combinations

# Create an empty co-occurrence matrix
# Create an empty co-occurrence matrix
genre_list = list(set(genre for sublist in genre_df['genre'] for genre in sublist))
co_occurrence_matrix = pd.DataFrame(0, columns=genre_list, index=genre_list)

# Iterate through rows and update the matrix
for genres in genre_df['genre']:
    for genre1, genre2 in combinations(genres, 2):
        co_occurrence_matrix.at[genre1, genre2] += 1
        co_occurrence_matrix.at[genre2, genre1] += 1
        
import numpy as np
# Set the diagonal to zero
np.fill_diagonal(co_occurrence_matrix.values, 0)
# remove the rows and columns with only zeros
co_occurrence_matrix = co_occurrence_matrix.loc[:, (co_occurrence_matrix != 0).any(axis=0)]
co_occurrence_matrix = co_occurrence_matrix.loc[(co_occurrence_matrix != 0).any(axis=1), :]
#sort columns by most occuring genre
co_occurrence_matrix = co_occurrence_matrix.sort_values(by=list(co_occurrence_matrix.columns), ascending=False)
#sort rows by most occuring genre
co_occurrence_matrix = co_occurrence_matrix.sort_values(by=list(co_occurrence_matrix.columns), ascending=False, axis=1)


# plot the co-occurrence matrix
import seaborn as sns
plt.figure(figsize=(15,10))
sns.heatmap(co_occurrence_matrix.iloc[:10,:10], cmap='Blues', annot=True, fmt='g')
plt.title('Co-occurrence matrix for the 10 most occuring genres')
plt.show()

# use preprocess function to preprocess genre_df['genre]
genre_df['genre'] = genre_df['genre'].apply(lambda x: [preprocess_text(genre) for genre in x])
#replace film, films, and movie, movies, cinema, 's with ""
genre_df['genre'] = genre_df['genre'].apply(lambda x: [re.sub(r'\b(film|films|movie|movies|cinema|\'s)\b', '', genre) for genre in x])
#remove trailing and leading spaces
genre_df['genre'] = genre_df['genre'].apply(lambda x: [genre.strip() for genre in x])

#reduce categories
unique_categories = list(set([category for sublist in genre_df['genre'] for category in sublist]))

#remove empty strings
unique_categories = [category for category in unique_categories if category != '']

unique_categories


#tokenize the categories
tokenized_categories = [category.split() for category in unique_categories]
#show word cloud of the categories
generate_wordcloud(Counter([word for sublist in tokenized_categories for word in sublist]), 'Wordcloud of categories')

# from sentence_transformers import SentenceTransformer
# from sklearn.cluster import KMeans

# # Define the SBERT model (distilbert-base-uncased)
# model = SentenceTransformer('distilbert-base-uncased')

# # Extract category names
# category_names = [category for category in unique_categories]

# # Encode the category names into SBERT embeddings
# category_embeddings = model.encode(category_names, convert_to_tensor=True)

# # Perform K-means clustering with n clusters
# kmeans = KMeans(n_clusters=15)
# kmeans.fit(category_embeddings)

# # Get the cluster labels for each category
# cluster_labels = kmeans.labels_

# # Create a dictionary to group categories by their cluster label
# clusters = {}
# for i, label in enumerate(cluster_labels):
#     if label not in clusters:
#         clusters[label] = []
#     clusters[label].append(category_names[i])

# # Print the movie categories grouped by clusters
# for cluster_label, cluster_categories in clusters.items():
#     print(f"Cluster {cluster_label + 1}:")
#     print(", ".join(cluster_categories))
#     print()

# #Create a DataFrame from the 'clusters' dictionary
# df_clusters = pd.DataFrame.from_dict(clusters, orient='index')
# df_clusters = df_clusters.transpose()

# # Write the DataFrame to a CSV file with a semicolon (;) separator
# df_clusters.to_csv('Data/MovieSummaries/genre_clusters_embedding.csv', index=False, sep=';')

#show balance of clusters
plt.figure(figsize=(15,10))
plt.bar(clusters.keys(), [len(cluster) for cluster in clusters.values()])
plt.title('Number of categories per cluster')
plt.xlabel('Cluster')
plt.ylabel('Number of categories')
plt.show()

#clustering with TF-IDF (Term Frequency-Inverse Document Frequency) vectorization process in the provided code converts textual data (in this case, movie category names) into numerical vectors. 


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# Extract category names
category_names = [category for category in unique_categories]

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the category names using TF-IDF
tfidf_matrix = tfidf_vectorizer.fit_transform(category_names)

# Perform K-means clustering with n clusters
kmeans = KMeans(n_clusters=15)
kmeans.fit(tfidf_matrix)

# Get the cluster labels for each category
cluster_labels = kmeans.labels_

# Create a dictionary to group categories by their cluster label
clusters = {}
for i, label in enumerate(cluster_labels):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(category_names[i])

# Print the movie categories grouped by "over genres"
for cluster_label, cluster_categories in clusters.items():
    print(f"Over Genre {cluster_label + 1}:")
    print(", ".join(cluster_categories))
    print()

# #Create a DataFrame from the 'clusters' dictionary
# df_clusters = pd.DataFrame.from_dict(clusters, orient='index')

# #sort by index desc
# df_clusters = df_clusters.sort_index(ascending=True)
# #transpose
# df_clusters = df_clusters.transpose()

# # Write the DataFrame to a CSV file with a semicolon (;) separator
# df_clusters.to_csv('Data/MovieSummaries/genre_clusters_tf-idf.csv', index=False, sep=';')

#load the clusters from the csv file
df_clusters = pd.read_csv('Data/MovieSummaries/genre_clusters_tf-idf.csv', sep=';')

#define names for the clusters according to the categories
# 0 = miscellaneous
# 1 = thriller
# 2 = buddy picture
# 3 = adventre

#show balance of clusters
plt.figure(figsize=(15,10))
plt.bar(clusters.keys(), [len(cluster) for cluster in clusters.values()])
plt.title('Number of categories per cluster')
plt.xlabel('Cluster')
plt.ylabel('Number of categories')
plt.show()

# #do embedding with word2vec

# from gensim.models import Word2Vec
# import gensim.downloader as api
# from sklearn.cluster import KMeans
# import numpy as np


# # Load a pre-trained Word2Vec model (for example, using gensim)
# # Replace 'path_to_pretrained_model' with the actual path to your Word2Vec model file
# # if model does not exist, download 

# info = api.info()  # show info about available models/datasets
# word2vec_model = api.load("glove-wiki-gigaword-300")  # download the model and return as object ready for use

# # Create an average vector for each movie category
# category_vectors = []
# for category in unique_categories:
#     words = category.split()  # Split the category name into words
#     vectors = [word2vec_model[word] for word in words if word in word2vec_model]
#     if vectors:
#         category_vector = np.mean(vectors, axis=0)  # Compute the average vector
#         category_vectors.append(category_vector)

# # Convert the list of category vectors to a NumPy array
# X = np.array(category_vectors)

# # Perform K-means clustering with n clusters (adjust as needed)
# kmeans = KMeans(n_clusters=15)
# kmeans.fit(X)

# # Get the cluster labels for each category
# cluster_labels = kmeans.labels_

# # Create a dictionary to group categories by their cluster label
# clusters = {}
# for i, label in enumerate(cluster_labels):
#     if label not in clusters:
#         clusters[label] = []
#     clusters[label].append(unique_categories[i])

# # Print the movie categories grouped by clusters
# for cluster_label, cluster_categories in clusters.items():
#     print(f"Cluster {cluster_label + 1}:")
#     print(", ".join(cluster_categories))
#     print()


# use the clusters to reduce the categories
# Create a dictionary to map each category to its cluster label
category_to_cluster = {}
for cluster_label, cluster_categories in clusters.items():
    for category in cluster_categories:
        category_to_cluster[category] = cluster_label
        

category_to_cluster


# Create 13 empty columns for the clusters
for i in range(15):
    genre_df[i] = 0
    
# Iterate through rows and update the cluster columns
for i, row in genre_df.iterrows():
    for category in row['genre']:
        genre_df.at[i, category_to_cluster[category]] = 1
        
genre_df.head()

# count occurences of genres in columsn 0-14
genre_df[range(15)].sum().sort_values(ascending=False)


# Step 1: Create a list of all unique genre labels
# if 1 is in column 0-14, replace with the column name
genre_df['genre_hm'] = genre_df[range(15)].apply(lambda x: [x.index[i] for i in range(len(x)) if x[i]==1], axis=1)

# Step 2: Create an empty co-occurrence matrix
co_occurrence_matrix = pd.DataFrame(0, columns=genre_df['genre_hm'], index=genre_df['genre_hm'])

# Step 3: Iterate through rows and update the matrix
for genres_str in genre_df['genre_hm']:
    genres = [int(genre) for genre in genres_str.strip('[]').split(',')]
    for genre1, genre2 in combinations(genres, 2):
        co_occurrence_matrix.at[genre1, genre2] += 1
        co_occurrence_matrix.at[genre2, genre1] += 1

# Set the diagonal to zero
np.fill_diagonal(co_occurrence_matrix.values, 0)

# Remove the rows and columns with only zeros
co_occurrence_matrix = co_occurrence_matrix.loc[:, (co_occurrence_matrix != 0).any(axis=0)]
co_occurrence_matrix = co_occurrence_matrix.loc[(co_occurrence_matrix != 0).any(axis=1), :]

# Sort columns by most occurring genre
co_occurrence_matrix = co_occurrence_matrix.sort_values(by=list(co_occurrence_matrix.columns), ascending=False)

# Sort rows by most occurring genre
co_occurrence_matrix = co_occurrence_matrix.sort_values(by=list(co_occurrence_matrix.columns), ascending=False, axis=1)

# Plot the co-occurrence matrix
plt.figure(figsize=(15, 10))
sns.heatmap(co_occurrence_matrix.iloc[:10, :10], cmap='Blues', annot=True, fmt='g')
plt.title('Co-occurrence matrix for the 10 most occurring genres')
plt.show()