import pandas as pd
import requests
import tarfile
import json
import regex as re
import string

from wordcloud import WordCloud
from textwrap import wrap
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import seaborn as sns

# Check if MovieSummaries.tar.gz exists, download and decompress
try:
    open('MovieSummaries.tar.gz')
except FileNotFoundError:
    print("Downloading MovieSummaries.tar.gz, decompressing, and saving to ./Data")
    url = 'http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz'
    r = requests.get(url)
    with open('MovieSummaries.tar.gz', 'wb') as f:
        f.write(r.content)
    # Decompress downloaded data
    with tarfile.open('MovieSummaries.tar.gz') as f:
        f.extractall('./Data')

# Read data - files are eighter saved as .txt or as tsv-format (tab separated)
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


# If plot_summaries.csv exists, read it, else create it
try:
    df_plot_summary = pd.read_csv('Data/MovieSummaries/plot_summaries.csv').iloc[: , 1:]
except FileNotFoundError:
    print('Creating plot_summaries.csv')
    # Preprocess the text
    df_plot_summary[1] = df_plot_summary[1].apply(preprocess_text)
    # Save the preprocessed data
    df_plot_summary.to_csv('Data/MovieSummaries/plot_summaries.csv')

# Load movie summaries and only keep ID and text columns
plot = pd.read_csv('Data/MovieSummaries/plot_summaries.csv')
plot_df = plot.drop('Unnamed: 0', axis=1).rename(columns={"0": "id", "1": "text"})

# Load movie metadata and only keep ID, movie title and genre json information
df_movie_metadata = pd.read_csv('Data/MovieSummaries/movie.metadata.tsv', delimiter='\t', header=None)
genre_df = df_movie_metadata[[0,2,8]]

# Restructuring genre jsons to genre lists 
genre_df['genre'] = genre_df[8].apply(lambda x: list(json.loads(x).values()))
genre_df.columns = ['movie_id','title','genre_json','genre']
genre_df = genre_df.drop(columns=['genre_json'])

# Merging movie summaries and genres
df = plot_df.merge(genre_df, left_on='id', right_on='movie_id', how='left')
df = df.drop('movie_id', axis=1)

# There exsist 99 movies with movie summaries but no metadata available. These are removed.
df_length_old = len(df)
df.dropna(inplace=True)

# There furthermore exsist 411 movies with empty genre lists. These are also removed.
df = df[~df['genre'].apply(lambda x: len(x) == 0)]

print("Number of removed rows: ", df_length_old-len(df))

# Final dataset
df.head()

# Saving the final file in pickle format
df.to_pickle("./Data/MovieSummaries/plots_genres.pkl")

# Get english stopwords
stopwords = set(stopwords.words("english"))

# Function to remove stopwords
def remove_stopwords(text):
    no_stop = []
    for i in text.split():
        if i not in stopwords:
            no_stop.append(i)
    return " ".join(no_stop)

# Function to get a frequency plot
def freq_plot(text):
    words = " ".join([x for x in text])
    words = words.split()
    fdist = nltk.FreqDist(words)
    return fdist

# Function for generating word clouds
def generate_wordcloud(data,title):
  wc = WordCloud(width=400, height=330, max_words=150,colormap="Dark2").generate_from_frequencies(data)
  plt.figure(figsize=(10,8))
  plt.imshow(wc, interpolation='bilinear')
  plt.axis("off")
  plt.title('\n'.join(wrap(title,60)),fontsize=13)
  plt.show()


# Creating document term matrix
cv = CountVectorizer(analyzer='word')
data = cv.fit_transform(df['text'].apply(remove_stopwords))  # returns sparse DTM
# Extract one row (film) from the DTM and expand to all rows (sparse -> dense)
df_dtm = pd.DataFrame(data.getrow(1).toarray(), columns=cv.get_feature_names_out())

# Plot wordcloud
generate_wordcloud(df_dtm.iloc[0], 'Wordcloud')

# Creating a frequency plot
fdist = freq_plot(df["text"])
words_df = pd.DataFrame.from_dict(fdist, orient="index")
words_df = words_df.reset_index()
words_df.columns = ["Word","Count"]

plt.figure(figsize=(12,12))
sns.barplot(data= words_df.sort_values(by="Count",ascending= False).iloc[:20, :], x = "Count", y= "Word")