import pandas as pd
import regex as re
import string
from matplotlib import pyplot as plt
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

seed = 1

df = pd.read_pickle("./Data/MovieSummaries/plots_genres.pkl")
# Remove rows with empty genre
df = df[~df['genre'].apply(lambda x: len(x) == 0)]
#remove rows with nan genre
df = df[~df['genre'].isna()]

print(len(df))

# remove rows with less than count_threshold=50 movies in genre
def process_less_than_threshold(df, count_threshold=50):
    df_exploded = df.explode('genre')

    # Count the occurrences of each string
    result = df_exploded['genre'].value_counts().reset_index()

    # Rename the columns for better clarity
    result.columns = ['String', 'Count']

    # Filter strings with count less than the threshold
    less_than_threshold = result[result['Count'] < count_threshold]
    less_than_threshold_list = less_than_threshold['String'].tolist()

    print(less_than_threshold_list)

    # Filter the original dataframe to remove rows containing less frequent genres
    df[df['genre'].apply(lambda x: any(target in x for target in less_than_threshold_list))]

    # Remove less frequent genres from the 'genre' column
    df['genre'] = df['genre'].apply(lambda x: [item for item in x if item not in less_than_threshold_list])

    # Remove rows with empty genre again
    df = df[~df['genre'].apply(lambda x: len(x) == 0)]

    return df

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

# additional preprocessing for genre, remove words, which confuse the clustering
def preprocess_genre(genre):
    genre = [preprocess_text(genre) for genre in genre]
    genre = [re.sub(r'\b(film|films|movie|movies|cinema|new|\'s)\b', '', genre) for genre in genre]
    genre = [genre.strip() for genre in genre]
    return genre

df['genre'] = df['genre'].apply(lambda x: preprocess_genre(x))

# Clustering with TF-IDF (Term Frequency-Inverse Document Frequency) vectorization process in the provided code converts textual data (in this case, movie category names) into numerical vectors. 
# Using k-means clustering, the categories are grouped into 15 clusters. 15 clusters were chosen after experimenting with different values of k.

#reduce categories
unique_categories = list(set([category for sublist in df['genre'] for category in sublist]))

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

#show balance of clusters
plt.figure(figsize=(15,10))
plt.bar(clusters.keys(), [len(cluster) for cluster in clusters.values()])
plt.title('Number of categories per cluster')
plt.xlabel('Cluster')
plt.ylabel('Number of categories')
plt.show()

# #Create a DataFrame from the 'clusters' dictionary
#if genre_clusters_tf-idf.csv already exists, skip this step
if not os.path.isfile('Data/MovieSummaries/genre_clusters_tf-idf.csv'):
    df_clusters = pd.DataFrame.from_dict(clusters, orient='index')

    #sort by index desc
    df_clusters = df_clusters.sort_index(ascending=True)
    #transpose
    df_clusters = df_clusters.transpose()

    # Write the DataFrame to a CSV file with a semicolon (;) separator
    df_clusters.to_csv('Data/MovieSummaries/genre_clusters_tf-idf.csv', index=False, sep=';')

#load the clusters from the csv file
df_clusters = pd.read_csv('Data/MovieSummaries/genre_clusters_tf-idf.csv', sep=';')

# rename clusters to more meaningful names

new_column_names = {
    '0': 'suspense',
    '1': 'drama',
    '2': 'war',
    '3': 'western',
    '4': 'horror',
    '5': 'adventure',
    '6': 'family',
    '7': 'action',
    '8': 'crime',
    '9': 'gay',
    '10': 'political',
    '11': 'buddy',
    '12': 'miscellaneous',
    '13': 'comedy',
    '14': 'childrens'
}

# Rename the columns
df_clusters.rename(columns=new_column_names, inplace=True)

df_clusters
#remove column "miscellaneous"
df_clusters.drop('miscellaneous', axis=1, inplace=True)

#replace in df the genre categories with the cluster names
# Create a dictionary to map the clusters to the genre categories
cluster_genre_map = {}
for cluster_label, cluster_categories in df_clusters.items():
    for category in cluster_categories:
        cluster_genre_map[category] = cluster_label

# explode df['genre'] again and do the same as above
df_exploded = df.explode('genre')
#if key is in cluster_genre_map, replace with value
df_exploded['genre'] = df_exploded['genre'].apply(lambda x: cluster_genre_map[x] if x in dict(cluster_genre_map) else x)

#remove duplicates
df_exploded.drop_duplicates(inplace=True)

#count occurces of each genre
df_exploded['genre'].value_counts()
#show as txt
df_exploded['genre'].value_counts().to_csv('Data/MovieSummaries/genre_count.txt', sep=';')

# aggregate the rows again

df = df_exploded.groupby('id').agg({'text': 'first', 'title': 'first', 'genre': list}).reset_index()
print(len(df))
# 341 because 342 is the last aggregated cluster
df = process_less_than_threshold(df, count_threshold=341)
print(len(df))
#count occurces of each genre
df_exploded = df.explode('genre')
df_exploded['genre'].value_counts()

# balance of most occuring genres/least occuring genres
19406/342

print(len(df))


#plot the number of occurences of each genre
plt.figure(figsize=(15,10))
plt.bar(df_exploded['genre'].value_counts().index, df_exploded['genre'].value_counts())
plt.title('Number of occurences of each genre')
plt.xlabel('Genre')
#turn x labels 90 degrees
plt.xticks(rotation=90)
plt.ylabel('Number of occurences')
plt.show()
#print number of unique genres
print(len(df_exploded['genre'].unique()))

#make coocurence matrix for genres

# Create a list of all genres
genres = list(set([genre for sublist in df['genre'] for genre in sublist]))

if not os.path.isfile('Data/MovieSummaries/cooccurrence_matrix.pkl'):
    # Create an empty co-occurrence matrix
    cooccurrence_matrix = pd.DataFrame(index=genres, columns=genres)

    # Fill in the missing values with zeros
    cooccurrence_matrix.fillna(0, inplace=True)

    # Calculate the co-occurrence matrix
    for genre_list in df['genre']:
        for genre_1 in genre_list:
            for genre_2 in genre_list:
                if genre_1 != genre_2:
                    cooccurrence_matrix.loc[genre_1, genre_2] += 1

    #save the cooccurence matrix to a pickle file
    cooccurrence_matrix.to_pickle("./Data/MovieSummaries/cooccurrence_matrix.pkl")

#load the cooccurence matrix from the pickle file
cooccurrence_matrix = pd.read_pickle("./Data/MovieSummaries/cooccurrence_matrix.pkl")

#plot the cooccurence matrix
import seaborn as sns
plt.figure(figsize=(15,10))
sns.heatmap(cooccurrence_matrix, annot=False, cmap='Blues')
plt.title('Co-occurrence matrix of genres')
plt.xlabel('Genre')
plt.ylabel('Genre')
plt.show()

#save the df to a pickle file
if not os.path.isfile('./Data/MovieSummaries/plots_genres_reduced_to_60.pkl'):
    df.to_pickle("./Data/MovieSummaries/plots_genres_reduced_to_60.pkl")

# Perform a train-test-split and save dfs as pickle files
if not os.path.isfile('Data/MovieSummaries/train_plots_genres_reduced_to_60.pkl') or not os.path.isfile('Data/MovieSummaries/test_plots_genres_reduced_to_60.pkl'):
    train_reduced, test_reduced = train_test_split(df, test_size=0.2, random_state=4)
    train_reduced.to_pickle("./Data/MovieSummaries/train_plots_genres_reduced_to_60.pkl")
    test_reduced.to_pickle("./Data/MovieSummaries/test_plots_genres_reduced_to_60.pkl")

# create a more balanced dataset
# for each genre a random sample of the amount of the least frequent genre is taken which in this case is 342

# Create an empty DataFrame
df_balanced = pd.DataFrame(columns=['id', 'text', 'title', 'genre'])

#get the amount of least frequent genre
least_frequent_genre = df_exploded['genre'].value_counts().min()

# Iterate over each genre
for genre in genres:
    # Filter out rows containing the genre
    df_genre = df[df['genre'].apply(lambda x: genre in x)]


    # Randomly select 342 rows from the filtered DataFrame
    if len(df_genre) >= least_frequent_genre:
        df_genre = df_genre.sample(least_frequent_genre, random_state=42)
    else:
        df_genre = df_genre.sample(len(df_genre), random_state=42)


    # Append the selected rows to the DataFrame
    df_balanced = pd.concat([df_balanced, df_genre])



# Shuffle the DataFrame
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Print the number of movies per genre
df_balanced['genre'].value_counts()



from collections import Counter
flattened_list = [item for sublist in df_balanced['genre'] for item in sublist]

# Step 2: Count the occurrences of each element
element_counts = Counter(flattened_list)
print(len(element_counts))
sorted(element_counts.items(), key=lambda x:x[1], reverse=True)

#print the dimensions of the balanced dataset
print(len(df_balanced))

#sort the genres by number of occurences
element_counts = dict(sorted(element_counts.items(), key=lambda x:x[1], reverse=True))


#redo, but turn the ticks 90 degrees
plt.figure(figsize=(15,10))
plt.bar(element_counts.keys(), element_counts.values())
plt.xticks(rotation=90)
plt.title('Number of occurrences of each genre')
plt.xlabel('Genre')
plt.ylabel('Number of occurrences')
plt.show()

# balance of most occuring genres/least occuring genres
# before 19406/342 = 56.74
# the amount of data is reduced from 41549 to 19494

9289/505

#do the cooccurence matrix again
if not os.path.isfile('Data/MovieSummaries/cooccurrence_matrix_balanced.pkl'):

    # Create an empty co-occurrence matrix
    cooccurrence_matrix = pd.DataFrame(index=genres, columns=genres)

    # Fill in the missing values with zeros
    cooccurrence_matrix.fillna(0, inplace=True)

    # Calculate the co-occurrence matrix
    for genre_list in df_balanced['genre']:
        for genre_1 in genre_list:
            for genre_2 in genre_list:
                if genre_1 != genre_2:
                    cooccurrence_matrix.loc[genre_1, genre_2] += 1


    #save the cooccurence matrix to a pickle file
    cooccurrence_matrix.to_pickle("./Data/MovieSummaries/cooccurrence_matrix_balanced.pkl")
    
#load the cooccurence matrix from the pickle file
cooccurrence_matrix = pd.read_pickle("./Data/MovieSummaries/cooccurrence_matrix_balanced.pkl")

#plot the cooccurence matrix
import seaborn as sns
plt.figure(figsize=(15,10))
sns.heatmap(cooccurrence_matrix, annot=False, cmap='Blues')
plt.title('Co-occurrence matrix of genres')
plt.xlabel('Genre')
plt.ylabel('Genre')
plt.show()

#write the balanced df to a pickle file if not already exists
if not os.path.isfile('Data/MovieSummaries/plots_genres_balanced.pkl'):
    df_balanced.to_pickle("./Data/MovieSummaries/plots_genres_balanced.pkl")

# Perform a train-test-split on the balanced data and save dfs as pickle files
if not os.path.isfile('Data/MovieSummaries/train_plots_genres_balanced.pkl') or not os.path.isfile('Data/MovieSummaries/test_plots_genres_balanced.pkl'):
    train_balanced, test_balanced = train_test_split(df_balanced, test_size=0.2, random_state=6)
    train_balanced.to_pickle("./Data/MovieSummaries/train_plots_genres_balanced.pkl")
    test_balanced.to_pickle("./Data/MovieSummaries/test_plots_genres_balanced.pkl")