import json
import nltk
import re
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
nltk.download('stopwords')
pd.set_option('display.max_colwidth', 300)

# Read data
df = pd.read_pickle('Data\MovieSummaries\plots_genres_reduced_to_60.pkl')
df_train = pd.read_pickle('Data/MovieSummaries/train_plots_genres_reduced_to_60.pkl')
df_test = pd.read_pickle('Data/MovieSummaries/test_plots_genres_reduced_to_60.pkl')

# Preprocessing functions
def clean_text(text):
    text = re.sub("\'", "", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = " ".join(text.split())
    text = text.lower()
    return text

def remove_stopwords(text):
    no_stop = []
    for i in text.split():
        if i not in stopwords:
            no_stop.append(i)
    return " ".join(no_stop)

def freq_plot(text):
    
    words = " ".join([x for x in text])
    words = words.split()
    fdist = nltk.FreqDist(words)
    return fdist



#Dataset overview
df.reset_index(inplace=True)
df.head()

df_train.reset_index(inplace=True)
df_train.head()

df_test.reset_index(inplace=True)
df_test.head()

#clean text
df["text"] = df["text"].apply(lambda x : clean_text(x))
df_train["text"] = df_train["text"].apply(lambda x : clean_text(x))
df_test["text"] = df_test["text"].apply(lambda x : clean_text(x))

# Get stopwords
#if stopwords not downloaded
#
stopwords = set(stopwords.words("english"))
df["text"] = df["text"].apply(lambda x : remove_stopwords(x))
df_train["text"] = df_train["text"].apply(lambda x : remove_stopwords(x))
df_test["text"] = df_test["text"].apply(lambda x : remove_stopwords(x))

# Binarize genres
from sklearn.preprocessing import MultiLabelBinarizer
multilabel_bina = MultiLabelBinarizer()
multilabel_bina.fit(df["genre"])

ytrain = multilabel_bina.transform(df_train["genre"])
yval = multilabel_bina.transform(df_test["genre"])
xtrain = df_train["text"]
xval = df_test["text"]

# TF-IDF Vectorizer
tfidf_vect = TfidfVectorizer(max_df= 0.8, max_features= 10000)
#xtrain, xval, ytrain, yval = train_test_split(df["text"], y, test_size = 0.2, random_state= 9)
xtrain_tfidf = tfidf_vect.fit_transform(xtrain)
xval_tfidf = tfidf_vect.transform(xval)

#make list aff all genre
genre_list = []
for i in range(len(df["genre"])):
    genre_list.append(df["genre"].iloc[i][0])

#unique genre
unique_genre = list(set(genre_list))
len(unique_genre)

# Documents Example with 5 Plots
documents = df["text"][0:5].reset_index(drop=True)

# Term Frequency-Inverse Document Frequency vectorizer
Tfid_vect = TfidfVectorizer()

# Transform the documents using the vectorizer
documents_vect = Tfid_vect.fit_transform(documents)

# Convert the transformed for better visability. Normally sparse matrix
#df = pd.DataFrame(documents_vect.toarray(), columns=Tfid_vect.get_feature_names_out())

#df


# Logistic Regression
logistic_mod = LogisticRegression()

# Separate binary classifier for each class label for multi-label classification
onevsall = OneVsRestClassifier(logistic_mod) 

# Train model
onevsall.fit(xtrain_tfidf, ytrain)

# Predict and evaluate
y_pred = onevsall.predict(xval_tfidf)
print(classification_report(yval, y_pred))

# Sample predictions
def new_val(x):  
    x = remove_stopwords(clean_text(x))
    x_vec = tfidf_vect.transform([x])
    x_pred = onevsall.predict(x_vec)
    return multilabel_bina.inverse_transform(x_pred)



#write classification report to csv
report = classification_report(yval, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv('Results/Metrics/classification_report_reduced_to_60_log_reg.csv', index=True, sep=';')

for i in range(5): 
    k = xval.sample(1).index[0] 
    print("Movie: ", df_test['title'][k], "\nPredicted genre: ", new_val(xval[k]))
    print("Actual genre: ",df_test['genre'][k], "\n")

#Apply it to all Film
f = pd.DataFrame(xval)
f['generated_genre'] = pd.DataFrame(xval)['text'].apply(new_val)
f= f.merge(df_test, left_index=True, how='left', right_index=True).reset_index(drop=True)
f.head(10)

# do with naive bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier

# Create a classifier: a support vector classifier
classifier = OneVsRestClassifier(GaussianNB())

# Train the classifier on the training set
classifier.fit(xtrain_tfidf.toarray(), ytrain)

# Predict and evaluate
y_pred = classifier.predict(xval_tfidf.toarray())
print(classification_report(yval, y_pred))

# Sample predictions
def new_val(x):  
    x = remove_stopwords(clean_text(x))
    x_vec = tfidf_vect.transform([x])
    x_pred = classifier.predict(x_vec.toarray())
    return multilabel_bina.inverse_transform(x_pred)

for i in range(5): 
    k = xval.sample(1).index[0] 
    print("Movie: ", df_test['title'][k], "\nPredicted genre: ", new_val(xval[k]))
    print("Actual genre: ",df_test['genre'][k], "\n")

#write classification report to csv
report = classification_report(yval, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv('Results\Metrics\classification_report_reduced_60_NB.csv', index=True, sep=';')

#do state vector machine
import sklearn.svm as svm
from sklearn.multiclass import OneVsRestClassifier

# Create a classifier: a support vector classifier
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=0))

# Train the classifier on the training set
classifier.fit(xtrain_tfidf, ytrain)

# Predict and evaluate
y_pred = classifier.predict(xval_tfidf)
print(classification_report(yval, y_pred))

# Sample predictions
def new_val(x):  
    x = remove_stopwords(clean_text(x))
    x_vec = tfidf_vect.transform([x])
    x_pred = classifier.predict(x_vec)
    return multilabel_bina.inverse_transform(x_pred)

for i in range(5): 
    k = xval.sample(1).index[0] 
    print("Movie: ", df['title'][k], "\nPredicted genre: ", new_val(xval[k]))
    print("Actual genre: ",df['genre'][k], "\n")

#clean lists
for i in range(len(f["generated_genre"])):
    f["generated_genre"][i] = [x for x in f["generated_genre"][i][0]]

#sort lists in genre alphabetically
f["genre"] = f["genre"].apply(lambda x: sorted(x))


###prepare for library

# Create a set 
unique_genre = set()
for genres in f["genre"]:
    unique_genre.update(genres)
unique_genre = list(unique_genre)


#Numpy Version problems solved
from mlcm_changed import mlcm

conf_mat,normal_conf_mat = mlcm.cm(yval,y_pred,False)

one_vs_rest = mlcm.stats(conf_mat,False)

colnames = unique_genre + ['NPL']
rownames = unique_genre + ['NTL']


#make dataframe from confusion matrix
df_confusion = pd.DataFrame(normal_conf_mat, columns=colnames, index=rownames)
df_confusion.drop('NTL', inplace=True)
df_confusion.drop('NPL', axis=1, inplace=True)

#plot df_confusion
plt.figure(figsize = (20,10))
sns.heatmap(df_confusion, annot=True, cmap="YlGnBu")
plt.title('Confusion Matrix', fontsize=20)
plt.xlabel('Predicted Genres', fontsize=15)
plt.ylabel('True Genres', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


plt.figure(figsize=(25, 25))


n=32
#  first n
for i in range(n):  
    #column names
    colnames = ['rest', unique_genre[i]]
    
    #DataFrame
    df1 = pd.DataFrame(one_vs_rest[i], columns=colnames, index=['rest' , unique_genre[i]]).transpose()

    ax = plt.subplot(int(n/4), 4, i + 1)
    
    #heatmap
    sns.heatmap(df1, annot=True, fmt="d", cmap="Blues", square=True, linewidths=.5, cbar_kws={"shrink": .75})

    # Titles and labels
    ax.set_title(f'{unique_genre[i]} vs Rest', fontsize=12)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=9)

plt.tight_layout()
plt.show()
