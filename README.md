# NLP_Project
NLP on the CMU Movie Summary Corpus.

1. FINAL_Data_extraction -> produces plots_genres.pkl 
   - visualistion of the data
      - wordclouds
      - stopwords
2. reducing_categories -> takes plots_genres.pkl and produces plots_genres_reduced_to_60.pkl and plots_genres_balanced
   - plots_genres_reduced_to_60 reduces the categories to 60 - 41549 entries balance of most occuring genres/least occuring genres 19406/342 = 56.74
      - reduces via TF-IDF and K-Means, throws away the least occuring categories
   - plots_genres_balanced has the same amount of categories - 19494 entries balance of most occuring genres/least occuring genres 9289/505 = 18.39
      - takes the amount of the least occuring genre, and takes a random sample for each genre with the same amount of entries
   - creates train and test data for the reduced and reduced and balanced dataset in 0.8/0.2 ratio
   - Cooccurence Matrix
   - Barcharts
3. Logistic_Regression_try_balanced/Logistic_Regression_try_reduced_to_60
   beinhaltet categorien zusammenfassen, balacnieren & Logistische Regression, Naive Bayes, SVM
   -> takes plots_genres_reduced_to_60/plots_genres_balanced 
4. Transformer_classification_v2


Mentions: Cleaning_and_EDA -> Wordcloud eventuell noch einbauen