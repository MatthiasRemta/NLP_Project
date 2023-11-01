# NLP_Project
NLP on the CMU Movie Summary Corpus.
Submitted by: Erlacher Felix, Remta Matthias, Schachinger Gabriel & Zechmeister Anna-Christina

Order of the notebooks provided:

1. Data_extraction_preprocessing_EDA
2. reducing_categories
3. Logistic_Regression_try_balanced/Logistic_Regression_try_reduced_to_60
4. Transformer_classification_v2


Details about the notebook contents:

1. Data_extraction_preprocessing_EDA
   - Output: plots_genres.pkl and visualisations
   - Loads, preprocesses and merges the relevant data
   - Visualistion of the data (movie summaries and genre information)

2. reducing_categories
   - Output: Takes plots_genres.pkl and produces plots_genres_reduced_to_60.pkl, plots_genres_balanced as well as the regarding train and test sets
   - plots_genres_reduced_to_60 reduces the categories to 60 - 41549 entries balance of most occuring genres/least occuring genres 19406/342 = 56.74
      - reduces via TF-IDF and K-Means, throws away the least occuring categories
   - plots_genres_balanced has the same amount of categories - 19494 entries balance of most occuring genres/least occuring genres 9289/505 = 18.39
      - takes the amount of the least occuring genre, and takes a random sample for each genre with the same amount of entries
   - creates train and test data for the reduced and reduced and balanced dataset in 0.8/0.2 ratio
   - Cooccurence Matrix
   - Barcharts
     
3. Logistic_Regression_try_balanced/Logistic_Regression_try_reduced_to_60
   - Output: Logistic regression, naive bayes and svm models
   - includes reduced categories and balanced data for logistic regression, Naive Bayes and SVM
   - takes plots_genres_reduced_to_60/plots_genres_balanced
   - saves results csv in results folder

4. Transformer_classification_v6
   - Output: Transformer model
