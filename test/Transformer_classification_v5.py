# for Google-Colab

# packages
!pip install transformers &> /dev/null
!pip install datasets &> /dev/null
!pip install evaluate &> /dev/null
!pip install accelerate -U &> /dev/null
!pip install huggingface_hub &> /dev/null

# if you want to upload models to huggingface
from huggingface_hub import notebook_login

notebook_login()

# datasets
!wget https://raw.githubusercontent.com/MatthiasRemta/NLP_Project/main/Data/MovieSummaries/train_plots_genres_reduced_to_60.pkl &> /dev/null
!wget https://raw.githubusercontent.com/MatthiasRemta/NLP_Project/main/Data/MovieSummaries/test_plots_genres_reduced_to_60.pkl &> /dev/null

!wget https://raw.githubusercontent.com/MatthiasRemta/NLP_Project/main/Data/MovieSummaries/train_plots_genres_balanced.pkl &> /dev/null
!wget https://raw.githubusercontent.com/MatthiasRemta/NLP_Project/main/Data/MovieSummaries/test_plots_genres_balanced.pkl &> /dev/null

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, EvalPrediction
from transformers import TextClassificationPipeline
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report
from datasets import Dataset
import torch
import accelerate
import pandas as pd
import regex as re
import string
import numpy as np
import evaluate
import pickle
import os

# load the data
# df_train_raw = pd.read_pickle('train_plots_genres_reduced_to_60.pkl')
# df_test_raw = pd.read_pickle('test_plots_genres_reduced_to_60.pkl')
df_train_raw = pd.read_pickle('train_plots_genres_balanced.pkl')
df_test_raw = pd.read_pickle('test_plots_genres_balanced.pkl')

# Specify mappings (id -> label) and (label -> id)
genres =[]
for row in df_train_raw['genre']:
  for genre in row:
    genres.append(genre)

unique_genres = []

for item in genres:
    if item not in unique_genres:
        unique_genres.append(item)

label2id = dict([(tuple[1], tuple[0]) for tuple in enumerate(unique_genres)])
id2label = dict([(label2id[key], key) for key in label2id])

# look at the mappings
print(label2id)
print(id2label)

# encode the labels as vector
def labels_to_binary(labels, unique_labels):
    binary_vector = np.zeros(len(unique_labels))
    for label in labels:
        binary_vector[unique_labels[label]] = 1
    return binary_vector


labels = []
for ele in df_train_raw['genre']:
    labels.append(labels_to_binary(ele, label2id))

df_train_raw['labels'] = labels

labels = []
for ele in df_test_raw['genre']:
    labels.append(labels_to_binary(ele, label2id))

df_test_raw['labels'] = labels

# convert to dataset
df_train = Dataset.from_pandas(df_train_raw)
df_test = Dataset.from_pandas(df_test_raw)

# define tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased",
                                          truncation=True,
                                          padding='max_length',
                                          max_length=512)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# function for tokenization
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=tokenizer.model_max_length)

# preprocess the plot summaries
df_train_tokenized = df_train.map(preprocess_function)

# Create data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# define metrics
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result

# Create train/validation split
df_train_tokenized = df_train_tokenized.train_test_split(test_size=0.2)

# define model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
    problem_type='multi_label_classification'
    )

# alternatively, load checkpoint from previous finetuning
model = AutoModelForSequenceClassification.from_pretrained("matthiasr/genre_pred_model_balanced")

# check whether cuda is available
print(torch.cuda.is_available())

# create folder for checkpoints
path_wd = os.getcwd()

if not os.path.exists(path_wd + '/genre_pred_model'):
  os.mkdir(path_wd + '/genre_pred_model')

# finetune model
training_args = TrainingArguments(
    output_dir="genre_pred_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=df_train_tokenized["train"],
    eval_dataset=df_train_tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

def predict_genres(text, tokenizer, model, id2label, threshold=0.5):
  # Tokenize the text and get model predictions
  inputs = tokenizer(text, truncation=True, padding='max_length', max_length=tokenizer.model_max_length, return_tensors="pt")
  outputs = model(**inputs)

  # Get the predicted logits (scores) for each label
  logits = outputs.logits
  sigmoid = torch.nn.Sigmoid()
  probs = sigmoid(logits)

  # Apply threshold to determine the labels
  predicted_labels = (probs > threshold).tolist()[0]

  # convert ids to actual labels
  indices = [i for i, x in enumerate(predicted_labels) if x]
  genres = [id2label[x] for x in indices]
  return genres


# put model into eval mode
model.eval()

# pipeline for Inference
# this takes quite some time
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=None)

tokenizer_kwargs = {'padding':True, 'truncation':True, 'max_length':512}
pred = pipe(df_test_raw['text'].to_list(), **tokenizer_kwargs)

# load saved scores, faster than predicting each time
!wget https://raw.githubusercontent.com/MatthiasRemta/NLP_Project/main/Data/MovieSummaries/transformer_balanced_scores.pkl &> /dev/null

with open('transformer_balanced_scores.pkl', 'rb') as f:
    pred = pickle.load(f)

threshold = 0.5

pred_list = []
for movie in pred:
  score = []
  for label in movie:
    if label['score'] > threshold:
      result = 1.0
    else:
      result = 0.0
    score.append(result)
  pred_list.append(score)

print(classification_report(pred_list, df_test['labels']))