import pandas as pd
import re
import nltk
import joblib
import json
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from celery import Celery
from celery.schedules import crontab
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from app import reload_models

# nltk.download('punkt')

app = Celery('tasks', broker='redis://localhost:6379/0')

json_data_path = 'data.json'

tagalog_stopwords = [
    "ako", "ay", "ang", "na", "sa", "kay", "ni", "mga", "ng", "ngayon", "ito", "ka", "sila", "ka", "ngayo'y", "kapag", "kung", "pa", "para", "saka", "siya", "siyang", "sya", "sya'y", "taon", "tayo", "tulad", "un", "wala"
]

def get_new_training_data(json_data):
    # Get the counts of 1s and 0s in json_data['predictions']
    count_1s = sum(1 for _, prediction in json_data['predictions'] if prediction == 1)
    count_0s = sum(1 for _, prediction in json_data['predictions'] if prediction == 0)

    # print(json_data['predictions'])
    # print(count_1s)
    # print(count_0s)

    predictions = json_data['predictions']

    # Calculate the number of elements to select to balance the counts
    num_to_select = min(count_1s, count_0s)

    # Create a new list with balanced counts of 1s and 0s
    leftover_training_data = []
    new_training_data = []
    x0 = 0
    x1 = 0
    for text, prediction in predictions[:]:
        if( not (x0 == num_to_select and x1 == num_to_select)):
            if prediction == 1 and x1 != num_to_select:
                new_training_data.append(predictions.pop(0))
                x1 += 1
            elif prediction == 0 and x0 != num_to_select:
                new_training_data.append(predictions.pop(0))
                x0 += 1
            else:
                leftover_training_data.append(predictions.pop(0))
        else:
            leftover_training_data.append(predictions.pop(0))

    json_data['predictions'] = leftover_training_data

    return json_data, new_training_data

def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens

def remove_english_stopwords(tokens):
    stop_words = set(stopwords.words("english"))
    return [token for token in tokens if token.lower() not in stop_words]

def remove_tagalog_stopwords(tokens):
    return [token for token in tokens if token.lower() not in tagalog_stopwords]

def preprocess(new_training_df):
    # DATA CLEANING

    # REMOVE LINKS
    pattern = r'http\S+|www\S+'
    replacement = ' '
    new_training_df['old'] = new_training_df['old'].str.replace(pattern, replacement, regex=True)
    # REMOVE EMOJIS
    pattern = r'&#\w*?;'
    replacement = ' '
    new_training_df['old'] = new_training_df['old'].str.replace(pattern, replacement, regex=True)
    pattern = r'&\w*?;'
    replacement = ' '
    new_training_df['old'] = new_training_df['old'].str.replace(pattern, replacement, regex=True)
    # REMOVE PUNCTUATIONS AND SYMBOLS
    pattern = r'[^\w\s@#]'
    replacement = ' '
    new_training_df['old'] = new_training_df['old'].str.replace(pattern, replacement, regex=True)
    # REMOVE HASHTAGS
    pattern = r'#(\w+)'
    replacement = ' '
    new_training_df['old'] = new_training_df['old'].str.replace(pattern, replacement, regex=True)
    # REMOVE MENTIONS
    pattern = r'@(\w+)'
    replacement = '@'
    new_training_df['old'] = new_training_df['old'].str.replace(pattern, replacement, regex=True)
    pattern = r'USERNAME'
    replacement = '@'
    new_training_df['old'] = new_training_df['old'].str.replace(pattern, replacement, regex=True)
    # REMOVE WHITESPACES
    pattern = r'\s+'
    replacement = ' '
    new_training_df['old'] = new_training_df['old'].str.replace(pattern, replacement, regex=True)

    # DATA NORMALIZATION

    # ENCODING
    new_training_df['label'] = new_training_df['label'].replace({'Non-Hate Speech': 0, 'Hate Speech': 1})

    # LOWERCASING
    new_training_df['old'] = new_training_df['old'].str.lower()

    # TOKENIZATION
    new_training_df['tokens'] = new_training_df['old'].apply(tokenize_text)

    # STOP WORDS REMOVAL
    new_training_df['tokens'] = new_training_df['tokens'].apply(remove_english_stopwords)
    new_training_df['tokens'] = new_training_df['tokens'].apply(remove_tagalog_stopwords)

    # TOKEN COMPILATION
    new_training_df['content'] = new_training_df['tokens'].apply(lambda tokens: ' '.join(tokens))

    # MENTIONS
    pattern = r'@'
    replacement = '@USER'
    new_training_df['content'] = new_training_df['content'].str.replace(pattern, replacement, regex=True)

    return new_training_df

@app.task
def training_task():
    # Enter code here
    print('\nTRAINING TASK:\n')

    with open(json_data_path, 'r') as file:
        json_data = json.load(file)
        print(len(json_data['predictions']))

    train_df = pd.read_csv('train(latest).csv')
    dataset_df = pd.read_csv('Dataset(latest).csv')

    json_data, new_training_data = get_new_training_data(json_data)

    print('\nNEW_JSON_DATA')
    print(json_data['predictions'])
    print('\nNEW_TRAINING_DATA')
    print(new_training_data)

    print()

    print(train_df.info())
    print(dataset_df.info())

    add_train = {
        'old': [item[0] for item in new_training_data],
        'label': [item[1] for item in new_training_data]
    }
    add_dataset = {
        'content': [item[0] for item in new_training_data],
        'label': [item[1] for item in new_training_data]
    }
    add_train_df = pd.DataFrame(add_train)
    add_dataset_df = pd.DataFrame(add_dataset)
    new_train_df = pd.concat([train_df, add_train_df], ignore_index=True)
    new_dataset_df = pd.concat([dataset_df, add_dataset_df], ignore_index=True)

    print(new_train_df.info())
    print(new_dataset_df.info())
    print()

    new_training_df = new_train_df.copy()
    print(new_training_df.info())
    new_training_df = preprocess(new_training_df)
    print(new_training_df.info())

    # TF-IDF Re-Training
    corpus = new_training_df['content']
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    # Generate Input Features using Re-Trained TF-IDF
    X_train_tfidf = tfidf_vectorizer.transform(new_training_df['content'])
    y_train = new_training_df['label']
    print()
    print(np.unique(y_train))

    # Logistic Regression Incremental Training
    # Create a Logistic Regression classifier
    log_reg_model = LogisticRegression()

    # Train the model on the training data
    log_reg_model.fit(X_train_tfidf, y_train)

    # AFTER EVERYTHING =>
    # SAVE UPDATED JSON_DATA[PREDICTIONS] (removed)
    # SAVE UPDATED TRAIN, DATASET CSV (added)
    # SAVE UPDATED TF-IDF AND LOG-REG MODEL (retrained)
    # Save the updated data to the file

    # with open(json_data_path, 'w') as file:
    #     json.dump(json_data, file, indent=2)
    # new_train_df.to_csv('new_train_df.csv', index=False)
    # new_dataset_df.to_csv('new_dataset_df.csv', index=False)
    # joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer(latest).pkl')
    # joblib.dump(log_reg_model, 'logistic_regression_model(latest).pkl')

# Schedule the task to run daily at midnight
app.conf.beat_schedule = {
    'training': {
        'task': 'tasks.training_task',
        'schedule': crontab(minute=0, hour=0),
    },
}