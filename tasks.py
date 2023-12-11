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

def generate_feature_names_coef(log_reg_model, tfidf_model):
    hate_threshold = 3.0

    # Assuming log_reg_model is already loaded
    # Get the coefficients and feature names
    coefficients = log_reg_model.coef_[0]
    feature_names = tfidf_model.get_feature_names_out()

    # Create a DataFrame with feature names and their corresponding coefficients
    coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

    # Filter rows where the coefficient is greater than the threshold
    top_hate_words = coefficients_df[coefficients_df['Coefficient'] > hate_threshold]

    # Sort the DataFrame by absolute coefficient values to identify the top hate words
    top_hate_words = top_hate_words.sort_values(by='Coefficient', ascending=False)

    return top_hate_words

def cross_matching_for_new_words(new_top_hate, json_data):
    hate_threshold = 4.0
    offensive_lower = 3.0
    offensive_upper = 4.0

    old_hate_keywords = set(json_data['hate_words_list'])
    old_offensive_keywords = set(json_data['offensive_words_list'])

    new_hate = new_top_hate[new_top_hate['Coefficient'] > hate_threshold]
    new_offensive = new_top_hate[(new_top_hate['Coefficient'] >= offensive_lower) & (new_top_hate['Coefficient'] <= offensive_upper)]

    new_hate_words = set(new_hate['Feature'])
    new_offensive_words = set(new_offensive['Feature'])

    new_hate_words = new_hate_words - old_hate_keywords
    new_offensive_words = new_offensive_words - old_offensive_keywords - old_hate_keywords

    print()
    print(new_hate_words)
    print(new_offensive_words)

    return new_hate_words, new_offensive_words

def updateRules(new_hate_words, new_offensive_words, json_data):
    hate_words_set = set(json_data['hate_words_list'])
    offensive_words_set = set(json_data['offensive_words_list'])

    # Update sets with new words
    hate_words_set.update(new_hate_words)
    offensive_words_set.update(new_offensive_words)

    # Convert sets back to lists
    json_data['hate_words_list'] = list(hate_words_set)
    json_data['offensive_words_list'] = list(offensive_words_set)

    return json_data

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

# MAIN SCHEDULED TASK FOR AUTO TRAIN
@app.task
def training_task():
    print('\nTRAINING TASK:\n')

    # ACCESS NEW STORED DATA FROM JSON DATA AND MERGE WITH ORIG TRAIN DATASET
    with open(json_data_path, 'r') as file:
        json_data = json.load(file)
        print(len(json_data['predictions']))

    train_df = pd.read_csv('train(latest).csv')
    dataset_df = pd.read_csv('Dataset(latest).csv')

    json_data, new_training_data = get_new_training_data(json_data)

    # print('\nUPDATED_JSON_DATA')
    # print(json_data['predictions'])
    # print('\nNEW_BALANCED_TRAINING_DATA')
    # print(new_training_data)

    # print()

    # print(train_df.info())
    # print(dataset_df.info())

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

    # print(new_train_df.info())
    # print(new_dataset_df.info())
    # print()

    # PREPROCESSING UPDATED TRAIN DATA FOR RE-TRAINING
    new_training_df = new_train_df.copy()
    new_training_df = preprocess(new_training_df)
    # print(new_training_df.info())

    # TF-IDF Re-Training
    corpus = new_training_df['content']
    new_tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = new_tfidf_vectorizer.fit_transform(corpus)

    # Generate Input Features using Re-Trained TF-IDF
    X_train_tfidf = new_tfidf_vectorizer.transform(new_training_df['content'])
    y_train = new_training_df['label']

    print()
    print(np.unique(y_train))

    # Logistic Regression Incremental Training
    # Create a Logistic Regression classifier
    new_log_reg_model = LogisticRegression()

    # Train the model on the training data
    new_log_reg_model.fit(X_train_tfidf, y_train)

    # GENERATE FEATURE NAMES W/ COEFFICIENTS
    new_top_hate = generate_feature_names_coef(new_log_reg_model, new_tfidf_vectorizer)

    print('\nNEW FEATURES')
    print(new_top_hate.head(10))

    new_hate_words, new_offensive_words = cross_matching_for_new_words(new_top_hate, json_data)

    print('\nOLD JSON')
    print(len(json_data['hate_words_list']))
    print(len(json_data['offensive_words_list']))

    print('\nNEW WORDS')
    print(len(new_hate_words))
    print(len(new_offensive_words))

    json_data = updateRules(new_hate_words, new_offensive_words, json_data)

    print('\nNEW JSON')
    print(len(json_data['hate_words_list']))
    print(len(json_data['offensive_words_list']))

    # AFTER EVERYTHING =>
    # SAVE UPDATED JSON_DATA[PREDICTIONS] (removed)
    # SAVE UPDATED TRAIN, DATASET CSV (added)
    # SAVE UPDATED TF-IDF AND LOG-REG MODEL (retrained)

    with open(json_data_path, 'w') as file:
        json.dump(json_data, file, indent=2)
    new_train_df.to_csv('train(latest).csv', index=False)
    new_dataset_df.to_csv('Dataset(latest).csv', index=False)
    joblib.dump(new_tfidf_vectorizer, 'tfidf_vectorizer(latest).pkl')
    joblib.dump(new_log_reg_model, 'logistic_regression_model(latest).pkl')
    
    reload_models()

# Schedule the task to run daily at midnight
app.conf.beat_schedule = {
    'training': {
        'task': 'tasks.training_task',
        # 'schedule': crontab(minute=0, hour=0),                          # Daily @ 12:00 AM
        'schedule': crontab(minute=0, hour=0, day_of_week='monday'),    # Weekly @ Monday 12:00 AM
    },
}