import re
import joblib
import pandas as pd

# LOAD MODEL
tfidf_model = joblib.load('tfidf_vectorizer.pkl')
log_reg_model = joblib.load('logistic_regression_model.pkl')

def preprocessText(text):
    # REMOVE: Links
    pattern = r'http\S+|www\S+'
    replacement = ' '
    text = re.sub(pattern, replacement, text)
    # REMOVE: Emojis
    pattern = r'&#\w*?;'
    replacement = ' '
    text = re.sub(pattern, replacement, text)
    pattern = r'&\w*?;'
    replacement = ' '
    text = re.sub(pattern, replacement, text)
    # REMOVE: Symbols & Punctuations
    pattern = r'[^\w\s"\'@#]'
    replacement = ' '
    text = re.sub(pattern, replacement, text)
    pattern = r'(?<=[a-zA-Z])\'(?=[a-zA-Z])'
    replacement = ''
    text = re.sub(pattern, replacement, text)
    # REMOVE: Hashtags
    pattern = r'#(\w+)'
    replacement = ' '
    text = re.sub(pattern, replacement, text)
    # REMOVE: Mentions
    pattern = r'@(\w+)'
    replacement = '@USER'
    text = re.sub(pattern, replacement, text)
    # REMOVE: Whitespaces
    pattern = r'\s+'
    replacement = ' '
    text = re.sub(pattern, replacement, text)

    # NORMALIZE: Lowercasing
    text = text.lower()

    # UPDATE: Mentions Casing
    pattern = r'@user'
    replacement = '@USER'
    text = re.sub(pattern, replacement, text)

    return text

def preprocessText1(text):
    # REMOVE: ALL Symbols & Punctuations
    pattern = r'[^\w\s@]'
    replacement = ''
    text = re.sub(pattern, replacement, text)
    # REMOVE: Whitespaces
    pattern = r'\s+'
    replacement = ' '
    text = re.sub(pattern, replacement, text)

    return text

stop_words = [
    "isang", "tungkol", "pagkatapos", "muli", "laban",
    "lahat", "ako", "ay", "at", "alinman", "hindi", "gaya",
    "maging", "dahil", "naging", "bago", "pagiging", "ibaba",
    "pagitan", "pareho", "ngunit", "pamamagitan", "maaari",
    "ginawa", "gawin", "ginagawa", "pababa", "habang",
    "bawat", "ilang", "para", "mula", "pa", "mayroon", "wala",
    "may", "siya", "siya'y", "kanya", "dito", "narito", "sarili",
    "kanyang", "paano", "ako'y", "nito", "tayo'y", "mas",
    "karamihan", "dapat", "aking", "mismo", "ni", "palayo",
    "beses", "lamang", "o", "iba", "atin", "ating", "mga", "labas",
    "kaya", "kaysa", "iyon", "ang",
    "kanilang", "kanila", "sila", "ito", "sa", "rin",
    "hanggang", "pataas", "napakas", "tayo", "ay", "kailan",
    "saan", "alin", "sino", "kanino", "bakit", "kasama",
    "gusto", "ikaw", "iyo", "inyong", "ang", "na", "sa",
    "kay", "ni", "ng", "ngayon", "ito", "ka", "sila", "ka",
    "ngayo'y", "kapag", "kung", "saka", "siya", "siyang",
    "sya", "sya'y", "tayo", "tulad", "yun", "yung"
]

def logistic_classifier(text):

    # FEATURE EXTRACTION: Create input features via trained TF-IDF
    input_features = tfidf_model.transform([text])

    # CLASSIFICATION: Logistic Regression Model
    prediction = log_reg_model.predict(input_features)

    # Identify probability scores of the prediction for 0 and 1
    class_probabilities = log_reg_model.predict_proba(input_features)

    probability_0 = class_probabilities[0][0]
    probability_1 = class_probabilities[0][1]

    result = {
        'prediction': int(prediction[0]),
        'probability_0': probability_0,
        'probability_1': probability_1,
    }

    return result

def logistic_model(text):
    # PREPROCESSING
    print('[RAW TEXT] > ', text)
    text = preprocessText(text)
    text = preprocessText1(text)
    print('[CLEANED TEXT] > ', text)

    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    text = ' '.join(filtered_words)
    print('[TOKENS] > ', filtered_words)

    #CLASSIFICATION
    result = logistic_classifier(text)

    print(f"[PREDICTION] >>> {'1 - Hate Speech' if result['prediction'] == 1 else '0 - Non-Hate Speech'}")
    return result

def calculate_evaluation(actual, predicted):
    if actual == 1 and predicted == 1:
        return 'TP'
    elif actual == 0 and predicted == 1:
        return 'FP'
    elif actual == 0 and predicted == 0:
        return 'TN'
    elif actual == 1 and predicted == 0:
        return 'FN'
    else:
        return 'Unknown'

# MAIN FUNCTION
while True:
    print('\n===========================================')
    print('HATE SPEECH DETECTION TOOL - LOGISTIC MODEL')
    print('===========================================\n')
    select_mode = input('[1] Single or [2] Multiple  >')

    if select_mode == '1':
        text_input = input('\n[MODE - SINGLE]  Enter a text >')

        print()
        result = logistic_model(text_input)

        print('\n[TEXT] > ', text_input)
        print('[PREDICTION] >', '1 - Hate Speech' if result['prediction'] == 1 else '0 - Non-Hate Speech')

        print('\n[HATE %] > ', result['probability_1'])
        print('[NON-HATE %] > ', result['probability_0'])
        print()

    elif select_mode == '2':
        # Load CSV into a DataFrame
        dataset = pd.read_csv('dataset/test2.csv')
        # dataset = pd.read_csv('dataset/test(result).csv')

        # Rename
        column_mapping = {'old': 'text', 'label': 'actual'}
        dataset.rename(columns=column_mapping, inplace=True)
        print()
        dataset.info()
        print()

        input()

        # Prediction
        dataset['predicted_log'] = dataset['text'].apply(lambda x: logistic_model(x)['prediction'])

        # Evaluation
        dataset['evaluation_log'] = dataset.apply(lambda row: calculate_evaluation(row['actual'], row['predicted_log']), axis=1)

        print()
        dataset.info()
        print()
        print(dataset)

        # COUNT
        print('\n[# ACTUAL]')
        counts = dataset['actual'].value_counts()
        print(counts)
        print('\n[# PREDICTED]')
        counts = dataset['predicted_log'].value_counts()
        print(counts)
        print('\n[# EVALUATION]')
        counts = dataset['evaluation_log'].value_counts()
        print(counts)

        # ACCURACY
        TP = counts.get('TP', 0)
        TN = counts.get('TN', 0)
        FP = counts.get('FP', 0)
        FN = counts.get('FN', 0)
        accuracy = (TP + TN) / counts.sum()
        print(f'\n[ACCURACY] => {accuracy:.2%}')

        # EXPORT
        print('\n[EXPORT TO CSV]')
        is_export = input('[1] Save or [Any key to Close]  >')
        if is_export == '1':
            dataset.to_csv('dataset/test(result).csv', index=False)

    else:
        print('Invalid choice. Please enter 1 or 2.')

