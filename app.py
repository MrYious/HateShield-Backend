import re
import joblib
import json
import concurrent.futures
from flask import Flask, jsonify, request
from flask_cors import CORS
from collections import OrderedDict
# from tasks import training_task

app = Flask(__name__)

CORS(app)

# dataset
# 0 : 20676
# 1 : 22697

json_data_path = 'data.json'
json_data = {}
# json_data = {
#     'hate_words_list': [],
#     'offensive_words_list': [],
#     'predictions': [
#         ['text', 1],
#     ]
# }

#  INITIAL LOAD DATA FILE  ->

#  LOAD DATA FILE -> UPDATE DATA -> UPDATE AND SAVE DATA FILE


# Load the updated values
try:
    with open(json_data_path, 'r') as file:
        json_data = json.load(file)
        print(len(json_data['predictions']))
except FileNotFoundError:
    print('Load File Error')

# json_data['predictions'] = []

# # Save the updated data to the file
# with open(json_data_path, 'w') as file:
#     json.dump(json_data, file, indent=2)

# Offensive Word List 151
offensive_words_list = json_data['offensive_words_list']

# Hate Speech Word List 72
hate_words_list = json_data['hate_words_list']

# Negation Word List 5
negation_words_list = [
    "not", 'hindi',
    "neither", "nor",
    "walang"
]

# Stop Word List 111
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

# Target Word List 47
target_words = [
    'you', 'all', 'them', 'her', 'him',
    'ni', 'mo', 'ikaw',  'kayo', 'kamo', 'kang', 'kayong',
    'siya', 'sya', 'sila', 'sina', 'siyang', 'syang', 'silang',
    'niya', 'nya', 'niyo', 'nyo', 'nila', 'nina', 'niyang', "nyang", 'niyong', 'nyong', 'nilang',
    'yon', 'iyon', 'iyan', 'yan', 'iyang', 'iyong', 'yang', 'yong', 'yun', 'yung', 'itong', 'etong',
    '@USER', 'ka', 'kau', 'si', 'ni'
]

# Merged Hate and Offensive Words
hate_x_offensive = []
for item in offensive_words_list + hate_words_list:
    if item not in hate_x_offensive:
        hate_x_offensive.append(item)

# Load the TF-IDF model
tfidf_model = joblib.load('tfidf_vectorizer(latest).pkl')

# Load the Logistic Regression Model
log_reg_model = joblib.load('logistic_regression_model(latest).pkl')

def reload_models():
    global tfidf_model
    global log_reg_model
    global offensive_words_list
    global hate_words_list

    # Load TF-IDF Vectorizer
    tfidf_model = joblib.load('tfidf_vectorizer(latest).pkl')

    # Load Logistic Regression model
    log_reg_model = joblib.load('logistic_regression_model(latest).pkl')

    # Load Updated JSON DATA
    with open(json_data_path, 'r') as file:
        json_data = json.load(file)
        print(len(json_data['predictions']))

    offensive_words_list = json_data['offensive_words_list']
    hate_words_list = json_data['hate_words_list']

# SAVE JSON DATA
def save_new_prediction(text,prediction):
    # Load existing data from the file
    try:
        with open(json_data_path, 'r') as file:
            json_data = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, create a new one with default structure
        json_data = {
            'hate_words_list': [],
            'offensive_words_list': [],
            'predictions': []
        }

    # Add the new prediction to the 'predictions' list
    json_data['predictions'].append([text, prediction])

    # Save the updated data to the file
    with open(json_data_path, 'w') as file:
        json.dump(json_data, file, indent=2)

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

# Check for [hate] = 1
def ruleBased1(textArray, hate_words):
    matched_words = []

    for word in textArray:
        for hate_word in hate_words:
            if hate_word in word.lower():
                matched_words.append(word)
                break  # Stop the loop once a match is found

    result = bool(matched_words)  # True if there are matched words, False otherwise
    return {'word': matched_words, 'result': result}

# Check for [offensive/hate] + [pronoun] = 1
def ruleBased2(textArray, hate_words, target_words):
    result = False
    pairs = []
    newTextArray = textArray.copy()  # Create a copy of the original array

    # First Loop for consecutive pairs
    i = 0
    while i < len(newTextArray) - 1:
        current_word = newTextArray[i]

        # Check if there are enough elements for the next word
        if i + 1 < len(newTextArray):
            next_word = newTextArray[i + 1]

            for hate_word in hate_words:
                for target_word in target_words:
                    # Check for "hate + target" pair with substring matching for hate_word
                    if hate_word.lower() in current_word.lower() and target_word.lower() == next_word.lower():
                        pairs.append([hate_word, target_word])
                        result = True
                        # Remove the pair from newTextArray
                        newTextArray.pop(i)
                        newTextArray.pop(i)  # Pop again to remove the next word
                        i -= 1  # Move the index back to re-check the current position
                        break

                    # Check for "target + hate" pair with exact matching for target_word
                    elif target_word.lower() == current_word.lower() and hate_word.lower() in next_word.lower():
                        pairs.append([target_word, hate_word])
                        result = True
                        # Remove the pair from newTextArray
                        newTextArray.pop(i)
                        newTextArray.pop(i)  # Pop again to remove the next word
                        i -= 1  # Move the index back to re-check the current position
                        break

        i += 1

    # Second loop for pairs with one word in between
    i = 0
    while i < len(newTextArray) - 2:
        current_word = newTextArray[i]
        next_word = newTextArray[i + 1]
        third_word = newTextArray[i + 2]

        for hate_word in hate_words:
            for target_word in target_words:
                # Check for "hate + any word + target" pair
                if hate_word.lower() in current_word.lower() and target_word.lower() == third_word.lower() and next_word.lower() not in hate_words + target_words:
                    pairs.append([hate_word, target_word])
                    result = True
                    # Remove the pair from newTextArray
                    newTextArray.pop(i)
                    newTextArray.pop(i)  # Pop again to remove the next word
                    newTextArray.pop(i)  # Pop again to remove the next word
                    i -= 1  # Move the index back to re-check the current position
                    break

                # Check for "target + any word + hate" pair
                elif target_word.lower() == current_word.lower() and hate_word.lower() in third_word.lower() and next_word.lower() not in hate_words + target_words:
                    pairs.append([target_word, hate_word])
                    result = True
                    # Remove the pair from newTextArray
                    newTextArray.pop(i)
                    newTextArray.pop(i)  # Pop again to remove the next word
                    newTextArray.pop(i)  # Pop again to remove the next word
                    i -= 1  # Move the index back to re-check the current position
                    break
        i += 1

    return {'pairs': pairs, 'result': result}, newTextArray

# Check for [negation] + [offensive/hate] = 0
def ruleBased3(textArray, hate_words, negation_words):
    result = False
    pairs = []
    newTextArray = textArray.copy()  # Create a copy of the original array

    # First Loop for consecutive pairs
    i = 0
    while i < len(newTextArray) - 1:
        current_word = newTextArray[i]

        # Check if there are enough elements for the next word
        if i + 1 < len(newTextArray):
            next_word = newTextArray[i + 1]

            for hate_word in hate_words:
                for negation_word in negation_words:
                    # Check for "hate + negation" pair
                    # if hate_word.lower() in current_word.lower() and negation_word.lower() in next_word.lower():
                    #     pairs.append([hate_word, negation_word])
                    #     result = True
                    #     # Remove the pair from newTextArray
                    #     newTextArray.pop(i)
                    #     newTextArray.pop(i)  # Pop again to remove the next word
                    #     i -= 1  # Move the index back to re-check the current position
                    #     break

                    # Check for "negation + hate" pair
                    if negation_word.lower() in current_word.lower() and hate_word.lower() in next_word.lower():
                        pairs.append([negation_word, hate_word])
                        result = True
                        # Remove the pair from newTextArray
                        newTextArray.pop(i)
                        newTextArray.pop(i)  # Pop again to remove the next word
                        i -= 1  # Move the index back to re-check the current position
                        break

        i += 1

    # Second loop for pairs with one word in between
    i = 0
    while i < len(newTextArray) - 2:
        current_word = newTextArray[i]
        next_word = newTextArray[i + 1]
        third_word = newTextArray[i + 2]

        for hate_word in hate_words:
            for negation_word in negation_words:
                # Check for "hate + any word + negation" pair
                # if hate_word.lower() in current_word.lower() and negation_word.lower() in third_word.lower() and not next_word.lower() in hate_words + negation_words:
                #     pairs.append([hate_word, negation_word])
                #     result = True
                #     # Remove the pair from newTextArray
                #     newTextArray.pop(i)
                #     newTextArray.pop(i)  # Pop again to remove the next word
                #     newTextArray.pop(i)  # Pop again to remove the next word
                #     i -= 1  # Move the index back to re-check the current position
                #     break

                # Check for "negation + any word + hate" pair
                if negation_word.lower() in current_word.lower() and hate_word.lower() in third_word.lower() and not next_word.lower() in hate_words + negation_words:
                    pairs.append([negation_word, hate_word])
                    result = True
                    # Remove the pair from newTextArray
                    newTextArray.pop(i)
                    newTextArray.pop(i)  # Pop again to remove the next word
                    newTextArray.pop(i)  # Pop again to remove the next word
                    i -= 1  # Move the index back to re-check the current position
                    break

        i += 1

    return {'pairs': pairs, 'result': result}, newTextArray

# Check for "[offensive/derogatory]" = 0
def ruleBased4(text, hate_words):
    text_quotations = re.findall(r'["\']([^"\']*)["\']', text)
    matching_indices = []

    for i, text_value in enumerate(text_quotations):
        for hate_word in hate_words:
            if hate_word in text_value:
                matching_indices.append(i)

    new_text = text

    for index in sorted(matching_indices, reverse=True):
        start = text.find(text_quotations[index])
        end = start + len(text_quotations[index])
        new_text = new_text[:start] + new_text[end:]

    return {
        'indices': matching_indices,
        'result': bool(matching_indices)
    }, new_text

# MODELS
def ex_logistic_regression_classifier(text):
    # FEATURE EXTRACTION: Create input features via trained TF-IDF
    input_features = tfidf_model.transform([text])

    # CLASSIFICATION: Logistic Regression Model
    prediction = log_reg_model.predict(input_features)

    # Identify probability scores of the prediction for 0 and 1
    class_probabilities = log_reg_model.predict_proba(input_features)

    probability_0 = class_probabilities[0][0]
    probability_1 = class_probabilities[0][1]

    # Get the feature names from the TF-IDF model
    feature_names = tfidf_model.get_feature_names_out()

    # Get the coefficients from the logistic regression model
    coefficients = log_reg_model.coef_[0]

    # Map feature names to their corresponding coefficients
    feature_coefficients = dict(zip(feature_names, coefficients))

    # Identify words in the input text and their absolute coefficients
    contributing_words = {word: abs(feature_coefficients.get(word, 0)) for word in text.split()}

    result = {
        'prediction': int(prediction[0]),
        'probability_0': probability_0,
        'probability_1': probability_1,
        'contributing_words': contributing_words
    }
    return result

def hybrid_logistic_regression_classifier(text):
    # PREPROCESSING
    textArray = text.split()

    filteredText = [word for word in textArray if word.lower() not in stop_words]
    text = ' '.join(filteredText)

    # FEATURE EXTRACTION: Create input features via trained TF-IDF
    input_features = tfidf_model.transform([text])

    # CLASSIFICATION: Logistic Regression Model
    prediction = log_reg_model.predict(input_features)

    # Identify probability scores of the prediction for 0 and 1
    class_probabilities = log_reg_model.predict_proba(input_features)

    probability_0 = class_probabilities[0][0]
    probability_1 = class_probabilities[0][1]

    # Get feature names from the TF-IDF vectorizer
    feature_names = tfidf_model.get_feature_names_out()

    # Get the coefficients from the logistic regression model
    coefficients = log_reg_model.coef_[0]

    # Map feature names to their corresponding coefficients
    feature_coefficients = dict(zip(feature_names, coefficients))

    # Identify words in the input text and their absolute coefficients for hate speech
    contributing_words_hate_speech = {word: abs(feature_coefficients.get(word, 0)) for word in text.split()}

    # Filter out words that don't contribute to hate speech
    contributing_words_hate_speech = {word: coefficient for word, coefficient in contributing_words_hate_speech.items() if coefficient > 0}

    # Filter out words with absolute coefficients less than or equal to 3.0
    contributing_hate_words = {word: coefficient for word, coefficient in contributing_words_hate_speech.items() if abs(coefficient) > 3.0}

    # Sort the words by their absolute coefficients in descending order
    sorted_contributing_words = dict(sorted(contributing_words_hate_speech.items(), key=lambda item: item[1], reverse=True))

    result = {
        'prediction': int(prediction[0]),
        'probability_0': probability_0,
        'probability_1': probability_1,
        'contributing_words': sorted_contributing_words,
        'contributing_hate_words': contributing_hate_words
    }

    return result

def hybrid_rule_based_classifier(text):

    isRule4, newText = ruleBased4(text, hate_x_offensive)
    textArray = preprocessText1(newText).split()
    textArray.append('[END]')
    textArray.append('[END]')

    isRule3, textArray = ruleBased3(textArray, hate_x_offensive, negation_words_list)
    isRule2, textArray = ruleBased2(textArray, hate_x_offensive, target_words)
    isRule1 = ruleBased1(textArray, hate_words_list)

    if isRule4['result'] and (not isRule2['result'] ) and (not isRule1['result'] ):
        unique_indices = list(OrderedDict.fromkeys(isRule4['indices']))

        result = {
            'prediction': 0,
            'rule': 4,
            'quotations': unique_indices,
        }
    elif isRule3['result'] and (not isRule2['result'] ) and (not isRule1['result'] ):

        result = {
            'prediction': 0,
            'rule': 3,
            'negation_words_pair': isRule3['pairs'],
        }
    elif isRule2['result']:
        result = {
            'prediction': 1,
            'rule': 2,
            'hate_words_pairs': isRule2['pairs'],
        }
    elif isRule1['result']:
        result = {
            'prediction': 1,
            'rule': 1,
            'hate_detected_words': isRule1['word'],
        }
    else:
        result = {
            'prediction': 0,
            'rule': 5,
        }

    rule_dicts = [isRule1, isRule2, isRule3, isRule4]

    # Create a list containing the rule numbers where 'prediction' is 1
    result['rules'] = [i + 1 for i, rule_dict in enumerate(rule_dicts) if rule_dict.get('result') == True]

    return result

# VOTING SYSTEM
def majority_voting(rule_result, logistic_result):
    result = {}

    # SAME PREDICTION
    if  rule_result['prediction'] == logistic_result['prediction']:
        rule_result.update(logistic_result)
        result = rule_result
        result['selected'] = 'both'

        print('HYBRID RESULT = SAME')
        print(result)
        print("\n")

    # DIFFERENT PREDICTION
    else :
        # ALGORITHM TO SELECT WHICH PREDICTION TO CHOOSE
        #  1 0
        #  0 1

        if (rule_result['prediction'] == 1 and logistic_result['prediction'] == 0):
            if (1 in rule_result['rules'] and 2 in rule_result['rules']):
                result['selected'] = 'rule'
            elif (3 in rule_result['rules'] and 4 in rule_result['rules']):
                result['selected'] = 'logreg'
            elif (1 in rule_result['rules'] or 2 in rule_result['rules']):
                if logistic_result['probability_0'] > 0.80:
                    result['selected'] = 'logreg'
                elif logistic_result['probability_0'] > 0.65 and (3 in rule_result['rules'] or 4 in rule_result['rules']):
                    result['selected'] = 'logreg'
                else:
                    result['selected'] = 'rule'
            # 1 0
            # rule 1-2 3-4 | logReg prob1<50% prob0>50%
                # if rule1 & rule2
                    # Select Rule 1
                # if rule3 & rule 4
                    # Select Logistic 0
                # else if rule1 | rule2
                    # if prob0 > 80%
                        # Select Logistic 0
                    # else if rule3 | rule 4 & prob0 > 65%
                        # Select Logistic 0
                    # else
                        # Select Rule 1
        elif (rule_result['prediction'] == 0 and logistic_result['prediction'] == 1):
            if (3 in rule_result['rules'] and 4 in rule_result['rules']):
                result['selected'] = 'rule'
            elif not all(word in hate_x_offensive for word in logistic_result['contributing_hate_words']):
                print('here1')
                result['selected'] = 'logreg'
            elif rule_result['rule'] == 5 and logistic_result['probability_1'] > 0.65:
                print('here2')
                result['selected'] = 'logreg'
            else:
                result['selected'] = 'rule'
            # 0 1
            # rule 3-4 5 | logReg prob1>50% prob0<50% words
                # if rule3 & rule 4
                    # Select Rule 0
                # else if word not in hate/offensive
                    # Select Logistic 1 >
                # else if rule5 & prob1 > 65%
                    # Select Logistic 1 >
                # else
                    # Select Rule 0


        if result['selected'] == 'rule':
            result.update(logistic_result)
            result.update(rule_result)
            result['prediction'] = rule_result['prediction']
        elif result['selected'] == 'logreg':
            result.update(rule_result)
            result.update(logistic_result)
            result['prediction'] = logistic_result['prediction']

        print('HYBRID RESULT = DIFFERENT')
        print(result)
        print("\n")

    return result

# LOGISTIC REGRESSION CLASSIFIER
@app.route('/api/logistic', methods=['GET', 'POST'])
def logistic():
    data = request.json
    text = data.get('text')

    # PREPROCESSING
    text = preprocessText(text)
    text = preprocessText1(text)

    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    text = ' '.join(filtered_words)

    #CLASSIFICATION
    result = ex_logistic_regression_classifier(text)

    return jsonify(result)

# HYBRID CLASSIFICATION
@app.route('/api/hybrid', methods=['GET', 'POST'])
def hybrid():
    data = request.json
    text = data.get('text')

    print("\n")
    print("TEXT")
    print(text)
    print("\n")

    # PREPROCESSING
    text = preprocessText(text)
    text1 = preprocessText1(text)

    # PARALLEL PROCESSING
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Execute model functions in parallel
        rule_model = executor.submit(hybrid_rule_based_classifier, text)
        logistic_model = executor.submit(hybrid_logistic_regression_classifier, text1)

        # Wait for both tasks to complete
        concurrent.futures.wait([rule_model, logistic_model])

        # Get results from completed tasks
        result_model1 = rule_model.result()
        result_model2 = logistic_model.result()

        print('Rule-Based Model')
        print(result_model1)
        print("\n")
        print('Logistic Regression Model')
        print(result_model2)
        print("\n")

    result = majority_voting(result_model1, result_model2)

    save_new_prediction( data.get('text'), result['prediction'] )

    return jsonify(result)

# training_task()

if __name__ == '__main__':
    app.run(debug=True)
