import re
import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

# Offensive Word List
offensive_words_list = [
  'abnormal', 'abnoy', 'animal', 'arabo', 'aso',
  'aswang', 'baboy', 'backstabber', 'bading', 'badjao',
  'bakla', 'balimbing', 'baliw', 'baluga', 'balyena',
  'bansot', 'basura', 'batchoy', 'bayot', 'beho',
  'bekimon', 'bingi', 'bingot', 'bingot', 'biot',
  'bisakol', 'bisaya', 'bitch', 'bobita', 'bobo',
  'bruha', 'buang', 'bulag', 'bumbay', 'bungol',
  'burikat', 'butiki', 'buwaya', 'chekwa', 'chingchong',
  'chink', 'corrupt', 'cunt', 'dambuhala', 'demon',
  'demonyo', 'dugyot', 'duling', 'dumbass', 'dwende',
  'elitista', 'engkanto', 'fag', 'faggot', 'fatass',
  'fatty', 'gaga', 'gago', 'gasul', 'gunggong',
  'gurang', 'hampaslupa', 'hayop', 'hipon', 'hudas',
  'idiot', 'impakta', 'indogs', 'intsik', 'intsikbeho',
  'inutil', 'itim', 'judas', 'kabayo', 'kalabaw',
  'kalbo', 'kapre', 'korap', 'kulto', 'kurakot',
  'ladyboy', 'lamang-lupa', 'laspag', 'lesbo', 'linta',
  'mabaho', 'magnanakaw', 'maitim', 'maligno', 'manananggal',
  'mangkukulam', 'mangmang', 'manyak', 'manyakis', 'matanda',
  'saltik', 'sayad', 'mongoloid', 'multo', 'negra',
  'negro', 'nganget', 'ngongo', 'nigga', 'nigger',
  'nognog', 'pandak', 'pandak', 'panget', 'pango',
  'panot', 'peenoise', 'pignoys', 'plastik', 'pokpok',
  'prick', 'prostitute', 'pulpol', 'pussy', 'puta',
  'retard', 'retokada', 'sadako', 'sakang', 'sakim',
  'salot', 'satan', 'satanas', 'satanist', 'shunga',
  'sinto-sinto', 'sinungaling', 'siraulo', 'skwater',
  'slapsoil', 'slut', 'squammy', 'squatter', 'stupid',
  'supot', 'taba', 'tababoy', 'tabaching-ching', 'tabachoy',
  'tanda', 'tanga', 'tarantado', 'terrorista', 'tibo',
  'tikbalang', 'tingting', 'tiyanak', 'tomboy', 'topak',
  'tranny', 'trans', 'trapo', 'trash',
  'tuta', 'ugly', 'ulaga', 'unano', 'unggoy'
]

# Hate Speech Word List
hate_words_list = [
  'abnormal', 'abnoy', 'animal', 'arabo', 'aso',
  'aswang', 'baboy', 'backstabber', 'bading', 'badjao',
  'bakla', 'balimbing', 'baliw', 'baluga', 'balyena',
  'bansot', 'basura', 'batchoy', 'bayot', 'beho',
  'bekimon', 'bingi', 'bingot', 'bingot', 'biot',
  'bisakol', 'bisaya', 'bitch', 'bobita', 'bobo',
  'bruha', 'buang', 'bulag', 'bumbay', 'bungol',
  'burikat', 'butiki', 'buwaya', 'chekwa', 'chingchong',
  'chink', 'corrupt', 'cunt', 'dambuhala', 'demon',
  'demonyo', 'dugyot', 'duling', 'dumbass', 'dwende',
  'elitista', 'engkanto', 'fag', 'faggot', 'fatass',
  'fatty', 'gaga', 'gago', 'gasul', 'gunggong',
  'gurang', 'hampaslupa', 'hayop', 'hipon', 'hudas',
  'idiot', 'impakta', 'indogs', 'intsik', 'intsikbeho',
  'inutil', 'itim', 'judas', 'kabayo', 'kalabaw',
  'kalbo', 'kapre', 'korap', 'kulto', 'kurakot',
  'ladyboy', 'lamang-lupa', 'laspag', 'lesbo', 'linta',
  'mabaho', 'magnanakaw', 'maitim', 'maligno', 'manananggal',
  'mangkukulam', 'mangmang', 'manyak', 'manyakis', 'matanda',
  'saltik', 'sayad', 'mongoloid', 'multo', 'negra',
  'negro', 'nganget', 'ngongo', 'nigga', 'nigger',
  'nognog', 'pandak', 'pandak', 'panget', 'pango',
  'panot', 'peenoise', 'pignoys', 'plastik', 'pokpok',
  'prick', 'prostitute', 'pulpol', 'pussy', 'puta',
  'retard', 'retokada', 'sadako', 'sakang', 'sakim',
  'salot', 'satan', 'satanas', 'satanist', 'shunga',
  'sinto-sinto', 'sinungaling', 'siraulo', 'skwater',
  'slapsoil', 'slut', 'squammy', 'squatter', 'stupid',
  'supot', 'taba', 'tababoy', 'tabaching-ching', 'tabachoy',
  'tanda', 'tanga', 'tarantado', 'terrorista', 'tibo',
  'tikbalang', 'tingting', 'tiyanak', 'tomboy', 'topak',
  'tranny', 'trans', 'trapo', 'trash',
  'tuta', 'ugly', 'ulaga', 'unano', 'unggoy'
]

# Negation Word List
negation_words_list = ["hindi", 'not']

# Stop Word List
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

# Target Word List
target_words = [
    'ni', 'mo', 'ikaw', 'ka', 'kayo', 'kamo', 'kang', 'kayong'
    'siya', 'sya', 'sila', 'sina', 'siyang', 'syang', 'silang',
    'niya', 'nya', 'niyo', 'nyo', 'nila', 'nina', 'niyang', "nyang", 'niyong', 'nyong', 'nilang',
    'yon', 'iyon', 'iyan', 'yan', 'iyang', 'iyong', 'yang', 'yong', 'yun', 'yung'
    '@USER',
]

# Load the TF-IDF model
tfidf_model = joblib.load('tfidf_vectorizer.pkl')

# Load the Logistic Regression Model
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

def ruleBased0(text, hate_words):
    text_quotations = re.findall(r'["\']([^"\']*)["\']', text)
    print(text_quotations)

    matching_indices = []

    for i, text_value in enumerate(text_quotations):
        for hate_word in hate_words:
            if hate_word in text_value:
                matching_indices.append(i)

    if matching_indices:
        return {
            'indices': matching_indices,
            'result': True
        }
    else:
        return {
            'result': False
        }

def ruleBased1(textArray, hate_words, negation_words):
    result = False
    pairs = []

    for i in range(len(textArray) - 1):
        first_word = textArray[i]
        second_word = textArray[i + 1]

        if first_word in negation_words and second_word in hate_words:
            pairs.append([first_word, second_word])
            result = True

    return {'pairs': pairs, 'result': result}

def ruleBased2(textArray, offensive_words, target_words):
    result = False
    pairs = []

    for i in range(len(textArray) - 1):
        first_word = textArray[i]
        second_word = textArray[i + 1]

        if first_word in offensive_words and second_word in target_words:
            pairs.append([first_word, second_word])
            result = True

    return {'pairs': pairs, 'result': result}

def ruleBased3(text, hate_words):
    for hate_word in hate_words:
        if hate_word in text:
            return True

    return False


# if logistic : use all stop words, like will be used in the another training of the model
# if rule : no use english stop words, only use limited tagalog stopwords

# LOGISTIC REGRESSION CLASSIFIER
@app.route('/api/logistic', methods=['GET', 'POST'])
def logistic():
    data = request.json
    text = data.get('text')

    text = preprocessText(text)
    text = preprocessText1(text)

    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    text = ' '.join(filtered_words)

    # FEATURE EXTRACTION: Create input features via trained TF-IDF
    input_features = tfidf_model.transform([text])

    # CLASSIFICATION: Logistic Regression Model
    prediction = log_reg_model.predict(input_features)

    # Get the feature names from the TF-IDF model
    feature_names = tfidf_model.get_feature_names_out()

    # Get the coefficients from the logistic regression model
    coefficients = log_reg_model.coef_[0]

    # Map feature names to their corresponding coefficients
    feature_coefficients = dict(zip(feature_names, coefficients))

    # Identify words in the input text and their absolute coefficients
    contributing_words = {word: abs(feature_coefficients.get(word, 0)) for word in text.split()}

    # Identify probability scores of the prediction for 0 and 1
    class_probabilities = log_reg_model.predict_proba(input_features)

    probability_0 = class_probabilities[0][0]
    probability_1 = class_probabilities[0][1]

    result = {
        'prediction': int(prediction[0]),
        'probability_0': probability_0,
        'probability_1': probability_1,
        'contributing_words': contributing_words
    }
    return jsonify(result)

# HYBRID CLASSIFICATION
@app.route('/api/hybrid', methods=['GET', 'POST'])
def hybrid():
    data = request.json
    text = data.get('text')
    print(text)

    text = preprocessText(text)
    text1 = preprocessText1(text)
    print(text)

    textArray = text1.split()
    print(textArray)

    # Merged Hate and Offensive Words
    hate_x_offensive = []
    for item in offensive_words_list + hate_words_list:
        if item not in hate_x_offensive:
            hate_x_offensive.append(item)

    # Check for "[offensive/derogatory]"
    # Check for [negation] + [offensive/hate]
    # Check for [offensive] + [pronoun]
    # Check for [hate]
    isRule0 = ruleBased0(text, hate_x_offensive)
    isRule1 = ruleBased1(textArray, hate_words_list, negation_words_list)
    isRule2 = ruleBased2(textArray, offensive_words_list, target_words)
    isRule3 = ruleBased3(text, hate_words_list)

    if isRule0['result']:
        # HALF COMPLETE

        result = {
            'model': 'rule',
            'prediction': 0,
            'quotations': isRule0['indices'], #index
            'rule': 0
        }
    elif isRule1['result']:
        # HALF COMPLETE

        result = {
            'model': 'rule',
            'prediction': 0,
            'negation_words_pair': isRule1['pairs'],
            'rule': 1
        }
    elif isRule2['result']:
        # HALF COMPLETE
        result = {
            'model': 'rule',
            'prediction': 1,
            'hate_words_pairs': isRule2['pairs'],
            'rule': 2
        }
    elif isRule3:
        # DONE
        hate_detected_words = [word for word in textArray if word in hate_words_list]

        result = {
            'model': 'rule',
            'prediction': 1,
            'hate_detected_words': hate_detected_words,
            'rule': 3
        }
    else:
        # LOGISTIC REGRESSION MODEL

        filteredText = [word for word in textArray if word.lower() not in stop_words]
        text = ' '.join(filteredText)

        # FEATURE EXTRACTION: Create input features via TF-IDF
        input_features = tfidf_model.transform([text])

        # CLASSIFICATION: Logistic Regression Model
        prediction = log_reg_model.predict(input_features)

        # Get the feature names from the TF-IDF model
        feature_names = tfidf_model.get_feature_names_out()

        # Get the coefficients from the logistic regression model
        coefficients = log_reg_model.coef_[0]

        # Map feature names to their corresponding coefficients
        feature_coefficients = dict(zip(feature_names, coefficients))

        # Identify words in the input text and their absolute coefficients
        contributing_words = {word: abs(feature_coefficients.get(word, 0)) for word in text.split()}

        # Idenitfy probability scores of the prediction for 0 and 1
        class_probabilities = log_reg_model.predict_proba(input_features)

        probability_0 = class_probabilities[0][0]
        probability_1 = class_probabilities[0][1]

        result = {
            'model': 'logistic',
            'prediction': int(prediction[0]),
            'probability_0': probability_0,
            'probability_1': probability_1,
            'contributing_words': contributing_words
        }

    print(result)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
