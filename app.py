import re
import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

# Offensive Word List 151
offensive_words_list = [
    'abnormal', 'abusado', 'adik', 'animal', 'arabo',
    'arogante', 'aso', 'aswang', 'baboy', 'backstabber',
    'bading', 'badjao', 'baduy', 'bakla', 'balasubas',
    'balimbing', 'baliw', 'baluga', 'balyena', 'bangag',
    'bansot', 'basura', 'batchoy', 'bingi', 'bingot',
    'bisaya', 'bobo', 'bruha', 'bulag', 'bulok',
    'bumbay', 'bungol', 'butiki', 'buwaya', 'corrupt',
    'dambuhala', 'demon', 'demonyo', 'dugyot', 'duling',
    'dwende', 'echosera', 'elitista', 'engkanto', 'engot',
    'epal', 'eut', 'fuck', 'fucking', 'gago', 'gahaman',
    'ganid', 'garapal', 'gasul', 'hambog', 'hayop',
    'higad', 'hipokrita', 'hipokrito', 'hipon', 'hudas',
    'hypocrite', 'idiot', 'ingrata', 'ipokrita', 'ipokrito',
    'itim', 'kabayo', 'kalabaw', 'kalbo', 'kantot',
    'kantutin', 'kingina', 'kulto', 'kupal', 'kurakot',
    'lamang-lupa', 'laos', 'laspag', 'libagin', 'lutang',
    'maasim', 'mabaho', 'madaya', 'magnanakaw', 'maitim',
    'malandi', 'maligno', 'malisyosa', 'manananggal', 'mandurugas',
    'mangkukulam', 'mangmang', 'manyak', 'manyakis', 'mapayat',
    'mataba', 'matanda', 'mayabang', 'muchacha', 'ngongo',
    'palaka', 'palamunin', 'palpak', 'pandak', 'panget',
    'pango', 'panot', 'payat', 'peenoise', 'peste',
    'plastik', 'pokpok', 'poop', 'pulubi', 'punyemas',
    'punyeta', 'putangina', 'saltik', 'sayad', 'shit',
    'siraulo', 'skwater', 'squatter', 'stupid', 'supot',
    'suwapang', 'taba', 'tae', 'tanda', 'tanga',
    'tangina', 'terrorista', 'tibo', 'tikbalang', 'tite',
    'titi', 'tomboy', 'topak', 'trapo', 'trash',
    'traydor', 'tubol', 'tukmol', 'tuta', 'ugly',
    'ulol', 'ulopong', 'unggoy', 'mamatay', 'maghirap'
]

# Hate Speech Word List 70
hate_words_list = [
    'abnoy', 'asshole', 'bayot', 'beho', 'bekimon',
    'biot', 'bisakol', 'bitch', 'bobita', 'buang',
    'burikat', 'chekwa', 'chingchong', 'chink', 'cunt',
    'dumbass', 'fag', 'faggot', 'fatass', 'fatty',
    'gaga', 'gunggong', 'gurang', 'hampaslupa', 'hindot',
    'impakta', 'indogs', 'intsik', 'intsikbeho', 'inutil',
    'judas', 'kapre', 'ladyboy', 'lesbo', 'linta',
    'mongoloid', 'multo', 'negra', 'negro', 'nganget',
    'nigga', 'nigger', 'nognog', 'pignoys', 'prick',
    'prostitute', 'pulpol', 'pussy', 'puta', 'retard',
    'retokada', 'sakang', 'sakim', 'salot', 'shunga',
    'sintosinto', 'slapsoil', 'slut', 'squammy', 'tababoy',
    'tabachingching', 'tabachoy', 'tarantado', 'tingting',
    'tiyanak', 'tranny', 'trans', 'ulaga', 'unano', 'whore'
]

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

# Target Word List 44
target_words = [
    'you', 'all', 'them', 'her', 'him'
    'ni', 'mo', 'ikaw', 'ka', 'kayo', 'kamo', 'kang', 'kayong',
    'siya', 'sya', 'sila', 'sina', 'siyang', 'syang', 'silang',
    'niya', 'nya', 'niyo', 'nyo', 'nila', 'nina', 'niyang', "nyang", 'niyong', 'nyong', 'nilang',
    'yon', 'iyon', 'iyan', 'yan', 'iyang', 'iyong', 'yang', 'yong', 'yun', 'yung', 'itong', 'etong'
    '@USER', 'may', 'mga'
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

        for hate_word in hate_words:
            if first_word.lower() in negation_words and hate_word.lower() in second_word.lower():
                pairs.append([first_word, second_word])
                result = True

    return {'pairs': pairs, 'result': result}

def ruleBased2(textArray, hate_words, target_words):
    result = False
    pairs = []

    for i in range(len(textArray) - 1):
        first_word = textArray[i]
        second_word = textArray[i + 1]

        for hate_word in hate_words:
            if hate_word.lower() in first_word.lower() and second_word in target_words:
                pairs.append([first_word, second_word])
                result = True

    return {'pairs': pairs, 'result': result}

def ruleBased3(textArray, hate_words):
    matched_words = [word for word in textArray for hate_word in hate_words if hate_word in word.lower()]
    result = bool(matched_words)  # True if there are matched words, False otherwise

    return {'word': matched_words, 'result': result}


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
    # Check for [offensive/hate] + [pronoun]
    # Check for [hate]
    isRule0 = ruleBased0(text, hate_x_offensive)
    isRule1 = ruleBased1(textArray, hate_x_offensive, negation_words_list)
    isRule2 = ruleBased2(textArray, hate_x_offensive, target_words)
    isRule3 = ruleBased3(textArray, hate_words_list)

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
    elif isRule3['result']:
        # HALF COMPLETE

        result = {
            'model': 'rule',
            'prediction': 1,
            'hate_detected_words': isRule3['word'],
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
