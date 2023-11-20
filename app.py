import re
import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

# Stop Words List
stopwords = [
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
  'may saltik', 'may sayad', 'mongoloid', 'multo', 'negra',
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
  'tranny', 'trans', 'trapo', 'trash', 'traydor',
  'tuta', 'ugly', 'ulaga', 'unano', 'unggoy'
];

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

    return text

def ruleBased2(text, hate_words):
    # Convert text to lowercase for case-insensitive matching
    text = text.lower()

    # Check if any hate-containing words are present in the text
    for hate_word in hate_words:
        if hate_word in text:
            return True

    # No hate-containing words found in the text
    return False

# LOGISTIC REGRESSION CLASSIFIER
@app.route('/api/logistic', methods=['GET', 'POST'])
def logistic():
    data = request.json
    text = data.get('text')

    print(text)

    text = preprocessText(text)

    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]
    text = ' '.join(filtered_words)

    print(text)

    # FEATURE EXTRACTION: Create input features via TF-IDF
    input_features = tfidf_model.transform([text])

    # CLASSIFICATION: Logistic Regression Model
    prediction = log_reg_model.predict(input_features)

    class_probabilities = log_reg_model.predict_proba(input_features)

    probability_0 = class_probabilities[0][0]
    probability_1 = class_probabilities[0][1]

    result = {
        'prediction': int(prediction[0]),
        'probability_0': probability_0,
        'probability_1': probability_1
    }
    return jsonify(result)

# HYBRID CLASSIFICATION
@app.route('/api/hybrid', methods=['GET', 'POST'])
def hybrid():
    data = request.json
    text = data.get('text')

    text = preprocessText(text)
    textArray = text.split()

    negation_words_list = [""]

    isRule1 = False
    isRule2 = ruleBased2(text, hate_words_list)

    if isRule1:
        matching_words = [word for word in textArray if word in hate_words_list]

        result = {
            'model': 'rule',
            'prediction': 0,
            'negation_detected_words': matching_words,
            'rule': 1
        }
    elif isRule2:
        matching_words = [word for word in textArray if word in hate_words_list]

        result = {
            'model': 'rule',
            'prediction': 1,
            'hate_detected_words': matching_words,
            'rule': 2
        }
    else:
        text = preprocessText(text)

        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stopwords]
        text = ' '.join(filtered_words)

        # FEATURE EXTRACTION: Create input features via TF-IDF
        input_features = tfidf_model.transform([text])

        # CLASSIFICATION: Logistic Regression Model
        prediction = log_reg_model.predict(input_features)

        class_probabilities = log_reg_model.predict_proba(input_features)

        probability_0 = class_probabilities[0][0]
        probability_1 = class_probabilities[0][1]

        result = {
            'model': 'logistic',
            'prediction': int(prediction[0]),
            'probability_0': probability_0,
            'probability_1': probability_1
        }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
