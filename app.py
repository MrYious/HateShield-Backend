import re
import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

# Load the TF-IDF model
tfidf_model = joblib.load('tfidf_vectorizer.pkl')

# Load the Logistic Regression Model
log_reg_model = joblib.load('logistic_regression_model.pkl')

@app.route('/api/', methods=['GET'])
def get_data():
    # Implement your logic to retrieve data here
    data = {"message": "Hello from Flask!"}
    return jsonify(data)

# LOGISTIC REGRESSION CLASSIFIER
@app.route('/api/logistic', methods=['GET', 'POST'])
def logistic():
    data = request.json
    text = data.get('text')

    print(text)

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
    replacement = '@user'
    text = re.sub(pattern, replacement, text)
    # REMOVE: Whitespaces
    pattern = r'\s+'
    replacement = ' '
    text = re.sub(pattern, replacement, text)

    # NORMALIZE: Lowercasing
    text = text.lower()

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

    print(text)

    # FEATURE EXTRACTION: Create input features via TF-IDF
    input_features = tfidf_model.transform([text])

    # CLASSIFICATION: Logistic Regression Model
    prediction = log_reg_model.predict(input_features)

    class_probabilities = log_reg_model.predict_proba(input_features)

    probability_0 = class_probabilities[0][0]
    probability_1 = class_probabilities[0][1]

    result = {'prediction': int(prediction[0]), 'probability_0': probability_0, 'probability_1': probability_1}
    return jsonify(result)

# HYBRID CLASSIFICATION
@app.route('/api/hybrid', methods=['GET', 'POST'])
def hybrid():
    # Implement your logic to retrieve data here
    data = {"message": "Hello from Flask!"}
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
