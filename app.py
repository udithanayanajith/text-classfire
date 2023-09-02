from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('classifier.pkl', 'rb') as classifier_file:
    classifier = pickle.load(classifier_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    new_incident = data['incident']

    new_incident_vectorized = vectorizer.transform([new_incident])
    prediction = classifier.predict(new_incident_vectorized)

    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
