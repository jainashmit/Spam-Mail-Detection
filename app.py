from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the saved model and vectorizer
with open('spam_detection_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    feature_extraction = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON formatted'}), 400

    data = request.get_json()
    message = data.get('message')
    if not message:
        return jsonify({'error': 'Message field is required'}), 400

    try:
        input_data_features = feature_extraction.transform([message])
        prediction = model.predict(input_data_features)
        result = 'Ham mail' if prediction[0] == 1 else 'Spam mail'
        
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
