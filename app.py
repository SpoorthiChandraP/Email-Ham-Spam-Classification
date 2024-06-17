from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Preprocess function
def preprocess_text(text):
    if text is None:
        return ''
    preprocessed_text = text.lower()

    return preprocessed_text


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        # Preprocess the message
        preprocessed_message = preprocess_text(message)
        # Vectorize the preprocessed message
        vectorized_message = vectorizer.transform([preprocessed_message])
        # Get the prediction (numeric label)
        prediction_numeric = model.predict(vectorized_message)[0]
        # Map numeric label to class name
        prediction = "spam" if prediction_numeric == 1 else "ham"
        # Render the result template with the prediction
        return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
