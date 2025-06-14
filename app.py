from flask import Flask, render_template, request
import pickle
import nltk
from nltk.corpus import stopwords

# Download NLTK data (only needed the first time)
nltk.download('stopwords')

# Load stopwords
stp_words = stopwords.words('english')

# Function to clean input text
def clean_text(text):
    return " ".join(word for word in text.split() if word.lower() not in stp_words)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email_text = request.form['email_text']
        cleaned_text = clean_text(email_text)
        prediction = model.predict([cleaned_text])[0]
        result = "Not Spam" if prediction == 0 else "Spam"
        return render_template('index.html', prediction=result, email=email_text)

if __name__ == '__main__':
    app.run(debug=True)
