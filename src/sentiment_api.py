import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizer, TFDistilBertModel
import tensorflow as tf
from flask import Flask, request, jsonify, render_template

nltk.download('stopwords', quiet=True)

app = Flask(__name__)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
max_length = 128

class SentimentModel(tf.keras.Model):
    def __init__(self):
        super(SentimentModel, self).__init__()
        self.distilbert = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dense = tf.keras.layers.Dense(3, activation='softmax')

    def call(self, inputs):
        output = self.distilbert(inputs)
        pooled_output = output[0][:, 0, :]
        return self.dense(pooled_output)

model = SentimentModel()
dummy_input = tf.constant([[1] * max_length])
_ = model(dummy_input)
model.load_weights('sentiment_model_weights.h5')

label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['negative', 'neutral', 'positive'])

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.json
    text = data['text']
    cleaned_text = clean_text(text)
    
    encoded_text = tokenizer([cleaned_text], padding=True, truncation=True, max_length=max_length, return_tensors='tf').input_ids
    prediction = model(encoded_text)
    predicted_class = label_encoder.classes_[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    
    return jsonify({
        'sentiment': predicted_class,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True)