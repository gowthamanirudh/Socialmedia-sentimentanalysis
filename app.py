"""
Flask API for Sentiment Analysis
Simple REST API for sentiment prediction
"""

from flask import Flask, request, jsonify, render_template_string
from model import SentimentModel
import os

app = Flask(__name__)

# Load model at startup
print("Loading sentiment analysis model...")
sentiment_model = SentimentModel(
    model_path='sentiment_model_nb.pkl',
    vectorizer_path='tfidf_vectorizer.pkl'
)
print("Model loaded successfully!")

# HTML template for web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Sentiment Analysis</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            resize: vertical;
            box-sizing: border-box;
        }
        button {
            width: 100%;
            padding: 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 18px;
            cursor: pointer;
            margin-top: 10px;
        }
        button:hover {
            background-color: #45a049;
        }
        .result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 5px;
            display: none;
        }
        .positive {
            background-color: #d4edda;
            border: 2px solid #c3e6cb;
            color: #155724;
        }
        .negative {
            background-color: #f8d7da;
            border: 2px solid #f5c6cb;
            color: #721c24;
        }
        .sentiment {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .confidence {
            font-size: 18px;
        }
        .examples {
            margin-top: 30px;
            padding: 15px;
            background-color: #e7f3ff;
            border-radius: 5px;
        }
        .examples h3 {
            margin-top: 0;
        }
        .example-btn {
            display: inline-block;
            margin: 5px;
            padding: 8px 15px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 14px;
        }
        .example-btn:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 Sentiment Analysis</h1>
        <p style="text-align: center; color: #666;">Analyze the sentiment of social media posts and text</p>
        
        <textarea id="textInput" rows="5" placeholder="Enter your text here..."></textarea>
        <button onclick="analyzeSentiment()">Analyze Sentiment</button>
        
        <div id="result" class="result">
            <div class="sentiment" id="sentiment"></div>
            <div class="confidence" id="confidence"></div>
        </div>
        
        <div class="examples">
            <h3>Try these examples:</h3>
            <button class="example-btn" onclick="setExample('I love this product! It\\'s amazing and works perfectly!')">Positive Example</button>
            <button class="example-btn" onclick="setExample('This is terrible. Worst experience ever. Very disappointed.')">Negative Example</button>
            <button class="example-btn" onclick="setExample('Just finished an amazing workout! Feeling great 💪😊')">With Emojis</button>
        </div>
    </div>

    <script>
        function analyzeSentiment() {
            const text = document.getElementById('textInput').value;
            if (!text.trim()) {
                alert('Please enter some text to analyze');
                return;
            }

            fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            })
            .then(response => response.json())
            .then(data => {
                const resultDiv = document.getElementById('result');
                const sentimentDiv = document.getElementById('sentiment');
                const confidenceDiv = document.getElementById('confidence');
                
                sentimentDiv.textContent = '😊 ' + data.sentiment;
                confidenceDiv.textContent = 'Confidence: ' + (data.confidence * 100).toFixed(1) + '%';
                
                resultDiv.className = 'result ' + data.sentiment.toLowerCase();
                resultDiv.style.display = 'block';
            })
            .catch(error => {
                alert('Error analyzing sentiment: ' + error);
            });
        }

        function setExample(text) {
            document.getElementById('textInput').value = text;
        }

        // Allow Enter key to submit (with Shift+Enter for new line)
        document.getElementById('textInput').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                analyzeSentiment();
            }
        });
    </script>
</body>
</html>
"""


@app.route('/')
def home():
    """Serve the web interface"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint for sentiment prediction
    
    Request JSON:
        {
            "text": "Text to analyze"
        }
    
    Response JSON:
        {
            "text": "Original text",
            "sentiment": "Positive" or "Negative",
            "confidence": 0.95,
            "label": 1 or 0
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        
        if not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Make prediction
        result = sentiment_model.predict(text)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """
    API endpoint for batch sentiment prediction
    
    Request JSON:
        {
            "texts": ["Text 1", "Text 2", ...]
        }
    
    Response JSON:
        {
            "results": [
                {"text": "...", "sentiment": "...", "confidence": ...},
                ...
            ]
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({'error': 'No texts provided'}), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({'error': 'texts must be a list'}), 400
        
        # Make predictions
        results = sentiment_model.predict_batch(texts)
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': sentiment_model.model is not None})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
