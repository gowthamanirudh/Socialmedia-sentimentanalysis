# Quick Start Guide

Get the sentiment analysis model up and running in 5 minutes! This guide will help you set up and start analyzing sentiment in social media text.

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Step 2: Train the Model

```bash
python train_model.py
```

This will:
- Load the Sentiment140 dataset (1.6M tweets)
- Train on 10,000 tweets (adjustable in the script)
- Save the trained model files
- Show accuracy metrics

Training takes about 2-5 minutes depending on your machine.

## Step 3: Test the Model

```bash
python test_model.py
```

This runs automated tests to verify everything works.

## Step 4: Make Predictions

### Option A: Interactive CLI

```bash
python predict.py
```

Then type your text and press Enter!

### Option B: Command Line

```bash
python predict.py "I love this product!"
```

### Option C: Web Interface

```bash
python app.py
```

Then open your browser to: http://localhost:5000

## What You Get

✅ **Trained Model**: `sentiment_model_nb.pkl` (Naive Bayes classifier)
✅ **Vectorizer**: `tfidf_vectorizer.pkl` (TF-IDF feature extractor)
✅ **~71% Accuracy** on sentiment classification
✅ **Multiple Interfaces**: CLI, Python API, Web API
✅ **Production Ready**: Clean, modular, well-documented code
✅ **Handles Emojis**: Converts emojis to text for better analysis

## Usage Examples

### Python Code

```python
from model import SentimentModel

# Load model
model = SentimentModel('sentiment_model_nb.pkl', 'tfidf_vectorizer.pkl')

# Predict
result = model.predict("This is amazing!")
print(result['sentiment'])  # "Positive"
print(result['confidence'])  # 0.95
```

### REST API

```bash
# Start server
python app.py

# Make request
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this!"}'
```

## Troubleshooting

**Problem**: Model files not found
**Solution**: Run `python train_model.py` first

**Problem**: spaCy model not found
**Solution**: Run `python -m spacy download en_core_web_sm`

**Problem**: NLTK data not found
**Solution**: It downloads automatically, but you can manually run:
```python
import nltk
nltk.download('stopwords')
```

## Next Steps

1. **Improve accuracy**: Increase `sample_size` in `train_model.py` (try 50,000+)
2. **Deploy to production**: Use Gunicorn + Nginx or deploy to cloud (Heroku, AWS, etc.)
3. **Customize preprocessing**: Modify `preprocessor.py` to fit your needs
4. **Integrate**: Use the Python API in your own applications
5. **Analyze your data**: Load your own CSV files and analyze sentiment

## File Overview

- `train_model.py` - Train the model
- `predict.py` - CLI predictions
- `app.py` - Web API server
- `test_model.py` - Automated tests
- `model.py` - Core model class
- `preprocessor.py` - Text preprocessing

That's it! You're ready to analyze sentiment in social media posts! 🎉

## Need Help?

- Check `README.md` for detailed documentation
- See `PRODUCTION_SETUP.md` for deployment options
- Run `python test_model.py` to verify everything works
- Open an issue on GitHub if you encounter problems
