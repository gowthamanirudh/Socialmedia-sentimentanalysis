# Production Setup Guide ✅

This sentiment analysis system is **production-ready** and can be deployed immediately! This guide covers everything you need to know about the project structure, usage, and deployment.

## 📦 New Production Files

### Core Modules
- **`preprocessor.py`** - Text preprocessing with emoji handling, URL removal, lemmatization
- **`model.py`** - Main model class for training and prediction
- **`train_model.py`** - Training script for the Naive Bayes model
- **`predict.py`** - CLI tool for making predictions
- **`app.py`** - Flask web API with beautiful UI

### Documentation
- **`README.md`** - Complete documentation
- **`QUICKSTART.md`** - 5-minute setup guide
- **`PRODUCTION_SETUP.md`** - This file

### Configuration
- **`requirements.txt`** - Updated with production dependencies
- **`.gitignore`** - Proper git ignore rules
- **`test_model.py`** - Automated testing script

### Model Files (Generated)
- **`sentiment_model_nb.pkl`** - Trained Naive Bayes model (71% accuracy)
- **`tfidf_vectorizer.pkl`** - TF-IDF vectorizer for feature extraction

### Research Files (Optional)
- **`SocialMedia_Sentiment_Analysis.ipynb`** - Original research notebook
- **`distilbert_logistic_model.pkl`** - Experimental deep learning model

## 🚀 How to Use

### 1. Install Dependencies (First Time Only)

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Option A: Use Pre-trained Model

If you have the pre-trained model files (`sentiment_model_nb.pkl` and `tfidf_vectorizer.pkl`), you can use them directly:

```bash
# Test it
python test_model.py

# Use CLI
python predict.py

# Start web server
python app.py
```

### 3. Option B: Train Your Own Model

To train a fresh model or retrain with different parameters:

```bash
python train_model.py
```

**Note**: The training script is set to use 10,000 tweets by default. Edit `train_model.py` line 26 to change `sample_size`.

## 🎯 Three Ways to Use Your Model

### 1️⃣ Command Line Interface

```bash
# Interactive mode
python predict.py

# Single prediction
python predict.py "I love this product!"
```

### 2️⃣ Python API

```python
from model import SentimentModel

# Load model
model = SentimentModel(
    model_path='sentiment_model_nb.pkl',
    vectorizer_path='tfidf_vectorizer.pkl'
)

# Single prediction
result = model.predict("This is amazing!")
print(result)
# {'text': 'This is amazing!', 'sentiment': 'Positive', 'confidence': 0.95, 'label': 1}

# Batch predictions
results = model.predict_batch([
    "I love this!",
    "This is terrible",
    "Pretty good"
])
```

### 3️⃣ Web API (REST)

```bash
# Start server
python app.py

# Access web UI
# Open browser: http://localhost:5000

# API endpoint
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```

## 📊 Model Performance

- **Algorithm**: Multinomial Naive Bayes
- **Accuracy**: ~71% on test set
- **Training Data**: Sentiment140 dataset (1.6M tweets)
- **Features**: TF-IDF (5000 features)
- **Preprocessing**: Emoji conversion, URL removal, lemmatization

## 🏗️ Architecture

```
User Input
    ↓
Preprocessor (preprocessor.py)
    ↓ Clean text
TF-IDF Vectorizer
    ↓ Feature vectors
Naive Bayes Model
    ↓ Prediction
Result (sentiment + confidence)
```

## 📁 Project Structure

```
sentinmentAnalysis/
├── Core Production Code
│   ├── preprocessor.py      # Text preprocessing
│   ├── model.py             # Model class
│   ├── train_model.py       # Training script
│   ├── predict.py           # CLI tool
│   └── app.py               # Web API
│
├── Documentation
│   ├── README.md            # Full docs
│   ├── QUICKSTART.md        # Quick guide
│   └── PRODUCTION_SETUP.md  # This file
│
├── Models (Generated/Existing)
│   ├── sentiment_model_nb.pkl      # Trained model
│   └── tfidf_vectorizer.pkl        # Vectorizer
│
├── Research (Kept for Reference)
│   ├── SocialMedia_Sentiment_Analysis.ipynb
│   └── distilbert_logistic_model.pkl
│
└── Configuration
    ├── requirements.txt
    ├── .gitignore
    └── test_model.py
```

## 🔧 Customization

### Change Training Data Size

Edit `train_model.py`, line 26:
```python
sample_size = 10000  # Change this number
```

### Modify Preprocessing

Edit `preprocessor.py`, method `clean_tweet()`:
- Add/remove stopwords
- Change lemmatization rules
- Adjust emoji handling

### Tune Model Parameters

Edit `model.py`, method `train()`:
- Change `max_features` for TF-IDF
- Adjust `test_size` for train/test split
- Try different classifiers

## 🌐 Deployment Options

### Local Development
```bash
python app.py
```

### Production (Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker (Create Dockerfile)
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

### Cloud Platforms
- **Heroku**: Add `Procfile` with `web: gunicorn app:app`
- **AWS**: Use Elastic Beanstalk or EC2
- **Google Cloud**: Use Cloud Run or App Engine
- **Azure**: Use App Service

## ✅ What's Production-Ready

✅ **Modular Code**: Clean separation of concerns
✅ **Error Handling**: Proper exception handling
✅ **Documentation**: Comprehensive docs and examples
✅ **Multiple Interfaces**: CLI, Python API, REST API
✅ **Testing**: Automated test script
✅ **Preprocessing**: Robust text cleaning
✅ **Model Persistence**: Save/load functionality
✅ **Web UI**: Beautiful, user-friendly interface
✅ **API Endpoints**: RESTful design
✅ **Batch Processing**: Efficient batch predictions

## 🔬 Deep Learning Models (Experimental)

The repository includes experimental deep learning models in the Jupyter notebook:
- DistilBERT embeddings
- DistilRoBERTa fine-tuning
- Transformer-based models

These are **not included in production** due to:
- Higher computational requirements
- Longer inference time
- Incomplete training/debugging

The Naive Bayes model provides a good balance of accuracy and speed for production use. Deep learning models can be explored for potential accuracy improvements in the future.

## 📈 Next Steps

1. **Test the system**: Run `python test_model.py`
2. **Try predictions**: Use `python predict.py`
3. **Start web app**: Run `python app.py`
4. **Integrate**: Use the Python API in your applications
5. **Deploy**: Choose a deployment platform
6. **Monitor**: Track prediction accuracy in production
7. **Improve**: Retrain with more data as needed

## 🐛 Common Issues

**Issue**: Missing dependencies
```bash
pip install -r requirements.txt
```

**Issue**: spaCy model not found
```bash
python -m spacy download en_core_web_sm
```

**Issue**: Model files missing
```bash
# If tfidf_vectorizer.pkl is missing, retrain:
python train_model.py
```

**Issue**: Port 5000 already in use
```bash
# Use different port
export PORT=8000
python app.py
```

## 📞 Getting Help

- **Documentation**: Check `README.md` for detailed information
- **Quick Setup**: See `QUICKSTART.md` for 5-minute guide
- **Testing**: Run `python test_model.py` to verify your setup
- **Code**: Review inline comments for implementation details
- **Issues**: Open a GitHub issue if you encounter problems
- **Contributing**: Pull requests are welcome!

## 🎉 Ready to Deploy!

This sentiment analysis system is production-ready and can be:
- ✅ Used via CLI for quick analysis
- ✅ Integrated into Python applications
- ✅ Deployed as a REST API web service
- ✅ Scaled for production workloads
- ✅ Customized for specific use cases

**Start analyzing sentiment now:**
```bash
python predict.py "I love this!"
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs or issues
- Suggest new features
- Improve documentation
- Submit pull requests

## 📄 License

This project is open source and available for educational and commercial use.

---

Built with ❤️ for sentiment analysis. Happy analyzing! 🚀
