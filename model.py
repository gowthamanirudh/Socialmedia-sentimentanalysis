"""
Sentiment Analysis Model Module
Handles model training, loading, and prediction
"""

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocessor import TextPreprocessor


class SentimentModel:
    """Sentiment Analysis Model using Naive Bayes"""
    
    def __init__(self, model_path=None, vectorizer_path=None):
        """
        Initialize model
        
        Args:
            model_path (str): Path to saved model file
            vectorizer_path (str): Path to saved vectorizer file
        """
        self.preprocessor = TextPreprocessor()
        self.vectorizer = None
        self.model = None
        
        if model_path and vectorizer_path:
            self.load_model(model_path, vectorizer_path)
    
    def train(self, texts, labels, max_features=5000, test_size=0.2, random_state=42):
        """
        Train the sentiment analysis model
        
        Args:
            texts (list): List of raw text data
            labels (list): List of sentiment labels (0=negative, 1=positive)
            max_features (int): Maximum number of features for TF-IDF
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Training metrics including accuracy, confusion matrix, and report
        """
        print("Preprocessing texts...")
        cleaned_texts = self.preprocessor.preprocess_batch(texts)
        
        print("Vectorizing texts...")
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        X = self.vectorizer.fit_transform(cleaned_texts)
        
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=test_size, random_state=random_state
        )
        
        print("Training Naive Bayes model...")
        self.model = MultinomialNB()
        self.model.fit(X_train, y_train)
        
        print("Evaluating model...")
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        print(f"\nModel Accuracy: {metrics['accuracy']:.4f}")
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        print("\nClassification Report:")
        print(metrics['classification_report'])
        
        return metrics
    
    def predict(self, text):
        """
        Predict sentiment for a single text
        
        Args:
            text (str): Raw text to analyze
            
        Returns:
            dict: Prediction result with sentiment and confidence
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained or loaded. Train or load a model first.")
        
        # Preprocess
        cleaned_text = self.preprocessor.preprocess_text(text)
        
        # Vectorize
        X = self.vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = probabilities[prediction]
        
        return {
            'text': text,
            'sentiment': sentiment,
            'confidence': float(confidence),
            'label': int(prediction)
        }
    
    def predict_batch(self, texts):
        """
        Predict sentiment for multiple texts
        
        Args:
            texts (list): List of raw texts to analyze
            
        Returns:
            list: List of prediction results
        """
        return [self.predict(text) for text in texts]
    
    def save_model(self, model_path='sentiment_model_nb.pkl', 
                   vectorizer_path='tfidf_vectorizer.pkl'):
        """
        Save trained model and vectorizer
        
        Args:
            model_path (str): Path to save model
            vectorizer_path (str): Path to save vectorizer
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("No model to save. Train a model first.")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
    
    def load_model(self, model_path='sentiment_model_nb.pkl', 
                   vectorizer_path='tfidf_vectorizer.pkl'):
        """
        Load trained model and vectorizer
        
        Args:
            model_path (str): Path to model file
            vectorizer_path (str): Path to vectorizer file
        """
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        print(f"Model loaded from {model_path}")
        print(f"Vectorizer loaded from {vectorizer_path}")
