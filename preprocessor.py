"""
Text Preprocessing Module for Sentiment Analysis
Handles cleaning and preprocessing of social media text
"""

import re
import emoji
import spacy
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords


class TextPreprocessor:
    """Preprocessor for social media text with emoji handling"""
    
    def __init__(self):
        """Initialize preprocessor with required NLP tools"""
        # Download required NLTK data
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        # Load spaCy model
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading spaCy model...")
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
            self.nlp = spacy.load('en_core_web_sm')
        
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = TweetTokenizer()
    
    def clean_tweet(self, tweet):
        """
        Clean and preprocess a single tweet
        
        Args:
            tweet (str): Raw tweet text
            
        Returns:
            list: List of cleaned and lemmatized tokens
        """
        # Convert emojis to text with spaces between
        tweet = emoji.demojize(tweet, delimiters=(" ", " "))
        
        # Remove URLs
        tweet = re.sub(r"http\S+|www.\S+", "", tweet)
        
        # Remove mentions and hashtags
        tweet = re.sub(r"@\w+", "", tweet)
        tweet = tweet.replace("#", "")
        
        # Remove special characters (keep underscores for emoji text)
        tweet = re.sub(r"[^a-zA-Z0-9_ ]+", "", tweet)
        
        # Lowercase
        tweet = tweet.lower()
        
        # Tokenize
        tokens = self.tokenizer.tokenize(tweet)
        
        # Remove stopwords
        tokens = [word for word in tokens if word not in self.stop_words]
        
        # Lemmatize
        doc = self.nlp(" ".join(tokens))
        lemmas = [token.lemma_ for token in doc if token.lemma_ != "-PRON-"]
        
        return lemmas
    
    def preprocess_text(self, text):
        """
        Preprocess text and return as string
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text as string
        """
        tokens = self.clean_tweet(text)
        return " ".join(tokens)
    
    def preprocess_batch(self, texts):
        """
        Preprocess multiple texts
        
        Args:
            texts (list): List of raw texts
            
        Returns:
            list: List of cleaned texts as strings
        """
        return [self.preprocess_text(text) for text in texts]
