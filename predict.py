"""
Prediction Script for Sentiment Analysis
Load trained model and make predictions
"""

import sys
from model import SentimentModel


def main():
    """Main prediction function"""
    # Load trained model
    print("Loading trained model...")
    sentiment_model = SentimentModel(
        model_path='sentiment_model_nb.pkl',
        vectorizer_path='tfidf_vectorizer.pkl'
    )
    
    print("\n" + "=" * 60)
    print("Sentiment Analysis - Prediction Mode")
    print("=" * 60)
    print("Type 'quit' or 'exit' to stop\n")
    
    # Interactive prediction loop
    while True:
        text = input("Enter text to analyze: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not text:
            print("Please enter some text.\n")
            continue
        
        # Make prediction
        result = sentiment_model.predict(text)
        
        # Display result
        print("\n" + "-" * 60)
        print(f"Text: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("-" * 60 + "\n")


def predict_single(text):
    """
    Predict sentiment for a single text (for programmatic use)
    
    Args:
        text (str): Text to analyze
        
    Returns:
        dict: Prediction result
    """
    sentiment_model = SentimentModel(
        model_path='sentiment_model_nb.pkl',
        vectorizer_path='tfidf_vectorizer.pkl'
    )
    return sentiment_model.predict(text)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line argument provided
        text = " ".join(sys.argv[1:])
        result = predict_single(text)
        print(f"\nText: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2%}\n")
    else:
        # Interactive mode
        main()
