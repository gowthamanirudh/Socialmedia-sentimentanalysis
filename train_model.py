"""
Training Script for Sentiment Analysis Model
Trains the model on Sentiment140 dataset
"""

import pandas as pd
from model import SentimentModel


def main():
    """Main training function"""
    print("=" * 60)
    print("Sentiment Analysis Model Training")
    print("=" * 60)
    
    # Load dataset
    print("\nLoading Sentiment140 dataset...")
    df = pd.read_csv(
        'sentiment140-data/training.1600000.processed.noemoticon.csv',
        encoding='latin-1',
        header=None
    )
    df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
    
    # Convert labels: 0 stays 0 (negative), 4 becomes 1 (positive)
    df['target'] = df['target'].replace(4, 1)
    
    # Sample data for training (adjust sample_size as needed)
    sample_size = 10000  # Increase for better accuracy, decrease for faster training
    print(f"Sampling {sample_size} tweets for training...")
    df_sample = df.sample(sample_size, random_state=42)
    
    # Check class balance
    print("\nClass distribution:")
    print(df_sample['target'].value_counts())
    
    # Extract texts and labels
    texts = df_sample['text'].tolist()
    labels = df_sample['target'].tolist()
    
    # Initialize and train model
    print("\nInitializing model...")
    sentiment_model = SentimentModel()
    
    print("\nStarting training...")
    metrics = sentiment_model.train(
        texts=texts,
        labels=labels,
        max_features=5000,
        test_size=0.2,
        random_state=42
    )
    
    # Save model
    print("\nSaving model...")
    sentiment_model.save_model(
        model_path='sentiment_model_nb.pkl',
        vectorizer_path='tfidf_vectorizer.pkl'
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nFinal Accuracy: {metrics['accuracy']:.4f}")
    print("\nModel files saved:")
    print("  - sentiment_model_nb.pkl")
    print("  - tfidf_vectorizer.pkl")


if __name__ == "__main__":
    main()
