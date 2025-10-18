"""
Test script to verify the production model works correctly
"""

from model import SentimentModel


def test_predictions():
    """Test model with various examples"""
    print("=" * 60)
    print("Testing Sentiment Analysis Model")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    try:
        model = SentimentModel(
            model_path='sentiment_model_nb.pkl',
            vectorizer_path='tfidf_vectorizer.pkl'
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nPlease run 'python train_model.py' first to train the model.")
        return
    
    # Test cases
    test_cases = [
        # Positive examples
        "I love this product! It's amazing!",
        "Best day ever! So happy! 😊",
        "Great customer service, highly recommend!",
        "This is awesome! Thank you so much!",
        
        # Negative examples
        "This is terrible. Worst experience ever.",
        "I hate this. Complete waste of money.",
        "Disappointed and frustrated. Never again.",
        "Awful service. Very unhappy. 😠",
        
        # Mixed/Neutral
        "The weather is okay today",
        "Just finished work",
        "It's alright, nothing special",
        
        # With emojis
        "Love this! 😍❤️🎉",
        "So sad 😢😭",
        
        # With URLs and mentions
        "Check this out: https://example.com @user Amazing!",
        "Terrible product from @company http://link.com",
    ]
    
    print("\n" + "=" * 60)
    print("Running Test Cases")
    print("=" * 60)
    
    correct_predictions = 0
    total_tests = 0
    
    for text in test_cases:
        try:
            result = model.predict(text)
            
            # Display result
            emoji_icon = "😊" if result['sentiment'] == "Positive" else "😞"
            confidence_bar = "█" * int(result['confidence'] * 20)
            
            print(f"\n{emoji_icon} {result['sentiment']} ({result['confidence']:.1%})")
            print(f"   [{confidence_bar}]")
            print(f"   Text: {text[:70]}...")
            
            total_tests += 1
            
        except Exception as e:
            print(f"\n✗ Error predicting: {text}")
            print(f"   Error: {e}")
    
    print("\n" + "=" * 60)
    print(f"Tests completed: {total_tests}/{len(test_cases)}")
    print("=" * 60)
    
    # Test batch prediction
    print("\n" + "=" * 60)
    print("Testing Batch Prediction")
    print("=" * 60)
    
    batch_texts = [
        "I love this!",
        "This is terrible",
        "Pretty good experience"
    ]
    
    try:
        results = model.predict_batch(batch_texts)
        print(f"\n✓ Batch prediction successful for {len(results)} texts")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['sentiment']} ({result['confidence']:.1%})")
            print(f"   {result['text']}")
    except Exception as e:
        print(f"\n✗ Batch prediction failed: {e}")
    
    print("\n" + "=" * 60)
    print("All Tests Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_predictions()
