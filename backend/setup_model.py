# backend/setup_model.py
# Script to download and save the sentence transformer model locally

import os
from sentence_transformers import SentenceTransformer

def download_and_save_model():
    """Download and save the all-MiniLM-L6-v2 model locally."""
    model_name = 'all-MiniLM-L6-v2'
    save_path = './backend/all-MiniLM-L6-v2'
    
    print(f"Downloading {model_name} model...")
    try:
        model = SentenceTransformer(model_name)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the model
        model.save(save_path)
        print(f"✓ Model saved successfully to {save_path}")
        
        # Test the model
        test_sentences = ["This is a test sentence.", "Another test sentence."]
        embeddings = model.encode(test_sentences)
        print(f"✓ Model test successful. Generated embeddings shape: {embeddings.shape}")
        
    except Exception as e:
        print(f"✗ Error downloading/saving model: {e}")
        raise

if __name__ == "__main__":
    download_and_save_model()