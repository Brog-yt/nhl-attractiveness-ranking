import cv2
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from face_processer import FaceProcesser
from kaggle_data import KaggleData
from london_data_fetching import LondonDataFetching
import numpy as np
import pickle
import joblib
import pandas as pd

# Flag to regenerate embeddings cache
REGENERATE_EMBEDDINGS = False
CACHE_FILE = "embeddings_cache.pkl"
MODEL_FILE = "beauty_score_model.pkl"

# Load datasets
kaggle_data = KaggleData()
df_scut = kaggle_data.getSCUTData()

london_data = LondonDataFetching()
df_london = london_data.get_london_data()

# Combine datasets
df_combined = pd.concat([df_scut, df_london], ignore_index=True)
print(f"SCUT dataset size: {len(df_scut)}")
print(f"London dataset size: {len(df_london)}")
print(f"Combined dataset size: {len(df_combined)}")

# Initialize FaceProcesser
processor = FaceProcesser()

# Check if model already exists
model_path = Path(MODEL_FILE)
if model_path.exists():
    print(f"Model already exists: {MODEL_FILE}")
    print("Loading model...")
    model = joblib.load(model_path)
    print("Model loaded successfully!")
else:
    print(f"Model not found: {MODEL_FILE}")
    print("Training new model...")
    
    # Collect embeddings and scores
    cache_path = Path(CACHE_FILE)
    
    if cache_path.exists() and not REGENERATE_EMBEDDINGS:
        # Load from cache
        print(f"\nLoading embeddings from cache: {CACHE_FILE}")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
            embeddings = data["embeddings"]
            scores = data["scores"]
        print(f"Loaded {len(embeddings)} embeddings from cache")
    else:
        # Generate embeddings
        print(f"\nGenerating embeddings from combined dataset (this may take a while)...")
        embeddings = []
        scores = []
        
        for i in range(len(df_combined)):
            try:
                embedding = processor.get_embedding_from_path(df_combined.loc[i, "path"])
                embeddings.append(embedding)
                scores.append(df_combined.loc[i, "score"])
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(df_combined)} images")
            except Exception as e:
                print(f"Error on image {i}: {e}")
        
        # Save to cache
        print(f"\nSaving embeddings to cache: {CACHE_FILE}")
        with open(cache_path, "wb") as f:
            pickle.dump({"embeddings": embeddings, "scores": scores}, f)
        print("Cache saved successfully")
    
    # Convert to numpy arrays
    X = np.array(embeddings)  # shape: (n_samples, 512)
    y = np.array(scores)      # shape: (n_samples,)
    
    print(f"\nTotal samples: {len(X)}")
    
    # Split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train Ridge regression model
    model = Ridge(alpha=1.0)  # alpha controls regularization strength
    model.fit(X_train, y_train)
    
    # Evaluate on train and test data
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)
    
    print(f"\nTrain MSE: {train_mse}")
    print(f"Test MSE: {test_mse}")
    
    # Save the model
    model_path = Path(MODEL_FILE)
    joblib.dump(model, model_path)
    print(f"\nModel saved to {MODEL_FILE}")

# Predict brad.png
image_path = "tammy.png"
print(f"\nPredicting beauty score for image: {image_path}")
embedding = processor.get_embedding_from_path(image_path)
predicted_score = model.predict(embedding.reshape(1, -1))[0]
print(f"Predicted beauty score: {predicted_score:.4f}")
