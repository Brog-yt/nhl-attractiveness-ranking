import cv2
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
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

# Gender filter - set to 'male', 'female', or None for all
GENDER_FILTER = None  # Use all genders for better generalization

# Set cache and model filenames based on gender filter
CACHE_DIR = Path("cached-models")
CACHE_DIR.mkdir(exist_ok=True)

if GENDER_FILTER:
    CACHE_FILE = CACHE_DIR / f"embeddings_cache_{GENDER_FILTER}.pkl"
    MODEL_FILE = CACHE_DIR / f"beauty_score_model_{GENDER_FILTER}.pkl"
else:
    CACHE_FILE = CACHE_DIR / "embeddings_cache.pkl"
    MODEL_FILE = CACHE_DIR / "beauty_score_model.pkl"

# Load datasets with gender filter
kaggle_data = KaggleData()
df_scut = kaggle_data.getSCUTData(gender=GENDER_FILTER)

print(f"Gender filter: {GENDER_FILTER or 'None (all genders)'}")
print(f"SCUT samples: {len(df_scut)}")

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
        
        for i in range(len(df_scut)):
            try:
                embedding = processor.get_embedding_from_path(df_scut.loc[i, "path"])
                embeddings.append(embedding)
                scores.append(df_scut.loc[i, "score"])
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(df_scut)} images")
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
    
    # Standardize embeddings (important for Ridge regression)
    print("\nStandardizing embeddings...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Find optimal alpha using cross-validation
    print("\nFinding optimal alpha parameter...")
    alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    best_alpha = None
    best_cv_score = float('-inf')
    
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        cv_scores = cross_val_score(ridge, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_mse = -cv_scores.mean()
        print(f"  Alpha={alpha:<7.3f} -> CV MSE: {cv_mse:.6f}")
        if -cv_scores.mean() < best_cv_score or best_cv_score == float('-inf'):
            best_cv_score = -cv_scores.mean()
            best_alpha = alpha
    
    print(f"\nBest alpha: {best_alpha} (CV MSE: {best_cv_score:.6f})")
    
    # Train Ridge regression model with optimal alpha
    model = Ridge(alpha=best_alpha)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on train and test data
    train_predictions = model.predict(X_train_scaled)
    test_predictions = model.predict(X_test_scaled)
    
    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)
    
    print(f"\nTrain MSE: {train_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    
    # Save the model and scaler
    model_path = Path(MODEL_FILE)
    scaler_path = model_path.parent / (model_path.stem + "_scaler.pkl")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\nModel saved to {MODEL_FILE}")
    print(f"Scaler saved to {scaler_path}")

# Predict brad.png
image_path = "brad.png"
print(f"\nPredicting beauty score for image: {image_path}")

# Load scaler if it exists
scaler_path = model_path.parent / (model_path.stem + "_scaler.pkl") if 'model_path' in locals() else Path(MODEL_FILE).parent / (Path(MODEL_FILE).stem + "_scaler.pkl")
if scaler_path.exists():
    scaler = joblib.load(scaler_path)
    embedding = processor.get_embedding_from_path(image_path)
    embedding_scaled = scaler.transform(embedding.reshape(1, -1))
    predicted_score = model.predict(embedding_scaled)[0]
else:
    embedding = processor.get_embedding_from_path(image_path)
    predicted_score = model.predict(embedding.reshape(1, -1))[0]

print(f"Predicted beauty score: {predicted_score:.4f}")
