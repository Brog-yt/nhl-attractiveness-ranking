import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from face_processer import FaceProcesser
from kaggle_data import KaggleData
import numpy as np
import pickle
import joblib
import pandas as pd

# Configuration
REGENERATE_EMBEDDINGS = False
CACHE_FILE = "embeddings_cache.pkl"
XGBOOST_MODEL_FILE = "xgboost_attractiveness_model.pkl"
LIGHTGBM_MODEL_FILE = "lightgbm_attractiveness_model.pkl"

# Load datasets
print("Loading datasets...")
kaggle_data = KaggleData()
df_scut = kaggle_data.getSCUTData()

# Use only SCUT data
df_combined = df_scut
print(f"SCUT dataset size: {len(df_scut)}")

# Initialize FaceProcesser
processor = FaceProcesser()

# Check if models already exist
xgb_model_path = Path(XGBOOST_MODEL_FILE)
lgb_model_path = Path(LIGHTGBM_MODEL_FILE)

if xgb_model_path.exists() and lgb_model_path.exists():
    print(f"\nModels already exist!")
    print(f"Loading XGBoost model from {XGBOOST_MODEL_FILE}...")
    xgb_model = joblib.load(xgb_model_path)
    print("XGBoost model loaded successfully!")
    
    print(f"Loading LightGBM model from {LIGHTGBM_MODEL_FILE}...")
    lgb_model = joblib.load(lgb_model_path)
    print("LightGBM model loaded successfully!")
else:
    print(f"\nModels not found. Training new ensemble models...")
    
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
    
    # ============================================
    # XGBoost Model
    # ============================================
    print("\n" + "="*60)
    print("Training XGBoost Model...")
    print("="*60)
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=150,           # Number of boosting rounds (reduced)
        max_depth=5,                # Shallower trees (prevents overfitting)
        learning_rate=0.01,         # Slower learning (more conservative)
        subsample=0.7,              # Stronger row sampling randomness
        colsample_bytree=0.7,       # Stronger feature sampling randomness
        reg_alpha=0.5,              # Stronger L1 regularization
        reg_lambda=2.0,             # Stronger L2 regularization
        objective='reg:squarederror',
        random_state=42,
        verbosity=1
    )
    
    xgb_model.fit(X_train, y_train)
    
    # Evaluate XGBoost
    xgb_train_pred = xgb_model.predict(X_train)
    xgb_test_pred = xgb_model.predict(X_test)
    
    xgb_train_mse = mean_squared_error(y_train, xgb_train_pred)
    xgb_test_mse = mean_squared_error(y_test, xgb_test_pred)
    xgb_test_mae = mean_absolute_error(y_test, xgb_test_pred)
    
    print(f"\nXGBoost Results:")
    print(f"  Train MSE: {xgb_train_mse:.6f}")
    print(f"  Test MSE:  {xgb_test_mse:.6f}")
    print(f"  Test MAE:  {xgb_test_mae:.6f}")
    
    # Save XGBoost model
    print(f"\nSaving XGBoost model to {XGBOOST_MODEL_FILE}...")
    joblib.dump(xgb_model, xgb_model_path)
    print("XGBoost model saved successfully!")
    
    # ============================================
    # LightGBM Model
    # ============================================
    print("\n" + "="*60)
    print("Training LightGBM Model...")
    print("="*60)
    
    lgb_model = lgb.LGBMRegressor(
        n_estimators=150,           # Number of boosting rounds (reduced for stability)
        max_depth=5,                # Shallower trees (prevents overfitting)
        num_leaves=20,              # Fewer leaves per tree
        min_data_in_leaf=40,        # More samples required per leaf
        learning_rate=0.01,         # Slower learning (more conservative)
        subsample=0.7,              # Stronger row sampling randomness
        colsample_bytree=0.7,       # Stronger feature sampling randomness
        reg_alpha=0.5,              # Stronger L1 regularization
        reg_lambda=2.0,             # Stronger L2 regularization
        random_state=42,
        verbose=-1                  # Suppress verbose output
    )
    
    lgb_model.fit(X_train, y_train)
    
    # Evaluate LightGBM
    lgb_train_pred = lgb_model.predict(X_train)
    lgb_test_pred = lgb_model.predict(X_test)
    
    lgb_train_mse = mean_squared_error(y_train, lgb_train_pred)
    lgb_test_mse = mean_squared_error(y_test, lgb_test_pred)
    lgb_test_mae = mean_absolute_error(y_test, lgb_test_pred)
    
    print(f"\nLightGBM Results:")
    print(f"  Train MSE: {lgb_train_mse:.6f}")
    print(f"  Test MSE:  {lgb_test_mse:.6f}")
    print(f"  Test MAE:  {lgb_test_mae:.6f}")
    
    # Save LightGBM model
    print(f"\nSaving LightGBM model to {LIGHTGBM_MODEL_FILE}...")
    joblib.dump(lgb_model, lgb_model_path)
    print("LightGBM model saved successfully!")
    
    # ============================================
    # Model Comparison
    # ============================================
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"{'Model':<15} {'Train MSE':<12} {'Test MSE':<12} {'Test MAE':<12}")
    print("-"*60)
    print(f"{'XGBoost':<15} {xgb_train_mse:<12.6f} {xgb_test_mse:<12.6f} {xgb_test_mae:<12.6f}")
    print(f"{'LightGBM':<15} {lgb_train_mse:<12.6f} {lgb_test_mse:<12.6f} {lgb_test_mae:<12.6f}")
    print("="*60)


# ============================================
# Predictions on Test Image
# ============================================
test_image_path = "simpson.jpg"
if Path(test_image_path).exists():
    print(f"\n{'='*60}")
    print(f"Predicting attractiveness for: {test_image_path}")
    print(f"{'='*60}")
    
    try:
        test_embedding = processor.get_embedding_from_path(test_image_path)
        test_embedding = test_embedding.reshape(1, -1)
        
        xgb_prediction = xgb_model.predict(test_embedding)[0]
        lgb_prediction = lgb_model.predict(test_embedding)[0]
        
        # Ensemble prediction (average of both models)
        ensemble_prediction = (xgb_prediction + lgb_prediction) / 2
        
        print(f"XGBoost prediction:    {xgb_prediction:.4f}")
        print(f"LightGBM prediction:   {lgb_prediction:.4f}")
        print(f"Ensemble prediction:   {ensemble_prediction:.4f}")
        print(f"\nAttractiveness score (1-5 scale): {ensemble_prediction:.2f}")
    except Exception as e:
        print(f"Error predicting: {e}")
else:
    print(f"\nTest image not found: {test_image_path}")
    print("Skipping prediction step")
