import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.applications import ResNet50, EfficientNetB0
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from kaggle_data import KaggleData
from london_data_fetching import LondonDataFetching

# Configuration
IMG_SIZE = 224  # ResNet50 and EfficientNetB0 expect 224x224
BATCH_SIZE = 32
EPOCHS = 20
MODEL_FILE = "cnn_attractiveness_model.h5"
COMBINED_CACHE = "combined_images_cache.pkl"

# Load datasets
print("Loading datasets...")
kaggle_data = KaggleData()
df_scut = kaggle_data.getSCUTData()

# Combine datasets
df_combined = pd.concat([df_scut], ignore_index=True)
print(f"SCUT dataset size: {len(df_scut)}")
print(f"Combined dataset size: {len(df_combined)}")


def preprocess_image(image_path, img_size=IMG_SIZE):
    """
    Load and preprocess a single image.
    
    Args:
        image_path: Path to the image file
        img_size: Target image size (default 224)
    
    Returns:
        Preprocessed numpy array or None if loading fails
    """
    try:
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Could not load image: {image_path}")
            return None
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        img = cv2.resize(img, (img_size, img_size))
        
        # Normalize to [0, 1]
        img = img.astype('float32') / 255.0
        
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def create_image_dataset(df, img_size=IMG_SIZE):
    """
    Create image and score arrays from DataFrame.
    Filters out images that fail to load.
    
    Args:
        df: DataFrame with 'path' and 'score' columns
        img_size: Target image size
    
    Returns:
        Tuple of (images array, scores array)
    """
    images = []
    scores = []
    
    for idx, row in df.iterrows():
        img = preprocess_image(row['path'], img_size)
        if img is not None:
            images.append(img)
            scores.append(row['score'])
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)} images")
    
    return np.array(images), np.array(scores)


def build_transfer_learning_model(img_size=IMG_SIZE, model_type='resnet50'):
    """
    Build a transfer learning model for attractiveness prediction.
    
    Args:
        img_size: Image size (224 or 256)
        model_type: 'resnet50' or 'efficientnetb0'
    
    Returns:
        Compiled Keras model
    """
    # Load pre-trained base model (without top classification layer)
    if model_type == 'resnet50':
        base_model = ResNet50(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights='imagenet'
        )
    else:  # efficientnetb0
        base_model = EfficientNetB0(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights='imagenet'
        )
    
    # Freeze early layers (keep pre-trained weights)
    # Unfreeze last few layers for fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:-30]:  # Freeze all but last 30 layers
        layer.trainable = False
    
    # Create custom head for regression
    model = keras.Sequential([
        layers.Input(shape=(img_size, img_size, 3)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)  # Output: single score (1-5 range)
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


# Check if model already exists
model_path = Path(MODEL_FILE)
if model_path.exists():
    print(f"\nModel already exists: {MODEL_FILE}")
    print("Loading model...")
    model = keras.models.load_model(model_path)
    print("Model loaded successfully!")
else:
    print(f"\nModel not found: {MODEL_FILE}")
    print("Training new model...")
    
    # Load and preprocess images
    print("\nPreprocessing images...")
    X, y = create_image_dataset(df_combined, IMG_SIZE)
    print(f"Loaded {len(X)} images successfully")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Build model
    print("\nBuilding transfer learning model...")
    model = build_transfer_learning_model(IMG_SIZE, model_type='resnet50')
    print(model.summary())
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    
    # Get predictions for MSE calculation
    train_predictions = model.predict(X_train, verbose=0)
    test_predictions = model.predict(X_test, verbose=0)
    
    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)
    
    print(f"\nTrain MSE: {train_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    
    # Save model
    print(f"\nSaving model to {MODEL_FILE}...")
    model.save(model_path)
    print("Model saved successfully!")


# Predict on a test image
test_image_path = "simpson.jpg"
if Path(test_image_path).exists():
    print(f"\n{'='*50}")
    print(f"Predicting attractiveness for: {test_image_path}")
    print(f"{'='*50}")
    
    test_img = preprocess_image(test_image_path, IMG_SIZE)
    if test_img is not None:
        # Add batch dimension
        test_img_batch = np.expand_dims(test_img, axis=0)
        prediction = model.predict(test_img_batch, verbose=0)[0][0]
        print(f"Predicted attractiveness score: {prediction:.4f} (1-5 scale)")
    else:
        print("Failed to load test image")
else:
    print(f"\nTest image not found: {test_image_path}")
    print("Skipping prediction step")
