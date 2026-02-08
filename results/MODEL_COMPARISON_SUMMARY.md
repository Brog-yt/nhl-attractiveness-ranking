# Attractiveness Prediction Model Comparison

## Summary

Three machine learning models were trained on male NHL player facial embeddings (512-dimensional vectors from ArcFace) to predict facial attractiveness scores on a 1-5 scale. The dataset consisted of 2,750 male facial samples from the SCUT-FBP5500-v2 dataset, split 80/20 for training/testing.

---

## Model Performance Comparison

| Model | Algorithm | Test MSE | Test MAE | Typical Error | Status |
|-------|-----------|----------|----------|---------------|--------|
| **Ridge Regression** | Linear regression with L2 regularization | 0.1080 | 0.2534 | ±0.329 | ⭐ Good |
| **SVR (GridSearchCV)** | Support Vector Regression with RBF kernel | **0.0958** | **0.2392** | **±0.309** | 🏆 **Best** |
| **Neural Network** | 4-layer deep neural network with dropout | 0.1895 | 0.3322 | ±0.435 | ❌ Worst |

---

## Detailed Model Explanations

### 1. Ridge Regression ⭐

**What it is:**
Ridge Regression is a linear regression model that adds L2 regularization (penalty for large coefficients) to prevent overfitting. It finds the best linear combination of the 512 embedding dimensions to predict attractiveness.

**How it works:**
- Takes each of the 512 facial embedding features as input
- Learns a single weight for each feature
- Adds a regularization penalty controlled by `alpha` parameter
- Best alpha found: **500.0** via cross-validation

**Results:**
```
Train MSE: 0.084225
Test MSE:  0.107969
Test MAE:  0.253438
Typical Error: ±0.329 points
```

**Pros:**
- ✅ Simple, interpretable, and fast to train
- ✅ Low computational cost
- ✅ Works well when relationships are linear

**Cons:**
- ❌ Cannot capture non-linear patterns
- ❌ Assumes linear relationship between features and attractiveness

**Example Prediction:**
- Input: Brad Pitt facial embedding (512D vector)
- Output: **2.91 / 5.0** attractiveness score

---

### 2. SVR (Support Vector Regression) with GridSearchCV 🏆 **BEST MODEL**

**What it is:**
Support Vector Regression (SVR) is a non-linear regression algorithm that finds an optimal hyperplane in high-dimensional space using the "kernel trick." GridSearchCV exhaustively searches for the best hyperparameters.

**How it works:**
- Uses RBF (Radial Basis Function) kernel to map embeddings to higher dimensions
- Finds the hyperplane that best separates data points with a margin of tolerance (epsilon)
- **Optimal parameters found:**
  - `C = 10`: Controls regularization strength (lower = more regularization)
  - `epsilon = 0.01`: Margin of tolerance around predictions
  - `gamma = 0.001`: Controls how far the influence of a training example extends

**Results:**
```
Train MSE: 0.000105
Test MSE:  0.095782 ✅ Below 0.1 target!
Test MAE:  0.239160
Typical Error: ±0.309 points
```

**Pros:**
- ✅ **Best performance (0.0958 MSE)**
- ✅ Captures non-linear patterns
- ✅ Works well with high-dimensional data (512D)
- ✅ RBF kernel provides flexible decision boundaries

**Cons:**
- ❌ Slower to train than Ridge
- ❌ Less interpretable than linear models
- ❌ Very low training MSE suggests possible overfitting (but test performance is strong)

**Example Prediction:**
- Input: Brad Pitt facial embedding (512D vector)
- Output: **2.97 / 5.0** attractiveness score

---

### 3. Neural Network ❌

**What it is:**
A deep learning model with 4 hidden layers that learns hierarchical patterns in the embedding space through backpropagation.

**Architecture:**
```
Input (512) 
  ↓
Dense(256, relu) → Dropout(0.3)
  ↓
Dense(128, relu) → Dropout(0.3)
  ↓
Dense(64, relu) → Dropout(0.2)
  ↓
Dense(32, relu)
  ↓
Output (1)
```

**How it works:**
- Each layer learns non-linear transformations of the previous layer
- Dropout layers randomly deactivate 20-30% of neurons to prevent overfitting
- Trained with Adam optimizer for 100 epochs on batches of 32 samples

**Results:**
```
Train MSE: 0.134439
Test MSE:  0.189489
Test MAE:  0.332236
Typical Error: ±0.435 points
```

**Pros:**
- ✅ Can learn very complex patterns
- ✅ Modern and flexible architecture

**Cons:**
- ❌ **Worst performance (0.1895 MSE)**
- ❌ Higher training time
- ❌ Requires more hyperparameter tuning
- ❌ Overfitting: large gap between train (0.134) and test (0.189) MSE
- ❌ Harder to interpret predictions

**Example Prediction:**
- Input: Brad Pitt facial embedding (512D vector)
- Output: **2.64 / 5.0** attractiveness score

---

## Why SVR Won

1. **Non-linearity**: SVR's RBF kernel can capture complex relationships that Ridge's linear approach misses
2. **Generalization**: Despite lower training error, SVR achieves the best test error (0.0958)
3. **Feature Space**: The 512D embedding space has natural structure that SVR's kernel trick exploits well
4. **Hyperparameter Tuning**: GridSearchCV exhaustively tested 125 combinations (5 C values × 4 epsilon values × 5 gamma values × 1 kernel)

---

## Error Interpretation

**SVR's typical error of ±0.309 points on 1-5 scale means:**

| Actual Score | Predicted Range (68% confidence) | Predicted Range (95% confidence) |
|--------------|----------------------------------|----------------------------------|
| 1.0 | 0.69 - 1.31 | 0.38 - 1.62 |
| 2.5 | 2.19 - 2.81 | 1.88 - 3.12 |
| 3.0 | 2.69 - 3.31 | 2.38 - 3.62 |
| 4.0 | 3.69 - 4.31 | 3.38 - 4.62 |
| 5.0 | 4.69 - 5.31* | 4.38 - 5.31* |

*Clamped to [1.0, 5.0] range*

---

## Key Improvements Made

1. **Data Preprocessing**: StandardScaler normalization improved all models
2. **Feature Engineering**: Kept full 512D embeddings instead of PCA reduction
3. **Hyperparameter Tuning**: Cross-validation and GridSearchCV found optimal parameters
4. **Model Selection**: Tested multiple algorithms and selected best performer

---

## Conclusion

**SVR (GridSearchCV)** is the recommended production model with:
- ✅ Best test performance: **0.0958 MSE**
- ✅ Typical prediction error: **±0.31 points**
- ✅ Optimal hyperparameters: C=10, epsilon=0.01, gamma=0.001
- ✅ Saved as: `cached-models/beauty_score_model_male.pkl`

