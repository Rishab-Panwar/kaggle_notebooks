# System Threat Forecaster - Predicting Malware Infections (Machine Learning Approach)

> **Competition:** System Threat Forecaster · Kaggle Competition · Machine Learning Practice Course       
> **Task:** Predict probability of malware infection based on system telemetry data  
> **Metric:** Classification Accuracy - higher is better  
> **Approach:** Gradient Boosting ensemble (LightGBM + CatBoost + XGBoost)  
> **Result:** 42/905

---

## The Problem

The task is to predict whether a system will get infected by malware based on 75 features describing its configuration, security posture, hardware specs, and update history. Each row represents a unique machine identified by `MachineID`, and the target is binary: `1` if malware was detected, `0` if clean.

The data comes from real-world antivirus telemetry - threat reports collected from millions of Windows machines. This is not a synthetic dataset with clean feature relationships. It's messy, real-world security data with:

- **Missing values** (up to 98% for some features like `SMode`)
- **High cardinality categorical features** (`CityID` with 16,000+ unique values)
- **Mixed data types** (numerical, categorical, timestamps)
- **Class imbalance** (malware is rarer than clean systems)
- **Feature redundancy** (multiple OS version fields, overlapping security configurations)

Critically, **this is not a feature importance problem** where you identify "the top 10 risk factors." The signal is distributed across many weak features that only become predictive in combination. A single outdated antivirus signature means little; combined with disabled real-time protection, Windows 7, and no TPM, it's a strong signal.

---

## How I Thought About It

My initial instinct was to treat this as a standard classification problem: impute missing values, one-hot encode categoricals, train a Random Forest. This baseline achieved **~72% accuracy** - barely better than predicting the majority class.

The breakthrough came from recognizing this as a **tabular data problem** where gradient boosting models (LightGBM, XGBoost, CatBoost) consistently outperform traditional ML on real-world datasets due to:

1. **Native handling of categorical features** (no need for manual encoding that explodes dimensionality)
2. **Robustness to missing values** (can split on "missing" as a category)
3. **Automatic interaction learning** (discovers complex feature combinations without manual engineering)

But even gradient boosting needed help. The raw features contained noise and redundancy. The key insights:

### Insight 1: Temporal Features Hidden in Strings

Two critical features were disguised as strings: `DateAS` (antivirus signature date) and `DateOS` (OS update timestamp). These contain rich temporal signals:

- **Staleness** - How outdated is the antivirus? (days since signature update)
- **Seasonality** - Certain months have more infections (holiday shopping scams)
- **Day of week** - Weekend updates correlate with home users vs enterprise
- **Update frequency** - Gap between OS and AV updates signals neglect

Extracting these turned two string columns into 10 engineered features with strong predictive power.

### Insight 2: The Cardinality-Encoding Tradeoff

Features like `CityID` (16,000 categories) and `OEMModelID` (15,000 categories) pose a dilemma:

- **One-hot encoding** → 30,000+ sparse columns → memory explosion + overfitting
- **Label encoding** → ordinal relationship assumption (City 1 < City 2 < City 3) → garbage
- **Target encoding** → leaks training signal into features → validation overfit

**Solution:** Hybrid strategy:
- **Low cardinality** (<10 unique values) → One-hot encode
- **High cardinality** → Keep as-is, let tree models handle natively via splits

CatBoost and LightGBM have built-in categorical handling that learns optimal encodings during training.

### Insight 3: Ensemble Diversity via Algorithm Differences

Three gradient boosting algorithms, three different strengths:

| Model | Strength | Weakness |
|-------|----------|----------|
| **LightGBM** | Fastest, handles sparse features well | Can overfit on small datasets |
| **CatBoost** | Best native categorical handling, robust to noise | Slower training |
| **XGBoost** | Battle-tested, strong regularization | Needs manual categorical encoding |

Instead of picking "the best," I trained all three and averaged predictions. **Ensemble accuracy: 84.3%** vs best single model: 82.7%.

Why? Each model makes different mistakes based on its inductive bias. Averaging smooths out individual errors.

---

## How I Architected It

### The Full Pipeline

```
Raw Telemetry Data (75 features)
        ↓
┌────────────────────────────────┐
│   Feature Engineering          │
│  - Parse DateAS, DateOS        │
│  - Extract year, month, day    │
│  - Compute staleness (days)    │
│  - Day of week, hour           │
└───────────┬────────────────────┘
            ↓
┌────────────────────────────────┐
│   Missing Value Imputation     │
│  - Median for numerical        │
│  - Mode for categorical        │
│  - Indicator columns for NaNs  │
└───────────┬────────────────────┘
            ↓
┌────────────────────────────────┐
│   Cardinality-Based Encoding   │
│  - Low (<10): One-hot          │
│  - High (≥10): Native handling │
└───────────┬────────────────────┘
            ↓
        116 Features
            ↓
    ┌───────┴───────┐
    ↓               ↓
┌────────┐    ┌──────────┐    ┌──────────┐
│LightGBM│    │ CatBoost │    │ XGBoost  │
│ 500    │    │  500     │    │  500     │
│ trees  │    │  trees   │    │  trees   │
└───┬────┘    └─────┬────┘    └────┬─────┘
    │               │              │
    └───────┬───────┴──────┬───────┘
            ↓              ↓
      [pred₁]        [pred₂]        [pred₃]
            │              │              │
            └──────┬───────┴──────┬───────┘
                   ↓              
            Average Predictions
                   ↓
             Final Class
```

### Feature Engineering Details

#### Temporal Feature Extraction

```python
# DateAS: "2024-12-15T14:23:45Z" → multiple features
df['DateAS_year'] = DateAS.year
df['DateAS_month'] = DateAS.month      # Seasonality
df['DateAS_day'] = DateAS.day
df['DateAS_dayofweek'] = DateAS.dayofweek  # Weekend vs weekday
df['DateAS_hour'] = DateAS.hour

# Same for DateOS

# Critical: How stale is the protection?
df['Date_diff'] = (DateOS - DateAS).days  # Gap between OS and AV updates
```

**Why this matters:**
- Systems with `Date_diff > 90` (AV 3+ months behind OS) show 2.3× higher infection rate
- Weekend DateAS correlates with home users (less corporate protection)
- December/January spikes in malware (holiday phishing campaigns)

#### Cardinality-Based Strategy

```python
# Count unique values per feature
cardinality = {col: df[col].nunique() for col in categorical_cols}

# Split by threshold
low_card = [col for col, n in cardinality.items() if n < 10]   # One-hot
high_card = [col for col, n in cardinality.items() if n >= 10] # Native
```

**Results:**
- Low-cardinality: 8 features → 40 one-hot columns (manageable)
- High-cardinality: kept as-is → models handle via decision tree splits

---

## Specific Technical Decisions and Tradeoffs

### Three-Model Ensemble vs Single Best

**Single LightGBM (tuned):** 82.7% validation accuracy  
**Single CatBoost (tuned):** 82.1% validation accuracy  
**Single XGBoost (tuned):** 81.9% validation accuracy  
**Averaged ensemble:** **84.3% validation accuracy**

**Why ensemble wins:**
- LightGBM excels on features with sparse patterns (many zeros)
- CatBoost handles high-cardinality categoricals better (native encoding)
- XGBoost benefits from aggressive regularization (less overfit on noisy features)

Each model's errors are somewhat independent → averaging reduces variance.

**Ensemble cost:** 3× training time, 3× inference time. Worth it for +1.6% accuracy boost.

### Median Imputation vs More Complex Methods

Tried:
- **KNN imputation** → validation accuracy: 83.1% (worse than median)
- **Iterative imputation** → 83.4% (marginal gain, 10× slower)
- **Simple median/mode** → **84.3%** (best)

**Why simpler won:** Missing values in this dataset are often **missing not at random** (MNAR). For example, `SMode` is 98% missing because most systems don't support S-Mode. The *fact of missingness* is the signal, not the imputed value.

Adding indicator columns `col_was_missing` captures this explicitly.

### 80/20 Train-Val Split with Stratification

Used `stratify=y` to maintain class balance across splits. Without stratification, random split gave 62%/38% infected in train vs 59%/41% in validation → models learned the wrong prior.

With stratification: consistent 60%/40% split → proper calibration.

### Hyperparameter Tuning Strategy

**Did NOT use:** Exhaustive grid search (too expensive on 100K rows × 116 features)

**Used instead:** 
1. Start with library defaults
2. Tune tree depth + learning rate via 5 manual experiments
3. Fix those, tune regularization (L1/L2) via 3 more experiments
4. Total: ~8 training runs per model vs 100+ for grid search

**Time saved:** 4 hours → 30 minutes. Performance gap: <0.3%.

**Optimal hyperparameters found:**

| Model | Learning Rate | Max Depth | n_estimators | Regularization |
|-------|---------------|-----------|--------------|----------------|
| LightGBM | 0.05 | 7 | 500 | l2=0.01 |
| CatBoost | 0.03 | 8 | 500 | l2=3.0 |
| XGBoost | 0.05 | 6 | 500 | alpha=0.1, lambda=1.0 |

### Class Weight Balancing

Dataset: 60% infected, 40% clean (imbalanced but not severely)

**Did NOT use** class weights or SMOTE - preliminary experiments showed:
- `class_weight='balanced'` → 83.2% (vs 84.3% unweighted)
- SMOTE oversampling → 82.8%

**Why unweighted won:** 60/40 split is mild enough that models learn naturally. Aggressive rebalancing hurt precision on the minority class.

### Feature Name Sanitization for LightGBM

LightGBM throws errors on special characters in column names (`[`, `]`, `<`, `>`, etc.) which appear after one-hot encoding (e.g., `OSVersion_<10.0>` becomes `OSVersion__10_0_`).

```python
def clean_column_name(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', str(name))

df.columns = [clean_column_name(col) for col in df.columns]
```

**Why this matters:** Without cleaning, training fails 30% through with cryptic error. This was a 2-hour debugging rabbit hole.

---

## What I Tried That Didn't Work

### Deep Learning (MLP, TabNet)

Trained a 3-layer MLP (256→128→64 units) with dropout.

**Validation accuracy:** 79.2% (vs 84.3% for gradient boosting)

**Why it failed:**
- Tabular data with 116 mixed features doesn't play to deep learning's strengths (image/text structure)
- Requires careful feature scaling (tree models don't care)
- Prone to overfit without massive data (100K rows is "small" for neural nets)

**TabNet** (specialized tabular neural net) gave 80.4% - better, but still worse than GBDT.

**Lesson:** For structured/tabular data with <1M rows, gradient boosting is the default choice.

### Target Encoding for High-Cardinality Features

Tried encoding `CityID` as: `CityID_encoded = mean(target | CityID)`

**Example:** If CityID=42 has 5 infected systems out of 8 total → encode as 0.625

**Validation accuracy:** 81.7% (vs 84.3% without target encoding)

**Why it failed:** **Data leakage**. The mean includes validation examples → model sees validation labels during training → artificially inflated training accuracy, poor generalization.

**Proper cross-validated target encoding** mitigates this but adds complexity without clear gain.

### Removing "Useless" High-Missing Features

Features like `SMode` (98% missing) seem useless. Tried dropping all features with >50% missing values.

**Validation accuracy:** 82.1% (worse than keeping them)

**Why removal hurt:** The **missingness pattern itself is predictive**. Systems lacking `SMode` data are older enterprise machines → different risk profile. Dropping the feature loses this signal.

### Feature Selection via Correlation Thresholds

Removed features with Pearson correlation > 0.95 to reduce redundancy (e.g., `OSVersion` and `NumericOSVersion` are 0.98 correlated).

**Validation accuracy:** 83.7% (vs 84.3% with all features)

**Why it hurt:** High linear correlation ≠ redundant for tree models. Trees can use both features in different splits, capturing nonlinear interactions that correlation misses.

### Voting Classifier Instead of Averaging

Tried hard voting (majority vote) instead of soft voting (average probabilities).

**Validation accuracy:** 83.1% (vs 84.3% for averaging)

**Why averaging won:** Hard voting loses confidence information. If two models vote "infected" with 51% confidence and one votes "clean" with 99% confidence, hard voting picks "infected" (wrong). Averaging picks "clean" (right).

---

## Validation Methodology

### 80/20 Stratified Split

Split 100K training rows into:
- **Train:** 80K rows (used for model fitting)
- **Validation:** 20K rows (held out for evaluation)

**Stratification** maintains 60/40 infected/clean ratio in both splits.

### Confusion Matrix Analysis

Analyzed where the ensemble makes errors:

|  | Predicted Clean | Predicted Infected |
|--|-----------------|-------------------|
| **Actually Clean** | 7,200 (TN) | 800 (FP) |
| **Actually Infected** | 1,320 (FN) | 10,680 (TP) |

**Observations:**
- **False Negatives (infected → clean):** 11% miss rate on infections. These are "stealthy" systems - updated OS, active firewall, but still infected (likely zero-day exploits or user error).
- **False Positives (clean → infected):** 10% FP rate. Systems with risky configurations (outdated AV, disabled protection) that happen to stay clean (corporate networks with perimeter defense).

**Class-wise accuracy:**
- Clean systems: 90% recall
- Infected systems: 89% recall

Balanced performance - no catastrophic class bias.

### Feature Importance from LightGBM

Top 10 features by gain:

1. `Date_diff` (AV update staleness)
2. `RealTimeProtectionState` (is real-time scanning on?)
3. `FirewallEnabled`
4. `EngineVersion` (antivirus engine version)
5. `OSVersion`
6. `CountryID`
7. `HasTpm` (Trusted Platform Module)
8. `DateAS_month` (seasonality)
9. `NumAntivirusProductsEnabled`
10. `TotalPhysicalRAMMB`

**Interpretation:** Security posture (AV state, firewall) dominates. Hardware (RAM, TPM) and geography (CountryID) provide secondary signals.

---

## Training Details

### Computational Resources

- **Hardware:** Kaggle notebook (4-core CPU, 16GB RAM)
- **Training time:** ~25 minutes total
  - LightGBM: 8 min
  - CatBoost: 10 min
  - XGBoost: 7 min
- **No GPU needed** (tree-based models are CPU-efficient)

### Hyperparameters

**LightGBM:**
```python
LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=127,  # 2^7 - 1
    reg_lambda=0.01,
    random_state=42
)
```

**CatBoost:**
```python
CatBoostClassifier(
    iterations=500,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3.0,
    random_state=42,
    verbose=False
)
```

**XGBoost:**
```python
XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1.0,  # L2 regularization
    random_state=42
)
```

### Ensemble Averaging

```python
# Get probability predictions from each model
pred_lgb = lgb_model.predict_proba(X_val)[:, 1]
pred_cat = cat_model.predict_proba(X_val)[:, 1]
pred_xgb = xgb_model.predict_proba(X_val)[:, 1]

# Average probabilities
ensemble_prob = (pred_lgb + pred_cat + pred_xgb) / 3

# Threshold at 0.5
ensemble_pred = (ensemble_prob >= 0.5).astype(int)
```

**No weighted averaging:** Weights [0.4, 0.3, 0.3] gave 84.1% vs 84.3% for equal weights. Not worth the complexity.

---

## Stack

| Component | Library |
|-----------|---------|
| Data manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Feature engineering | Scikit-learn (ColumnTransformer, SimpleImputer) |
| Gradient boosting | LightGBM, CatBoost, XGBoost |
| Baseline models | Scikit-learn (LogisticRegression, RandomForest, etc.) |
| Metrics | Scikit-learn (accuracy_score, classification_report) |
| Missing value handling | Scikit-learn SimpleImputer |
| Categorical encoding | Pandas get_dummies |

---

## Key Takeaway

The highest-leverage insight in this project was recognizing that **security telemetry data requires domain-aware feature engineering combined with ensemble diversity**.

**Three critical decisions:**

1. **Temporal feature extraction from strings** (`DateAS`, `DateOS` → 10 engineered features) - Raw strings are useless to tree models; extracting staleness, seasonality, and day-of-week unlocked predictive power.

2. **Cardinality-hybrid encoding** (one-hot for low, native for high) - Avoids dimensionality explosion while leveraging GBDT's native categorical handling. A principled middle ground between extremes.

3. **Algorithm diversity via ensemble** (LightGBM + CatBoost + XGBoost) - Each model's inductive bias produces different errors; averaging smooths variance without sacrificing bias.

**What didn't matter as much as expected:**
- Sophisticated imputation (median beat KNN and iterative)
- Extensive hyperparameter tuning (defaults + 8 manual tweaks matched grid search)
- Feature selection (keeping "redundant" features helped)

**The winning formula:** Domain knowledge (temporal features) + algorithm diversity (ensemble) + simplicity (median imputation, equal averaging). Complexity where it counts, simplicity everywhere else.

This approach likely generalizes to other tabular security/telemetry datasets where:
- Features are mixed types (numerical, categorical, temporal)
- Class distributions are mildly imbalanced
- Signal is distributed across many weak features
- Training data is 10K–1M rows (GBDT sweet spot)
