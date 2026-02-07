import sys
import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

import warnings
# Suppress sklearn/lightgbm feature name warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- CRITICAL PROTOCOL: Path Setup ---
# We must add the local HyperFast repository to the system path to import it 
# without installation, as we are running in a replication environment.
sys.path.insert(0, os.path.join(os.getcwd(), 'HyperFast'))

# TRY IMPORTING HYPERFAST
try:
    # Agent: Check the exact import path in HyperFast/hyperfast/__init__.py
    from hyperfast import HyperFastClassifier
    print("SUCCESS: HyperFast imported successfully.")
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import HyperFast. Check the 'HyperFast' directory structure. {e}")
    sys.exit(1)

# --- CONFIGURATION ---
# Selected datasets for the "Mini-Test Probe"
# Protocol: Small datasets (<1000 samples for mini-test definitions in paper) [cite: 283]
DATASETS_CONFIG = {
    "diabetes": {"id": 37, "target": "class"},               # Binary, Numerical
    "pendigits": {"id": 32, "target": "class"},              # Multiclass, Numerical
    "credit-approval": {"id": 29, "target": "class"},        # Binary, Categorical/Mixed
    "banknote-authentication": {"id": 1462, "target": "Class"} # Binary, Numerical
}

# Protocol: 5-minute time budget for tuning baselines [cite: 288, 668]
BASELINE_TIME_BUDGET_SEC = 5 * 60 

def load_local_data(dataset_name):
    """
    Agent Task: Load and DOWNSAMPLE the dataset to match 'Mini-Test' protocols.
    
    CRITICAL PROTOCOL: 
    - Split 80/20 first.
    - THEN subsample Training set to max 1000 samples.
    - THEN subsample Features to max 100 features.
    """
    print(f"--> Loading {dataset_name}...")
    
    # 1. Load Data
    csv_path = os.path.join('mini_test_datasets', 'openml', f'{dataset_name}.csv')
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return None, None, None, None
    
    df = pd.read_csv(csv_path)
    target_col = DATASETS_CONFIG[dataset_name]['target']
    
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found")
        return None, None, None, None
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 2. Initial Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. CRITICAL: Downsample Training Samples to 1000
    if len(X_train) > 1000:
        print(f"    Downsampling train samples from {len(X_train)} to 1000")
        # Use stratify to maintain class balance even in small subset
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, 
            train_size=1000, 
            random_state=42, 
            stratify=y_train
        )
        
    # 4. CRITICAL: Downsample Features to 100
    if X_train.shape[1] > 100:
        print(f"    Downsampling features from {X_train.shape[1]} to 100")
        # Randomly select 100 feature indices
        np.random.seed(42)
        selected_features = np.random.choice(X_train.columns, 100, replace=False)
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]

    return X_train, X_test, y_train, y_test

def get_preprocessor(X):
    """
    Agent Task: Create a sklearn ColumnTransformer.
    
    CRITICAL PROTOCOL[cite: 280, 281]:
    - Numerical features: Mean imputation -> Standard Scaler.
    - Categorical features: Mode imputation -> One-Hot Encoding.
    - Failure to impute precisely as described invalidates the replication.
    """
    # Identify numerical and categorical features
    # Select numerical columns (float, int)
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    # Select categorical columns (object, bool, category)
    cat_cols = X.select_dtypes(include=['object', 'bool', 'category']).columns
    
    # Define pipelines
    # Numerical: Mean Imputation -> Standard Scaler [cite: 280]
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')), 
        ('scaler', StandardScaler())
    ])
    
    # Categorical: Mode Imputation -> One-Hot Encoding [cite: 281]
    # sparse_output=False is important for dense compatibility if needed by downstream models 
    # (though HF might handle sparse, simple imputer outputs dense)
    # create_categories='warn' is default. handle_unknown='ignore' crucial for unknowns in test
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine into ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    
    return preprocessor

def run_hyperfast(X_train, y_train, X_test):
    """
    Agent Task: Run HyperFast Inference.
    
    CRITICAL PROTOCOL:
    - Use 'Single Forward Pass' mode (no ensembling) for Claim 1 (Speed)[cite: 10].
    - Ensure device is set to 'cpu' if no GPU is available, or 'cuda' otherwise.
    """
    start_time = time.time()
    
    # Identify categorical feature indices for HyperFast
    # HyperFast requires implicit indices for categorical columns if they exist
    cat_features = []
    if hasattr(X_train, 'dtypes'):
        # Get integer indices of columns that are objects/categories
        cat_features = [i for i, dtype in enumerate(X_train.dtypes) if dtype.name in ['object', 'category', 'bool']]

    # Initialize model
    # FORCE CPU: "If running on a MacBook (MPS) or Standard Laptop, force device='cpu'"
    # protocol: Single Forward Pass -> n_ensemble=1, optimization=None
    model = HyperFastClassifier(device='cpu', n_ensemble=1, optimization=None, cat_features=cat_features) 
    
    # fit() in HyperFast uses X_train as the "Support Set" to generate weights
    model.fit(X_train, y_train)
    
    # predict() runs the forward pass on the Query Set (X_test)
    y_pred = model.predict(X_test)
    
    end_time = time.time()
    return y_pred, end_time - start_time

def run_baseline_logreg(X_train, y_train, X_test):
    """
    Agent Task: Run Logistic Regression.
    
    CRITICAL PROTOCOL:
    - Use "Default Configuration" as per paper baselines[cite: 225].
    """
    start_time = time.time()
    
    # Pipeline with preprocessor
    # [cite: 225] Default Config: LogisticRegression(random_state=42)
    # We increase max_iter to ensure convergence on some datasets
    model = Pipeline([
        ('preprocessor', get_preprocessor(X_train)), 
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    end_time = time.time()
    return y_pred, end_time - start_time

def run_baseline_lightgbm(X_train, y_train, X_test):
    """
    Agent Task: Run LightGBM with Time-Constrained Tuning.
    
    CRITICAL PROTOCOL:
    - Paper allows tuning up to a time budget (5 min for this mini-test)[cite: 232].
    - Use RandomizedSearchCV to adhere to the budget.
    """
    start_time = time.time()
    
    # Define search space (Subset of Table 6 in paper) [cite: 593]
    param_dist = {
        'clf__n_estimators': [50, 100, 200],
        'clf__learning_rate': [0.01, 0.1, 0.2],
        'clf__num_leaves': [31, 62, 127]
    }
    
    # Use Pipeline to include preprocessing
    pipeline = Pipeline([
        ('preprocessor', get_preprocessor(X_train)),
        ('clf', LGBMClassifier(random_state=42, verbose=-1))
    ])
    
    # Setup RandomizedSearchCV with n_iter or time constraints
    # Setting n_iter=20 as a conservative guess for 5 minutes
    model = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_dist, 
        n_iter=20, 
        cv=3, 
        scoring='balanced_accuracy', 
        random_state=42,
        n_jobs=1 
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    end_time = time.time()
    return y_pred, end_time - start_time

def evaluate():
    results = []
    
    for name, config in DATASETS_CONFIG.items():
        print(f"\nProcessing {name}...")
        
        # 1. Load Data
        X_train, X_test, y_train, y_test = load_local_data(name)
        if X_train is None:
            continue
        
        # CRITICAL PROTOCOL: Encoding Targets
        # Some datasets have string targets (e.g., "class_a"). Encode them.
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        # 2. Run HyperFast
        print("  Running HyperFast...")
        hf_pred, hf_time = run_hyperfast(X_train, y_train, X_test)
        hf_acc = balanced_accuracy_score(y_test, hf_pred) # [cite: 291]
        print(f"  HyperFast: {hf_acc:.4f} ({hf_time:.2f}s)")

        # 3. Run Logistic Regression
        print("  Running Logistic Regression...")
        lr_pred, lr_time = run_baseline_logreg(X_train, y_train, X_test)
        lr_acc = balanced_accuracy_score(y_test, lr_pred)
        print(f"  Logistic Regression: {lr_acc:.4f} ({lr_time:.2f}s)")

        # 4. Run LightGBM
        print(f"  Running LightGBM (Budget: {BASELINE_TIME_BUDGET_SEC}s)...")
        lgbm_pred, lgbm_time = run_baseline_lightgbm(X_train, y_train, X_test)
        lgbm_acc = balanced_accuracy_score(y_test, lgbm_pred)
        print(f"  LightGBM: {lgbm_acc:.4f} ({lgbm_time:.2f}s)")

        # 5. Store Results
        results.append({
            "Dataset": name,
            "HF Acc": hf_acc, "HF Time": hf_time,
            "LR Acc": lr_acc, "LR Time": lr_time,
            "LGBM Acc": lgbm_acc, "LGBM Time": lgbm_time
        })
        

    # 6. Print Final Report
    if results:
        df = pd.DataFrame(results)
        print("\n=== REPLICATION REPORT: MINI-TEST PROBE ===")
        print(df.to_markdown())
    else:
        print("No results generated.")

if __name__ == "__main__":
    evaluate()