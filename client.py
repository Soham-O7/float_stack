import requests
import pandas as pd
import numpy as np
#from datadiet import score_dataset, health_report, train_and_measure

def process_uploaded_file(uploaded_file, label_col, keep_frac):
    # Read the uploaded CSV file using pandas
    df = pd.read_csv(uploaded_file)
    
    # Separate features (X) and labels (y)
    X = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).values
    y = df[label_col].values
    
    # 1. Score the dataset for duplicates and noise
    # scores = score_dataset(X, y)
    
    # 2. Get the health report metrics
    # health = health_report(scores)
    
    # 3. Run the proof comparison (Full vs Random vs DataDiet)
    results_table, best_method, method_scores, dd_idx = train_and_measure(
        X_train=X, y_train=y, X_test=X, y_test=y, keep_frac=keep_frac
    )
    
    # 4. Prepare the pruned dataset for download
    X_pruned = X[dd_idx]
    y_pruned = y[dd_idx]
    pruned_df = pd.DataFrame(X_pruned)
    pruned_df[label_col] = y_pruned
    
    return health, results_table, best_method, pruned_df

def fetch_backend_stub():
    """Fetches data directly from your teammate's FastAPI stub endpoint."""
    url = "http://localhost:8000/stub/result"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Backend stub offline: {e}")
    return None