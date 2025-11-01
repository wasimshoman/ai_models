from matplotlib.pylab import seed
import pandas as pd
import numpy as np
import time
import textwrap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import minmax_scale
import os
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import tensorflow as tf
TARGET_COL = 'trip_duration'
SEED = 42

# --- Configuration for Speed and Reproducibility ---

def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def run_linear_regression():
    set_seed(SEED)

    # --- Load data ---
    try:
        df_filtered = pd.read_csv('../ignore_files/df_filtered.csv')
        print(f"✅ Data loaded successfully. Shape: {df_filtered.shape}")
    except FileNotFoundError:
        raise FileNotFoundError("df_filtered.csv not found in ../ignore_files/. Please verify path.")

    X = df_filtered.drop(columns=[TARGET_COL])
    y = df_filtered[TARGET_COL]

    # --- Define columns ---
    categorical_cols = [
        "store_and_fwd_flag_V", "time_of_day_category", "is_weekend",
        "pickup_cluster", "dropoff_cluster", "is_holiday"
    ]
    categorical_cols = [c for c in categorical_cols if c in X.columns]
    numerical_cols = [c for c in X.columns if c not in categorical_cols]

    # --- Preprocessing ---
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop"
    )

    # --- Split before fitting ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    # --- Fit preprocessor on training data ---
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # --- Get transformed feature names ---
    feature_names = numerical_cols
    if categorical_cols:
        cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols)
        feature_names = numerical_cols + list(cat_features)

    # --- Train model ---
    start_time = time.time()
    model = LinearRegression().fit(X_train, y_train)
    training_time = time.time() - start_time

    # --- Predictions and metrics ---
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # --- Results dictionary (same as NN) ---
    results = {
        'model_name': 'Linear Regression (Sklearn)',
        'data_shape': df_filtered.shape,
        'input_features': len(feature_names),
        'numerical_features': ', '.join(numerical_cols),
        'categorical_features': ', '.join(categorical_cols),
        'training_time': f"{training_time:.2f} seconds",
        'hardware': 'CPU',
        'mae': f"{mae:.4f}",
        'mse': f"{mse:.4f}",
        'rmse': f"{rmse:.4f}",
        'r2': f"{r2:.4f}",
        'total_params': f"{len(feature_names):,}",
        'trainable_params': f"{len(feature_names):,}",
        'non_trainable_params': "0",
        'model_summary': "Linear Regression model with normalized + one-hot encoded input.",
        'training_log': [],
    }

    return results


def generate_html_report(results):
    """Match the NN HTML report structure and style."""
    html_template = textwrap.dedent(f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{results['model_name']} Report</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="p-8 bg-gray-50">
        <div class="max-w-5xl mx-auto bg-white p-6 rounded-2xl shadow">
            <h1 class="text-3xl font-bold text-gray-800 mb-4 border-b-4 border-indigo-500 pb-2">
                {results['model_name']} – Trip Duration Prediction
            </h1>

            <!-- MODEL PERFORMANCE -->
            <div class="mb-8">
                <h2 class="text-2xl font-semibold text-indigo-700 mb-4">Model Performance Metrics</h2>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                    <div class="p-4 bg-green-100 rounded-lg">
                        <p class="text-xs font-medium text-green-800 uppercase">R² Score</p>
                        <p class="text-3xl font-extrabold text-green-700 mt-1">{results['r2']}</p>
                    </div>
                    <div class="p-4 bg-blue-100 rounded-lg">
                        <p class="text-xs font-medium text-blue-800 uppercase">MAE</p>
                        <p class="text-3xl font-extrabold text-blue-700 mt-1">{results['mae']}</p>
                    </div>
                    <div class="p-4 bg-purple-100 rounded-lg">
                        <p class="text-xs font-medium text-purple-800 uppercase">RMSE</p>
                        <p class="text-3xl font-extrabold text-purple-700 mt-1">{results['rmse']}</p>
                    </div>
                    <div class="p-4 bg-red-100 rounded-lg">
                        <p class="text-xs font-medium text-red-800 uppercase">MSE</p>
                        <p class="text-3xl font-extrabold text-red-700 mt-1">{results['mse']}</p>
                    </div>
                </div>
                <div class="mt-4 p-3 bg-gray-100 rounded-lg text-sm">
                    <span class="font-semibold text-gray-700">Training Duration:</span> 
                    <span class="font-mono text-base text-gray-800">{results['training_time']}</span>
                </div>
            </div>

            <!-- MODEL CONFIGURATION -->
            <div class="mb-8">
                <h2 class="text-2xl font-semibold text-indigo-700 mb-3">Model Configuration</h2>
                <ul class="text-gray-700 list-disc pl-6">
                    <li><strong>Total Parameters:</strong> {results['total_params']}</li>
                    <li><strong>Trainable Parameters:</strong> {results['trainable_params']}</li>
                    <li><strong>Hardware:</strong> {results['hardware']}</li>
                    <li><strong>Input Features:</strong> {results['input_features']}</li>
                    <li><strong>Numerical Features:</strong> {results['numerical_features']}</li>
                    <li><strong>Categorical Features:</strong> {results['categorical_features']}</li>
                </ul>
            </div>

            <!-- MODEL SUMMARY -->
            <div class="p-4 bg-gray-100 rounded-lg">
                <h2 class="text-xl font-semibold text-indigo-700 mb-2">Model Summary</h2>
                <pre class="text-sm font-mono text-gray-800 overflow-x-auto">{results['model_summary']}</pre>
            </div>
        </div>
    </body>
    </html>
    """)
    return html_template


if __name__ == "__main__":
    results = run_linear_regression()
    html_content = generate_html_report(results)

    with open("linear_regression_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    print("✅ Linear Regression report saved as 'linear_regression_report.html'")