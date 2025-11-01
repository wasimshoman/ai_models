#!/usr/bin/env python3
"""
Revised NN training script for NYC taxi trip-duration regression.

Notes:
- Expects cleaned dataframe at: ../ignore_files/df_filtered.csv
- Produces:
    - HTML reports via your existing generate_html_report(...)
    - Saved preprocessor: outputs/preprocessor.pkl
    - Saved models: models/<model_name>/
    - Metrics JSON: outputs/<model_name>_metrics.json
"""

import os
import time
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks, optimizers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

# Custom HTML report generator (user-provided)
from HTML_report_generator_all_nn import generate_html_report

# -------------------------
# Configuration & constants
# -------------------------
TARGET_COL = 'trip_duration'
SEED = 42
BATCH_SIZE = 4096
OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")
PREPROCESSOR_PATH = OUTPUT_DIR / "preprocessor.pkl"

# Ensure folders exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------
# Reproducibility helper
# -------------------------
def set_seed(seed=SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# GPU status check (returns string for report)
def get_gpu_status():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return "No GPU found. Training on CPU."
    else:
        # Mixed precision can be enabled if desired and compatible
        # try:
        #     tf.keras.mixed_precision.set_global_policy('mixed_float16')
        #     return "GPU found. Mixed precision enabled."
        # except Exception:
        #     return "GPU found. Mixed precision not enabled due to compatibility."
        return f"GPU found ({len(gpus)} device(s))."

# -------------------------
# Data loading & preprocessing
# -------------------------
def load_and_preprocess_data(csv_path: str = "../ignore_files/df_filtered.csv", test_size=0.2, random_state=SEED):
    """
    Loads CSV, preprocesses, and returns TF tensors and metadata.
    Also saves the preprocessor to disk (joblib).
    """
    logger.info("Loading data from %s", csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info("Data loaded. Shape: %s", df.shape)

    set_seed(SEED)

    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in dataframe columns.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Candidate categorical columns - ensure they exist in the data
    candidate_categorical = [
        "store_and_fwd_flag_V", "time_of_day_category", "is_weekend",
        "pickup_cluster", "dropoff_cluster", "is_holiday"
    ]
    categorical_cols = [c for c in candidate_categorical if c in X_train.columns]
    numerical_cols = [c for c in X_train.columns if c not in categorical_cols]

    logger.info("Identified %d numerical and %d categorical features.", len(numerical_cols), len(categorical_cols))
    logger.debug("Numerical cols: %s", numerical_cols)
    logger.debug("Categorical cols: %s", categorical_cols)

    # Column transformer with MinMax for numerics and OneHot for categoricals (ignore unknowns)
    preprocessor = make_column_transformer(
        (MinMaxScaler(), numerical_cols),
        (OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        remainder='drop'
    )

    logger.info("Fitting preprocessor...")
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Ensure data is float32 numpy arrays
    X_train_np = np.asarray(X_train_proc, dtype=np.float32)
    X_test_np = np.asarray(X_test_proc, dtype=np.float32)
    y_train_np = np.asarray(y_train.to_numpy(), dtype=np.float32).reshape(-1, 1)
    y_test_np = np.asarray(y_test.to_numpy(), dtype=np.float32).reshape(-1, 1)

    # Save preprocessor for inference later
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    logger.info("Saved preprocessor to %s", PREPROCESSOR_PATH)

    # Convert to TF tensors
    X_train_tf = tf.convert_to_tensor(X_train_np, dtype=tf.float32)
    X_test_tf = tf.convert_to_tensor(X_test_np, dtype=tf.float32)
    y_train_tf = tf.convert_to_tensor(y_train_np, dtype=tf.float32)
    y_test_tf = tf.convert_to_tensor(y_test_np, dtype=tf.float32)

    input_shape = X_train_tf.shape[1]
    return {
        "X_train_tf": X_train_tf,
        "X_test_tf": X_test_tf,
        "y_train_tf": y_train_tf,
        "y_test_tf": y_test_tf,
        "input_shape": input_shape,
        "data_shape": df.shape,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "preprocessor": preprocessor,
        "raw_df": df
    }

# -------------------------
# Generic model builder
# -------------------------
def build_dense_model(input_shape: int, layer_sizes: list, activation='relu',
                      output_activation='linear', l2_reg=0.0, dropout_rate=0.0,
                      batchnorm=False, name="model"):
    """
    Generic dense feedforward network builder.

    Args:
        input_shape: number of input features
        layer_sizes: list of ints - hidden layer sizes (e.g., [128, 64, 32])
        activation: activation for hidden layers ('relu', 'tanh', etc.)
        output_activation: usually 'linear' for regression
        l2_reg: L2 regularization coefficient (float)
        dropout_rate: dropout fraction between layers (0.0 means no dropout)
        batchnorm: whether to add BatchNormalization after dense
        name: model name
    Returns:
        tf.keras.Model
    """
    reg = regularizers.l2(l2_reg) if l2_reg and l2_reg > 0 else None
    inputs = layers.Input(shape=(input_shape,), name="input")
    x = inputs

    for i, size in enumerate(layer_sizes):
        x = layers.Dense(size, activation=None, kernel_regularizer=reg, name=f"dense_{i+1}")(x)
        if batchnorm:
            x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.Activation(activation, name=f"act_{i+1}")(x)
        if dropout_rate and dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f"dropout_{i+1}")(x)

    outputs = layers.Dense(1, activation=output_activation, name="output")(x)
    model = models.Model(inputs=inputs, outputs=outputs, name=name)
    return model

# -------------------------
# Training & reporting
# -------------------------
def run_training_and_reporting(model, config, data_bundle, gpu_status):
    """
    Trains model, evaluates metrics, saves artifacts and generates an HTML report.

    Args:
        model: compiled tf.keras.Model
        config: dict with keys 'name', 'epochs', 'output_file'
        data_bundle: dict returned by load_and_preprocess_data
    """
    name = model.name
    output_file = OUTPUT_DIR / config.get("output_file", f"{name}.html")
    epochs = int(config.get("epochs", 30))

    logger.info("Starting training for model %s (epochs=%d)", name, epochs)

    X_train_tf = data_bundle["X_train_tf"]
    y_train_tf = data_bundle["y_train_tf"]
    X_test_tf = data_bundle["X_test_tf"]
    y_test_tf = data_bundle["y_test_tf"]

    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=config.get("early_stopping_patience", 8),
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=1e-7
    )

    history = None
    try:
        start_time = time.time()
        history = model.fit(
            X_train_tf,
            y_train_tf,
            validation_data=(X_test_tf, y_test_tf),
            epochs=epochs,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        training_time = time.time() - start_time
        logger.info("Training completed in %.2f seconds", training_time)
    except Exception as e:
        logger.exception("Training failed for model %s: %s", name, e)
        raise

    # Evaluation
    logger.info("Evaluating model on test set...")
    y_test_np = y_test_tf.numpy().flatten()
    y_pred = model.predict(X_test_tf.numpy(), verbose=0).flatten()

    mae_val = mean_absolute_error(y_test_np, y_pred)
    mse_val = mean_squared_error(y_test_np, y_pred)
    rmse_val = np.sqrt(mse_val)
    r2_val = r2_score(y_test_np, y_pred)
    try:
        mape_val = mean_absolute_percentage_error(y_test_np, y_pred)
    except Exception:
        mape_val = None

    # Model summaries & params
    model_summary_list = []
    model.summary(print_fn=lambda x: model_summary_list.append(x))
    model_summary_str = "\n".join(model_summary_list)
    total_params = model.count_params()
    trainable_params = np.sum([np.prod(w.shape) for w in model.trainable_weights])
    non_trainable_params = total_params - int(trainable_params)

    # Save model
    model_output_path = MODEL_DIR / name
    model_output_path.mkdir(parents=True, exist_ok=True)
    try:
        # Save full model (architecture + weights + optimizer state)
        model.save(model_output_path, save_format="tf")
        logger.info("Saved model to %s", model_output_path)
    except Exception as e:
        logger.warning("Failed to save full model for %s: %s. Attempting to save weights only.", name, e)
        model.save_weights(model_output_path / "model.weights.h5")

    # Save metrics to JSON
    metrics = {
        "model_name": name,
        "data_shape": data_bundle.get("data_shape"),
        "input_features": data_bundle.get("input_shape"),
        "training_time_seconds": float(training_time),
        "hardware": gpu_status,
        "mae": float(mae_val),
        "mse": float(mse_val),
        "rmse": float(rmse_val),
        "r2": float(r2_val),
        "mape": float(mape_val) if mape_val is not None else None,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "non_trainable_params": int(non_trainable_params),
        "numerical_features": data_bundle.get("numerical_cols"),
        "categorical_features": data_bundle.get("categorical_cols"),
        "history": {k: [float(x) for x in v] for k, v in (history.history.items() if history is not None else [])},
    }

    metrics_path = OUTPUT_DIR / f"{name}_metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Saved metrics JSON to %s", metrics_path)

    # Prepare results for HTML generator (keep your format)
    results = {
        'data_shape': data_bundle.get("data_shape"),
        'input_features': data_bundle.get("input_shape"),
        'training_time': f"{training_time:.2f} seconds",
        'hardware': gpu_status,
        'mae': f"{mae_val:.4f}",
        'mse': f"{mse_val:.4f}",
        'rmse': f"{rmse_val:.4f}",
        'r2': f"{r2_val:.4f}",
        'numerical_features': ', '.join(data_bundle.get("numerical_cols") or []),
        'categorical_features': ', '.join(data_bundle.get("categorical_cols") or []),
        'training_log': [
            {
                'epoch': i + 1,
                'loss': f"{history.history['loss'][i]:.6f}",
                'mae': f"{history.history['mae'][i]:.6f}" if 'mae' in history.history else None,
                'val_loss': f"{history.history['val_loss'][i]:.6f}",
                'val_mae': f"{history.history['val_mae'][i]:.6f}" if 'val_mae' in history.history else None
            }
            for i in range(len(history.history['loss']))
        ] if history is not None else [],
        'model_summary': model_summary_str,
        'total_params': f"{total_params:,}",
        'trainable_params': f"{int(trainable_params):,}",
        'non_trainable_params': f"{int(non_trainable_params):,}",
        'model_name': name
    }

    # --------------------------------------------------
    #  Generate and Save HTML report (same as old script)
    # --------------------------------------------------
    html_filename = config.get("output_file", f"{name}.html")
    html_output_path = Path(html_filename)  # Save directly in working dir like before

    logger.info("Generating HTML report: %s", html_output_path)
    html_content = generate_html_report(results, TARGET_COL, BATCH_SIZE)

    with open(html_output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info("Saved HTML report to %s", html_output_path)



    logger.info("Model %s training+reporting complete.", name)
    return metrics_path, model_output_path

# -------------------------
# Main entry
# -------------------------
if __name__ == "__main__":
    tf.keras.backend.clear_session()
    try:
        logger.info("Starting pipeline execution...")
        gpu_status = get_gpu_status()
        data = load_and_preprocess_data()

        input_shape = data["input_shape"]

        # Define model configurations and hyperparams
        model_configs = [
            {
                "name": "small_model",
                "layer_sizes": [32, 16],
                "activation": "relu",
                "dropout": 0.0,
                "batchnorm": False,
                "l2": 0.0,
                "epochs": 30,
                "output_file": "nn_small_model.html"
            },
            {
                "name": "medium_model",
                "layer_sizes": [128, 64, 32],
                "activation": "relu",
                "dropout": 0.1,
                "batchnorm": True,
                "l2": 1e-4,
                "epochs": 20,
                "output_file": "nn_medium_model.html"
            },
            {
                "name": "tuned_medium_model",
                "layer_sizes": [128, 64, 32],
                "activation": "tanh",
                "dropout": 0.15,
                "batchnorm": True,
                "l2": 5e-4,
                "epochs": 20,
                "output_file": "nn_tuned_medium_model.html"
            },
            {
                "name": "larger_model",
                "layer_sizes": [512, 256, 128, 64],
                "activation": "relu",
                "dropout": 0.2,
                "batchnorm": True,
                "l2": 1e-3,
                "epochs": 20,
                "output_file": "nn_larger_model.html"
            }
        ]

        for cfg in model_configs:
            model_name = cfg["name"]
            # Build model
            model = build_dense_model(
                input_shape=input_shape,
                layer_sizes=cfg["layer_sizes"],
                activation=cfg["activation"],
                output_activation="linear",        # IMPORTANT: linear output for regression
                l2_reg=cfg.get("l2", 0.0),
                dropout_rate=cfg.get("dropout", 0.0),
                batchnorm=cfg.get("batchnorm", False),
                name=model_name
            )

            # Compile - use MSE loss (better for gradient stability) and monitor MAE
            model.compile(
                loss='mse',
                optimizer=optimizers.Adam(learning_rate=1e-3),
                metrics=['mae', 'mse']
            )

            # Train + report
            config_run = {
                "name": model_name,
                "epochs": cfg.get("epochs", 30),
                "output_file": cfg.get("output_file", f"{model_name}.html"),
                "early_stopping_patience": 8
            }
            try:
                metrics_path, saved_model_path = run_training_and_reporting(model, config_run, data, gpu_status)
                logger.info("Finished run for %s; metrics: %s; model saved at: %s", model_name, metrics_path, saved_model_path)
            except Exception as e:
                logger.exception("Run failed for model %s: %s", model_name, e)

            # Clear session to free memory/resources before next model
            tf.keras.backend.clear_session()

        logger.info("All model runs finished. Check outputs in %s and %s", OUTPUT_DIR, MODEL_DIR)

    except Exception as e_main:
        logger.exception("Pipeline failed: %s", e_main)
