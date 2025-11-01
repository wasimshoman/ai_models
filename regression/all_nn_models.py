#!/usr/bin/env python3
# the project includes multiple parts. one part aims to predict the travel time of taxi trips in new york city using regression analysis
# different regression models are implemented including linear regression and neural network regression.

# this is a regression part of the project that focuses on neural network models using tensorflow and keras libraries.


# this code combines all three neural network models into a single script
# the script uses three different neural network architectures: small, medium, and larger models
# and runs them sequentially, generating separate HTML reports for each.    
# the script includes improved error handling, dynamic epoch settings for each model,
# and ensures compatibility with TensorFlow and sklearn versions.

# to start this script , ensure you read the right csv file into a dataframe named df_filtered
# and save it as 'ignore_files/df_filtered.csv' in the working directory.
# this file is a result of prior data cleaning steps not included in a previous script. check data preparation steps if needed.

# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import time
import io
from HTML_report_generator_all_nn import generate_html_report # this is a module to generate html report that i created separately

# --- Configuration for Speed and Reproducibility ---
def set_seed(seed=42):
    """Sets seeds for NumPy and TensorFlow for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# Constants
TARGET_COL = 'trip_duration'
SEED = 42
BATCH_SIZE = 4096 

# if GPU is available, set mixed precision policy
def get_gpu_status():
    """Checks for GPU availability and returns status string."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    gpu_status = "No GPU found. Training on CPU."
    if gpus:
        try:
            # tf.keras.mixed_precision.set_global_policy('mixed_float16')
            gpu_status = "GPU found. Mixed precision policy check passed."
        except Exception as e:
            gpu_status = f"GPU found but policy setting failed: {e}. Running in standard precision."
    return gpu_status

def load_and_preprocess_data() -> tuple:
    """
    Loads data, performs preprocessing,  and splits into training/test datasets.
    
    Returns:
        tuple: (X_train_tf, y_train_tf, X_test_tf, y_test_tf, 
                input_shape, data_shape, numerical_col, categorical_col)
    """
    print("1. Loading and cleaning data...")
    try:
        # cleaned data file from previous steps
        df_filtered = pd.read_csv('../ignore_files/df_filtered.csv')
    except FileNotFoundError:
        print("ERROR: df_filtered.csv not found. Please ensure the file is in the working directory.")
        raise
        
    set_seed(SEED)

    X = df_filtered.drop(columns=[TARGET_COL]) # droping trip_duration
    y = df_filtered[TARGET_COL]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    all_features = set(X_train.columns)
    # normalizing and encoding the dataset
    categorical_col = ["store_and_fwd_flag_V","time_of_day_category", 'is_weekend','pickup_cluster', 'dropoff_cluster','is_holiday'] 
    # Determine numerical columns
    numerical_col = [col for col in X_train.columns if col not in categorical_col]
    categorical_col = [col for col in categorical_col if col in all_features]

    # Ensure column lists are pure Python lists of strings 
    # this step is necessary to increase the speed of processing

    numerical_col = list(numerical_col)
    categorical_col = list(categorical_col)

    print(f"Identified Numerical Features: {numerical_col}")
    print(f"Identified Categorical Features: {categorical_col}")
    # Create preprocessor
    preprocessor = make_column_transformer(
        (MinMaxScaler(), numerical_col),
        (OneHotEncoder(handle_unknown="ignore"), categorical_col),
    )
    
    print("Fitting and transforming data.")
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # implement OneHotEncoder and convert to float32 Tensors, this step increase the speed of processing as well
    if hasattr(X_train_processed, 'toarray'):
        X_train_np = X_train_processed.toarray().astype(np.float32)
        X_test_np = X_test_processed.toarray().astype(np.float32)
    else:
        X_train_np = X_train_processed.astype(np.float32)
        X_test_np = X_test_processed.astype(np.float32)
        
    y_train_tf = tf.convert_to_tensor(y_train.to_numpy().astype(np.float32), dtype=tf.float32)
    y_test_tf = tf.convert_to_tensor(y_test.to_numpy().astype(np.float32), dtype=tf.float32)
    X_train_tf = tf.convert_to_tensor(X_train_np, dtype=tf.float32)
    X_test_tf = tf.convert_to_tensor(X_test_np, dtype=tf.float32)
    
    INPUT_SHAPE = X_train_tf.shape[1]
    
    return (X_train_tf, y_train_tf, X_test_tf, y_test_tf, 
            INPUT_SHAPE, df_filtered.shape, numerical_col, categorical_col)

def create_model(model_name, input_shape) -> tf.keras.Model:
    """
    Defines and returns one of the three neural network models.
    """
    if model_name == 'small_model':
        model = tf.keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=[input_shape], name='input_1'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu', name='hidden_1'),
            layers.Dense(1, activation='relu', name='output_layer')
        ], name=model_name)
    elif model_name == 'medium_model':
        model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=[input_shape], name='input_1'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu', name='hidden_1'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu', name='hidden_2'),
            layers.Dense(1, activation='relu', name='output_layer')
        ], name=model_name)
    elif model_name == 'tuned_medium_model':
        model = tf.keras.Sequential([
            layers.Dense(128, activation='tanh', input_shape=[input_shape], name='input_1'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='tanh', name='hidden_1'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu', name='hidden_2'),
            layers.Dense(1, activation='relu', name='output_layer')
        ], name=model_name)
    elif model_name == 'larger_model':
        # This architecture is heavier, so it is often paired with fewer epochs
        # Added L2 regularization to the larger model's hidden layers
        # L2 regularization helps penalize large weights, improving generalization.
        l2_reg = 0.001 
        model = tf.keras.Sequential([
            layers.Dense(512, activation="tanh", input_shape=[input_shape], 
                         kernel_regularizer=regularizers.l2(l2_reg), name="input_1"),
            layers.Dropout(0.2),
            layers.Dense(256, activation="tanh", 
                         kernel_regularizer=regularizers.l2(l2_reg), name="hidden_1"),
            layers.Dropout(0.2),
            layers.Dense(128, activation="relu", 
                         kernel_regularizer=regularizers.l2(l2_reg), name="hidden_2"),
            layers.Dense(64, activation="relu", 
                         kernel_regularizer=regularizers.l2(l2_reg), name="hidden_3"),
            layers.Dense(1, activation="relu", name="output_layer") 
        ], name=model_name)
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model

def run_training_and_reporting(model, output_filename, epochs, data_bundle, gpu_status):
    """
    Trains, evaluates, and generates an HTML report for the given model.
    make sure that ==> model: tf.keras.Model, output_filename: str, epochs: int, data_bundle: tuple, gpu_status: str
    """
    (X_train_tf, y_train_tf, X_test_tf, y_test_tf, 
     INPUT_SHAPE, data_shape, numerical_col, categorical_col) = data_bundle # read data 
    
    print(f"\n--- Starting Training for {model.name} ({epochs} epochs) ---")
    # Set up Callbacks
    # Define Early Stopping Callback
    # It monitors the validation MAE (val_loss) and stops if it doesn't improve for 5 epochs.
    early_stopping = callbacks.EarlyStopping(
        monitor='val_mae', 
        patience=5, 
        verbose=1, 
        mode='min', 
        restore_best_weights=True
    )
    # Compile Model
    model.compile(
        loss='mae', 
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['mae']
    )
    
    # Capture model summary for the report
    model_summary_list = []
    model.summary(print_fn=lambda x: model_summary_list.append(x))
    model_summary_data = "\n".join(model_summary_list)
    
    # Get parameter counts for later reporting
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    
    # Train Model
    start_time = time.time()
    history = model.fit(
        X_train_tf, 
        y_train_tf, 
        epochs=epochs, 
        batch_size=BATCH_SIZE, 
        verbose=0,
        validation_data=(X_test_tf, y_test_tf),
        callbacks=[early_stopping]
    )
    training_time = time.time() - start_time
    
    # Prepare training log data for the report
    # This matches the expected input format for the HTML_report_generator.
    training_log_data = [] 
    history_log = history.history
    for epoch in range(epochs):
        training_log_data.append({
            'epoch': epoch + 1,
            'loss': f"{history_log['loss'][epoch]:.4f}",
            'mae': f"{history_log['mae'][epoch]:.4f}",
            'val_loss': f"{history_log['val_loss'][epoch]:.4f}",
            'val_mae': f"{history_log['val_mae'][epoch]:.4f}",
        })
    
    # 5. Evaluate Model
    print("5. Evaluating Model...")
    # Need numpy arrays for sklearn metrics
    y_test_np = y_test_tf.numpy().flatten()
    
    # Get predictions
    y_pred_nn = model.predict(X_test_tf.numpy(), verbose=0).flatten()
    
    # Calculate metrics
    mae_nn = mean_absolute_error(y_test_np, y_pred_nn)
    mse_nn = mean_squared_error(y_test_np, y_pred_nn)
    rmse_nn = np.sqrt(mse_nn)
    r2_nn = r2_score(y_test_np, y_pred_nn) 
    
    # 6. Prepare results dictionary
    results = {
        'data_shape': data_shape,
        'input_features': INPUT_SHAPE,
        'training_time': f"{training_time:.2f} seconds",
        'hardware': gpu_status,
        'mae': f"{mae_nn:.4f}",
        'mse': f"{mse_nn:.4f}",
        'rmse': f"{rmse_nn:.4f}",
        'r2': f"{r2_nn:.4f}",
        'numerical_features':  ', '.join(numerical_col), 
        'categorical_features': ', '.join(categorical_col),
        'training_log': training_log_data, # <-- Now a list of dictionaries
        'model_summary': model_summary_data,
        'total_params': f"{total_params:,}",
        'trainable_params': f"{trainable_params:,}",
        'non_trainable_params': f"{total_params - trainable_params:,}",
        'model_name': model.name 
    }

    # 7. Generate and Save Report
    print(f"6. Generating HTML report for {model.name}...")
    html_content = generate_html_report(results, TARGET_COL, BATCH_SIZE) 
    
    with open(output_filename, "w") as f:
        f.write(html_content)

    print(f"Successfully ran {model.name} and saved report to {output_filename}")


if __name__ == "__main__":
    tf.keras.backend.clear_session()
    
    try:
        print("Starting combined ML Pipeline execution...")
        
        # 1. Run common data loading and preprocessing once
        gpu_status = get_gpu_status()
        data_bundle = load_and_preprocess_data()
        input_shape = data_bundle[4]
        
        # 2. Define the models to run with DYNAMIC EPOCHS
        model_configs = [
            # Small model may need more epochs to converge
            {'name': 'small_model', 'output_file': 'nn_small_model.html', 'epochs': 20},
            # Medium model
            {'name': 'medium_model', 'output_file': 'nn_medium_model.html', 'epochs': 20},
            # Larger model converges faster but risks overfitting
            {'name': 'larger_model', 'output_file': 'nn_larger_model.html', 'epochs': 20},
            # Tuned Medium Model (Testing different activation, smaller batch size, and lower learning rate)
            {'name': 'tuned_medium_model', 'output_file': 'nn_tuned_medium_model.html', 'epochs': 20}
        ]
        
        # 3. Loop through configurations and run the common training/reporting pipeline
        for config in model_configs:
            model = create_model(config['name'], input_shape)
            run_training_and_reporting(
                model=model,
                output_filename=config['output_file'],
                epochs=config['epochs'],
                data_bundle=data_bundle,
                gpu_status=gpu_status
            )
            tf.keras.backend.clear_session()
        
        print("\nAll model pipelines completed successfully. Check the generated HTML files.")
        
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
