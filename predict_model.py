import joblib
import numpy as np
import pandas as pd
import os
import xgboost as xgb

# ---------------------------------------------------------
# Load model + scaler + feature columns + target encoder
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "thyroid_model_xgb_3772.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler_xgb_3772.pkl")
FEATURE_COLUMNS_PATH = os.path.join(BASE_DIR, "models", "feature_columns_xgb_3772.pkl")
TARGET_ENCODER_PATH = os.path.join(BASE_DIR, "models", "target_encoder_xgb_3772.pkl")
SEX_ENCODER_PATH = os.path.join(BASE_DIR, "models", "sex_encoder.pkl")
REFERRAL_ENCODER_PATH = os.path.join(BASE_DIR, "models", "referral_encoder.pkl")

def load_artifact(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {name} -> {path}")
    return joblib.load(path)

try:
    model = load_artifact(MODEL_PATH, "model")
    scaler = load_artifact(SCALER_PATH, "scaler")
    feature_columns = load_artifact(FEATURE_COLUMNS_PATH, "feature_columns")
    target_encoder = load_artifact(TARGET_ENCODER_PATH, "target_encoder")
    sex_encoder = load_artifact(SEX_ENCODER_PATH, "sex_encoder")
    referral_encoder = load_artifact(REFERRAL_ENCODER_PATH, "referral_encoder")
    print("XGBoost Model + Scaler + Encoders Loaded Successfully!")
except Exception as e:
    raise RuntimeError(f"Error loading model files: {e}")

# ---------------------------------------------------------
# Preprocess Input
# ---------------------------------------------------------

# Define preprocessing constants
BINARY_COLS = [
    'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_meds', 'sick', 'pregnant',
    'thyroid_surgery', 'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid',
    'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
    'TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured',
    'TSH_high', 'TT4_low', 'FTI_low'
]

NUMERIC_COLS = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']

def preprocess_input(input_data):
    df = pd.DataFrame([input_data])

    # Encode categorical variables
    if 'sex' in df.columns:
        df['sex'] = sex_encoder.transform(df['sex'].astype(str))
    if 'referral_source' in df.columns:
        df['referral_source'] = referral_encoder.transform(df['referral_source'].astype(str))

    # Convert binary columns to 0/1
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map({'t': 1, 'f': 0, 'true': 1, 'false': 0}).fillna(0).astype(int)

    # Ensure numeric and ratio columns are float
    for col in NUMERIC_COLS + ['T3_TT4_ratio', 'FTI_TSH_ratio']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Add missing features with 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Select features
    df = df[feature_columns]

    # Scale only numeric columns
    df_scaled = df.copy()
    df_scaled[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

    return df_scaled

# ---------------------------------------------------------
# Predict Function
# ---------------------------------------------------------

def predict_thyroid_condition(input_data):
    try:
        processed = preprocess_input(input_data)

        pred_index = model.predict(processed)[0]
        probas = model.predict_proba(processed)[0]
        confidence = float(np.max(probas))

        predicted_label = target_encoder.inverse_transform([pred_index])[0]

        return {
            "prediction": predicted_label,
            "confidence": round(confidence, 4),
            "raw": probas.tolist()
        }

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# ---------------------------------------------------------
# Debug
# ---------------------------------------------------------

if __name__ == "__main__":
    test = {
        "age": 45,
        "sex": "F",
        "TSH": 1.5,
        "T3": 2.5,
        "TT4": 110,
        "T4U": 1.0,
        "FTI": 100,
        "TBG": 15,
        "referral_source": "other"
    }
    print(predict_thyroid_condition(test))
