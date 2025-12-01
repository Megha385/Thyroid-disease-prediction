from flask import Flask, send_file, send_from_directory, redirect, url_for, render_template, request, jsonify
import os
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__, static_folder='../frontend/static', template_folder='../frontend/templates')

@app.route("/")
def index():
    return render_template('welcome.html')

@app.route("/dashboard")
def dashboard():
    return render_template('dashboard.html')

@app.route("/results")
def results():
    return render_template('results.html')

@app.route("/check_health")
def check_health():
    return render_template('check_health.html')

@app.route("/learn")
def learn_page():
    return render_template('learn.html')

@app.route("/nutrition")
def nutrition_page():
    return render_template('nutrition.html')

@app.route("/exercise")
def exercise_page():
    return render_template('exercise.html')

@app.route("/tips")
def tips_page():
    return render_template('tips.html')

# Serve static files (css/js/images) from ./static automatically
@app.route('/static/<path:filename>')
def static_files(filename):
    static_dir = os.path.join(app.root_path, '../frontend/static')
    if os.path.exists(os.path.join(static_dir, filename)):
        return send_from_directory(static_dir, filename)
    return "", 404


@app.route('/api/predict_medical_test', methods=['POST'])
def predict_medical_test():
    """Accepts JSON with lab-only fields (tsh, t3, tt4, t4u, fti, tbg, sex, age optional)
    and returns a model prediction (hypothyroid/hyperthyroid/normal) with confidence.
    The endpoint will try to map provided keys to the model's feature columns and
    fill missing features with zeros before prediction."""
    try:
        data = request.get_json() or {}

        # Load model artifacts (cached on first load)
        models_dir = os.path.join(app.root_path, 'models')
        model_path = os.path.join(models_dir, 'thyroid_model.pkl')
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        cols_path = os.path.join(models_dir, 'feature_columns.pkl')
        encoder_path = os.path.join(models_dir, 'target_encoder.pkl')

        if not os.path.exists(model_path):
            return jsonify({'error': 'Model file not found on server.'}), 500

        model = joblib.load(model_path)
        feature_columns = []
        if os.path.exists(cols_path):
            feature_columns = joblib.load(cols_path)

        scaler = None
        if os.path.exists(scaler_path):
            try:
                scaler = joblib.load(scaler_path)
            except Exception:
                scaler = None

        target_encoder = None
        if os.path.exists(encoder_path):
            try:
                target_encoder = joblib.load(encoder_path)
            except Exception:
                target_encoder = None

        # Build feature vector
        if not feature_columns:
            # Fallback: try to infer columns from model if possible
            feature_columns = []

        # Initialize input row with zeros
        row = {c: 0 for c in feature_columns}

        # Candidate mapping of input names -> feature column name patterns
        mapping = {
            'tsh': ['tsh', 'TSH', 'TSH_Value', 'tsh_level'],
            't3': ['t3', 'T3', 't3_level'],
            'tt4': ['tt4', 'TT4', 'total_t4', 'tt4_level'],
            't4u': ['t4u', 'T4U', 't4_uptake', 't4uptake'],
            'fti': ['fti', 'FTI', 'free_thyroxine_index'],
            'tbg': ['tbg', 'TBG']
        }

        # Helper to find matching column name in feature_columns
        def find_col(candidates):
            for cand in candidates:
                # try exact match
                if cand in feature_columns:
                    return cand
                # try lower-case match
                low = cand.lower()
                for fc in feature_columns:
                    if fc.lower() == low:
                        return fc
            return None

        # Map provided values into row
        for key, cand_list in mapping.items():
            if key in data and data[key] is not None and data[key] != '':
                col = find_col(cand_list)
                if col:
                    try:
                        row[col] = float(data[key])
                    except Exception:
                        row[col] = data[key]

        # Optionally map sex/age/referral if present
        if 'sex' in data:
            col = find_col(['sex', 'Sex'])
            if col:
                row[col] = data['sex']
        if 'age' in data:
            col = find_col(['age', 'Age'])
            if col:
                try:
                    row[col] = float(data['age'])
                except Exception:
                    row[col] = data['age']

        # Create DataFrame
        if feature_columns:
            X = pd.DataFrame([row], columns=feature_columns)
        else:
            # If no feature columns, try simple vector of provided numeric values
            X = pd.DataFrame([data])

        # If scaler exists and shapes match, apply
        try:
            if scaler is not None:
                X_scaled = scaler.transform(X.select_dtypes(include=[np.number]))
                # replace numeric columns with scaled values
                num_cols = X.select_dtypes(include=[np.number]).columns
                X[num_cols] = X_scaled
        except Exception:
            # ignore scaling errors
            pass

        # Predict
        try:
            preds = model.predict(X)
            pred = preds[0]
            prob = None
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)
                # take max probability
                prob = float(np.max(probs, axis=1)[0])
        except Exception as e:
            return jsonify({'error': 'Model prediction failed', 'details': str(e)}), 500

        label = str(pred)
        if target_encoder is not None:
            try:
                # If encoder is a LabelEncoder or similar
                label = target_encoder.inverse_transform([pred])[0]
            except Exception:
                pass

        # Compose response
        response = {
            'label': label,
            'probability': round(float(prob * 100) if prob is not None else 0.0, 2),
            'used_values': {k: data.get(k) for k in mapping.keys()}
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': 'Server error while processing prediction', 'details': str(e)}), 500

if __name__ == "__main__":
    # Use debug=True while developing. Remove in production.
    app.run(debug=True)
