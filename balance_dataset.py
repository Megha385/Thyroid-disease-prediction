#!/usr/bin/env python3
"""
Script to balance the merged thyroid dataset using SMOTE and safe undersampling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import os

def load_merged_dataset():
    """Load the merged thyroid dataset"""
    print("Loading merged thyroid dataset...")

    # Load the full merged dataset
    df = pd.read_csv('backend/merged_thyroid_dataset.csv')

    # Clean the target column
    df['target'] = df['target'].astype(str).str.lower().str.strip()

    # Drop rows with missing target values
    df = df.dropna(subset=['target'])

    # Quick check
    print(df['target'].value_counts())

    print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    return df

def prepare_data(df):
    """Prepare data for balancing"""
    print("\nPreparing data for balancing...")

    # Separate features and target
    if 'target' not in df.columns:
        raise ValueError("Target column 'target' not found in dataset")

    X = df.drop('target', axis=1)
    y = df['target']

    # Remove rows with NaN targets
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]

    print(f"After removing NaN targets: {len(X)} rows")

    # Handle categorical columns (if any)
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"Encoding categorical columns: {list(categorical_cols)}")
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Handle missing values
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)  # Fill any remaining NaN with 0

    # Encode target labels
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)

    print(f"Features shape: {X.shape}")
    print(f"Original class distribution: {Counter(y)}")

    return X, y_encoded, le_target

def split_dataset(X, y):
    """Split dataset into train/test before balancing"""
    print("\nSplitting dataset (stratified)...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Train class distribution: {Counter(y_train)}")
    print(f"Test class distribution: {Counter(y_test)}")

    return X_train, X_test, y_train, y_test

def apply_smote(X_train, y_train, le_target):
    """Apply SMOTE to balance minority classes"""
    print("\nApplying SMOTE for minority classes...")

    # Get current class counts
    class_counts = Counter(y_train)
    print(f"Before SMOTE: {class_counts}")

    # Decode class labels for strategy
    class_labels = le_target.inverse_transform(list(class_counts.keys()))
    label_to_class = dict(zip(class_counts.keys(), class_labels))

    # Set SMOTE strategy - moderate oversampling
    # Target: hypothyroid ~200, hyperthyroid ~1000, normal keep as is
    sampling_strategy = {
        le_target.transform(['hypothyroid'])[0]: 200,
        le_target.transform(['hyperthyroid'])[0]: 1000,
    }

    # Remove classes that don't exist or are already sufficient
    sampling_strategy = {k: v for k, v in sampling_strategy.items()
                        if k in class_counts and v > class_counts[k]}

    if sampling_strategy:
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: {Counter(y_train_smote)}")
        return X_train_smote, y_train_smote
    else:
        print("No SMOTE needed - classes already balanced")
        return X_train, y_train

def apply_undersampling(X_train, y_train):
    """Optional mild undersampling of majority class"""
    print("\nApplying mild undersampling for majority class...")

    class_counts = Counter(y_train)
    print(f"Before undersampling: {class_counts}")

    # Only undersample if normal class is much larger
    normal_count = class_counts.get('normal', 0)
    max_other = max([count for cls, count in class_counts.items() if cls != 'normal'])

    if normal_count > max_other * 3:  # If normal is 3x larger than biggest minority
        target_normal = max_other * 2  # Reduce to 2x the largest minority
        if target_normal < normal_count:
            rus = RandomUnderSampler(sampling_strategy={'normal': target_normal}, random_state=42)
            X_train_under, y_train_under = rus.fit_resample(X_train, y_train)
            print(f"After undersampling: {Counter(y_train_under)}")
            return X_train_under, y_train_under

    print("No undersampling needed")
    return X_train, y_train

def save_balanced_dataset(X_train_balanced, y_train_balanced, X_test, y_test, le_target):
    """Save the balanced dataset"""
    print("\nSaving balanced dataset...")

    # Combine train and test back
    X_full = pd.concat([X_train_balanced, X_test], ignore_index=True)
    y_full = pd.concat([y_train_balanced, y_test], ignore_index=True)

    # Decode target back to original labels
    y_full_decoded = le_target.inverse_transform(y_full)

    # Create final dataframe
    balanced_df = X_full.copy()
    balanced_df['target'] = y_full_decoded

    # Save to CSV
    output_path = 'backend/balanced_full_thyroid_dataset.csv'
    balanced_df.to_csv(output_path, index=False)

    print(f"Balanced dataset saved to: {output_path}")
    print(f"Final shape: {balanced_df.shape}")

    # Final distribution
    final_dist = Counter(y_full_decoded)
    print(f"Final class distribution: {final_dist}")

    # Percentages
    total = len(y_full_decoded)
    print("\nFinal percentages:")
    for cls, count in final_dist.items():
        pct = (count / total) * 100
        print(f"  {cls}: {count} ({pct:.2f}%)")

    return balanced_df

def train_and_evaluate_model(balanced_df):
    """Train RandomForest model and evaluate performance"""
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST CLASSIFIER")
    print("=" * 60)

    # Split into train/test
    X = balanced_df.drop('target', axis=1)
    y = balanced_df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Train RandomForest with class weights
    clf = RandomForestClassifier(class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return clf, X_test, y_test, y_pred

def main():
    """Main balancing and training procedure"""
    print("=" * 60)
    print("THYROID DATASET BALANCING AND TRAINING PROCEDURE")
    print("=" * 60)

    try:
        # Load data
        df = load_merged_dataset()

        # Prepare data
        X, y_encoded, le_target = prepare_data(df)

        # Split before balancing
        X_train, X_test, y_train, y_test = split_dataset(X, y_encoded)

        # Apply SMOTE
        X_train_balanced, y_train_balanced = apply_smote(X_train, y_train, le_target)

        # Apply undersampling if needed
        X_train_balanced, y_train_balanced = apply_undersampling(X_train_balanced, y_train_balanced)

        # Save balanced dataset
        balanced_df = save_balanced_dataset(X_train_balanced, y_train_balanced, X_test, y_test, le_target)

        # Train and evaluate model
        clf, X_test_final, y_test_final, y_pred = train_and_evaluate_model(balanced_df)

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nSummary:")
        print("1. Dataset balanced using SMOTE")
        print("2. Balanced dataset saved to 'backend/balanced_full_thyroid_dataset.csv'")
        print("3. RandomForest model trained with class_weight='balanced'")
        print("4. Model evaluated with F1-score, precision, and recall")

    except Exception as e:
        print(f"Error during pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()