# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os


def main():
    print("=" * 50)
    print("Starting Fraud Detection Model Training...")
    print("=" * 50)

    # 1. Load Data
    print("\n[1/6] Loading data...")
    df = pd.read_csv("data/creditcard.csv")
    X = df.drop("Class", axis=1)
    y = df["Class"]
    print(f"Total transactions: {len(df)}")
    print(f"Fraud cases: {y.sum()} ({y.mean()*100:.4f}%)")

    # 2. Train/Test Split
    print("\n[2/6] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 3. Scale Features
    print("\n[3/6] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Apply SMOTE
    print("\n[4/6] Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    print(f"After SMOTE: {len(X_train_res)} samples")
    print(f"Fraud ratio now: {y_train_res.mean()*100:.2f}%")

    # 5. Train XGBoost
    print("\n[5/6] Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
    )
    model.fit(X_train_res, y_train_res)

    # 6. Evaluate & Save
    print("\n[6/6] Evaluating and saving artifacts...")
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    # Save models
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/fraud_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(smote, "models/smote.pkl")

    print("\n✅ Training complete! Model artifacts saved to 'models/' folder.")
    print("=" * 50)


if __name__ == "__main__":
    main()
