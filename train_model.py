import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from features import build_features

def prepare_dataset(csv_path: str):
    df = pd.read_csv(csv_path)

    X = []
    y = []

    for _, row in df.iterrows():
        feats = build_features(str(row["resume_text"]), str(row["jd_text"]))
        X.append(feats)
        y.append(int(row["label"]))

    X = pd.DataFrame(X)
    return X, y

if __name__ == "__main__":
    X, y = prepare_dataset("data/labeled_resumes.csv")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y
    )

    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
    }

    best_model = None
    best_name = None
    best_f1 = -1

    print("\n========== Training Models ==========")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"\n--- {name} ---")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\nClassification Report:")
        report = classification_report(y_test, y_pred, output_dict=True)
        print(classification_report(y_test, y_pred))

        f1 = report["weighted avg"]["f1-score"]
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    print("\n========== Best Model ==========")
    print(f"Selected: {best_name}  |  Weighted F1: {round(best_f1, 4)}")

    # Save model + feature column names
    joblib.dump(
        {
            "model": best_model,
            "feature_columns": list(X.columns),
            "model_name": best_name
        },
        "saved_model.pkl"
    )

    print("\n✅ Model saved as: models/saved_model.pkl")
