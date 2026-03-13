import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE


def train_model():

    print("Loading dataset...")

    # Load dataset
    df = pd.read_csv("data/creditcard.csv")

    # Separate features and labels
    X = df.drop("Class", axis=1)
    y = df["Class"]

    print("Splitting dataset...")

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Scaling features...")

    # Feature scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Applying SMOTE to balance data...")

    # Handle imbalance
    smote = SMOTE(random_state=42)

    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train,
        y_train
    )

    print("Training Random Forest model...")

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train_resampled, y_train_resampled)

    print("Making predictions...")

    # Predictions
    predictions = model.predict(X_test)

    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, predictions))

    print("\nSaving trained model...")

    # Save model
    joblib.dump(model, "model/fraud_model.pkl")

    print("Model saved successfully!")


# Run training
train_model()