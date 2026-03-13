import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data():

    # Load dataset
    df = pd.read_csv("data/creditcard.csv")

    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Training set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)

    return X_train, X_test, y_train, y_test


# Run preprocessing
preprocess_data()