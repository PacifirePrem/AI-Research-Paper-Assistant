import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv("data/reditcard.csv")

    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows:\n", df.head())

    #Counting fraud vs normal transcations
    print("\nClass distribution:")
    print(df["Class"].value_counts())

    return df


load_data()


df = pd.read_csv("data/creditcard.csv")

df["Class"].value_counts().plot(kind="bar")

plt.title("Fraud vs Normal Transactions")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()