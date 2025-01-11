import pandas as pd
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame
df["target"] = iris.target  # Add target column

# Save to CSV
df.to_csv("data/iris.csv", index=False)
print("Iris dataset saved to data/iris.csv")
