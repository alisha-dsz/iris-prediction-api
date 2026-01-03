from sklearn.datasets import load_iris
import pandas as pd

# Load Iris dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Save to CSV (optional)
df.to_csv('iris.csv', index=False)
print(df.head())
