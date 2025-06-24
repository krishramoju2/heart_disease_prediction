import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, VariableElimination
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load and clean the data
df = pd.read_csv("heart_disease.csv")
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Step 2: Normalize numerical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
df[numeric_cols] = df[numeric_cols].apply(lambda x: (x * 10).round(0).astype(int))  # Discretize

# Save cleaned data
df.to_csv("cleaned_data.csv", index=False)

# Step 3: Define Bayesian model structure (must be discrete)
model = BayesianNetwork([
    ('age', 'fbs'),
    ('fbs', 'target'),
    ('target', 'chol'),
    ('target', 'thalach')
])

# Step 4: Train using MLE
model.fit(df, estimator=MaximumLikelihoodEstimator)

# Step 5: Inference
inference = VariableElimination(model)

print("\nProbability of Heart Disease (target=1) given age=6 (normalized):")
q1 = inference.query(variables=['target'], evidence={'age': 6})
print(q1)

print("\nDistribution of cholesterol (chol) given heart disease (target=1):")
q2 = inference.query(variables=['chol'], evidence={'target': 1})
print(q2)

# Step 6: Visualize Network
nx.draw(model, with_labels=True, node_size=3000, node_color='lightblue', font_size=12)
plt.title("Bayesian Network: Heart Disease")
plt.savefig("network_visual.png")
plt.close()
