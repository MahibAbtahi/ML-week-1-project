import os
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data from your 'ml' folder
data_path = "ml/Titanic-Dataset.xlsx"
df = pd.read_excel(data_path)

# 2. Build a quick, clean Histogram for Age
plt.figure(figsize=(7, 5))
plt.hist(df["Age"].dropna(), bins=20, color="skyblue", edgecolor="black")

# 3. Add simple labels
plt.title("Passenger Age Distribution (Day 2)")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

# 4. FORCE the graph to pop up on your screen instantly!
plt.show()