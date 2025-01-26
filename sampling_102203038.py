import pandas as pd
import math as math

# Load the dataset
data = pd.read_csv("Creditcard_data.csv")

# Separate the two classes
class_0 = data[data['Class'] == 0]
class_1 = data[data['Class'] == 1]

# Oversample the minority class by duplicating samples
class_1_oversampled = class_1.sample(n=len(class_0), replace=True, random_state=42)

# Combine the classes
balanced_data = pd.concat([class_0, class_1_oversampled])

# Shuffle the data
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the balanced dataset
balanced_data.to_csv("Balanced_Creditcard_data.csv", index=False)

Z = 1.96  # Z-score for 95% confidence
p = 0.5  # Proportion of the population
e = 0.05  # Margin of error

# Calculate sample size
sample_size = math.ceil((Z**2 * p * (1 - p)) / (e**2))
print(f"Sample size: {sample_size}")

# Generate 5 random samples
for i in range(1, 6):
    sample = data.sample(n=sample_size, random_state=i)
    sample.to_csv(f"Sample_{i}.csv", index=False)
    print(f"Sample {i} created with {sample_size} rows.")

