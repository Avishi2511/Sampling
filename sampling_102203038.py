import pandas as pd

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
