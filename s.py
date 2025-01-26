import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the samples
samples = [
    pd.read_csv(f"Sample_{i}.csv") for i in range(1, 6)
]

# ML Models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

# Initialize results storage
results = []

# Evaluate each model on each sample
for sample_idx, sample in enumerate(samples, 1):
    print(f"\nEvaluating on Sample {sample_idx}...")
    
    # Separate features and target
    X = sample.drop(columns=['Class'])
    y = sample['Class']

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    row_result = {"Sample": sample_idx}

    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)

        # Store results
        row_result[model_name] = accuracy
        
        print(f"Model: {model_name} on Sample {sample_idx}, Accuracy: {accuracy:.4f}")
    
    results.append(row_result)

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.set_index('Sample', inplace=True)
results_df.to_csv("Model_Sample_Comparison2.csv")
print("\nResults saved to 'Model_Sample_Comparison2.csv'")
