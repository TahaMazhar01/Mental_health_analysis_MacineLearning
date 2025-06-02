import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import warnings
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, DecisionTreeClassifier

warnings.filterwarnings('ignore')

# --- 1. Data Loading ---
# Load the dataset using pd.read_excel as it is an .xlsx file
df = pd.read_excel('Dataset.xlsx')

# --- 2. Data Preprocessing ---
# Clean column names by stripping leading/trailing spaces and replacing spaces with underscores
# Also, remove any characters that are not alphanumeric or underscores
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)

# Select the target variable for classification
# This is the column we want to predict
target_column = 'Have_you_ever_been_diagnosed_with_a_mental_health_condition_by_a_professional_doctor_therapist_etc'

# Identify all categorical columns (object type) for encoding, excluding Timestamp and the target
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if 'Timestamp' in categorical_cols:
    categorical_cols.remove('Timestamp')
if target_column in categorical_cols:
    categorical_cols.remove(target_column)

# Apply Label Encoding to convert categorical features into numerical format
# This is necessary for most machine learning algorithms
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Encode the target variable as well
le_target = LabelEncoder()
df[target_column] = le_target.fit_transform(df[target_column])
target_class_names = le_target.classes_ # Get the original class names for the plot

# Define features (X) and target (y)
# X contains all the columns that will be used to make predictions
# y contains the target variable we want to predict
X = df.drop(columns=[target_column, 'Timestamp']) # Drop Timestamp as it's not a feature for prediction
y = df[target_column]

# Get feature names for visualization and importance
feature_names = X.columns.tolist()

# --- 3. Data Splitting ---
# Split the dataset into training and testing sets
# X_train, y_train are used to train the model
# X_test, y_test are used to evaluate the model's performance on unseen data
# test_size=0.2 means 20% of the data will be used for testing, 80% for training
# random_state ensures reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Model Training and Evaluation (Random Forest Classifier) ---
# Initialize and train the Random Forest Classifier model
# RandomForestClassifier is an ensemble tree-based algorithm suitable for classification
# random_state ensures reproducibility of the model training
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the Random Forest model's performance using common classification metrics
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, zero_division=0)
recall_rf = recall_score(y_test, y_pred_rf, zero_division=0)
f1_rf = f1_score(y_test, y_pred_rf, zero_division=0)

# Print Random Forest evaluation metrics
print("--- Model Performance (Random Forest Classifier) ---")
print(f"Accuracy: {accuracy_rf:.2f}")
print(f"Precision: {precision_rf:.2f}")
print(f"Recall: {recall_rf:.2f}")
print(f"F1-Score: {f1_rf:.2f}")

# --- 5. Feature Importances from Random Forest ---
print("\n--- Feature Importances from Random Forest (based on Gini impurity reduction) ---")
feature_importances = pd.DataFrame({'feature': feature_names, 'importance': rf_model.feature_importances_})
feature_importances = feature_importances.sort_values('importance', ascending=False)
print(feature_importances.to_markdown(index=False, numalign="left", stralign="left"))

# --- 6. Visualize a Single Decision Tree (for structure and Gini impurity display) ---
# We train a separate DecisionTreeClassifier here with limited depth for clear visualization.
# A full Random Forest has many trees, and a very deep single tree is unreadable.
dt_visualization_model = DecisionTreeClassifier(random_state=42, max_depth=3) # Limiting depth for better visualization
dt_visualization_model.fit(X_train, y_train)

print("\n--- Visualizing a Single Decision Tree Structure (max_depth=3) ---")
print("A plot window should appear (or it will be inline in Colab).")
plt.figure(figsize=(25, 15)) # Adjust figure size for better readability
plot_tree(dt_visualization_model,
          feature_names=feature_names,
          class_names=target_class_names,
          filled=True,
          rounded=True,
          impurity=True, # This ensures Gini impurity is shown at each node
          fontsize=10)
plt.title("Decision Tree Classifier (Limited Depth: 3 for Visualization)")
plt.show() # Display the plot