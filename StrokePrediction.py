
# ================================= Import Libraries =================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.svm import SVC



# ================================= Data Preprocessing =================================

# Load the datasets
df = pd.read_csv('training.csv')

# Inspect the dataset
print("Training Data Info:")
print(df.info())
print("Missing Values:\n", df.isnull().sum())

# Handle missing values (Fill missing numerical values with median)
df.fillna({'bmi': df['bmi'].median()}, inplace=True)

print("Missing Values After Imputing:\n", df.isnull().sum())

# Select categorical columns
categorical_features = df.select_dtypes(include=['object']).columns

# Initialize the encoder
encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

# Fit and transform categorical features
categorical_encoded = pd.DataFrame(encoder.fit_transform(df[categorical_features]),
                             columns=encoder.get_feature_names_out(categorical_features))  

# Reset index to align with original DataFrame
categorical_encoded.index = df.index

df = pd.concat([df.drop(columns=categorical_features), categorical_encoded], axis=1)

# Scale Numerical Features
scaler = StandardScaler()
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.drop('stroke')
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Define features (X) and target variable (y)
X = df.drop(columns=['stroke'])
y = df['stroke']

# Split into training (80%) and validation (20%) sets before oversampling
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Check class distribution before oversampling
print("Class distribution before oversampling:")
print(y_train.value_counts())

# Apply SMOTE for oversampling the minority class
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check class distribution after resampling
print("Class distribution after oversampling:")
print(y_train_resampled.value_counts())

# ================================= Feature Selection & Engineering =================================

# Correlation Analysis
plt.figure(figsize=(12, 8))
correlation_matrix = pd.concat([X_train_resampled, y_train_resampled], axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Feature Importance using Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

# Convert to DataFrame for better visualization
feature_importance_df = pd.DataFrame({
    'Feature': X_train_resampled.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title("Feature Importance (Random Forest)")
plt.show()

# Select the Most Important Features
# Keep the top 10 most relevant features
top_features = feature_importance_df['Feature'].head(10).tolist()
X_train_selected = X_train_resampled[top_features].copy()
X_val_selected = X_val[top_features].copy()

# Feature Engineering (Creating New Features)
X_train_selected['bmi_age_ratio'] = X_train_resampled['bmi'] / (X_train_resampled['age'] + 1)
X_val_selected['bmi_age_ratio'] = X_val['bmi'] / (X_val['age'] + 1)

# ================================= Model Training & Evaluation =================================
# Remove empty or single-value columns
X_train_selected = X_train_selected.loc[:, (X_train_selected.nunique() > 1)]
X_val_selected = X_val_selected.loc[:, (X_val_selected.nunique() > 1)]


# Define models to train
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=10, min_samples_leaf=3, class_weight={0: 1, 1: 3}, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
}

# Train and evaluate each model
results = {}

for name, model in models.items():
    # Train model
    model.fit(X_train_selected, y_train_resampled)
    
    # Make predictions
    y_pred = model.predict(X_val_selected)
    y_pred_proba = model.predict_proba(X_val_selected)[:, 1] if hasattr(model, 'predict_proba') else y_pred

    # Store results
    results[name] = {
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred, zero_division=0),
        "Recall": recall_score(y_val, y_pred),
        "F1 Score": f1_score(y_val, y_pred),
        "ROC AUC": roc_auc_score(y_val, y_pred_proba)
    }

# Convert results to DataFrame for better visualization
results_df = pd.DataFrame(results).T

# Print results sorted by highest accuracy
print("Model Performance Metrics:")
print(results_df.sort_values(by="Accuracy", ascending=False))

# ================================= Model Hyperparameter Tuning =================================

# Define the parameter grid
param_grid = {
    'n_estimators': [150, 200],
    'max_depth': [8, 10],
    'min_samples_split': [10, 15],
    'min_samples_leaf': [3, 5]
}

# Perform Grid Search
grid_search = GridSearchCV(RandomForestClassifier(class_weight={0: 1, 1: 3}, random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_selected, y_train_resampled)
print("Best Parameters:", grid_search.best_params_)

# ================================= Final Model Training & Evaluation =================================

# Train the optimized model
best_rf = RandomForestClassifier(**grid_search.best_params_, class_weight={0: 1, 1: 3}, random_state=42)
best_rf.fit(X_train_selected, y_train_resampled)
y_pred = best_rf.predict(X_val_selected)
y_pred_proba = best_rf.predict_proba(X_val_selected)[:, 1]

# Final evaluation
final_metrics = {
    "Accuracy": accuracy_score(y_val, y_pred),
    "Precision": precision_score(y_val, y_pred, zero_division=0),
    "Recall": recall_score(y_val, y_pred),
    "F1 Score": f1_score(y_val, y_pred),
    "ROC AUC": roc_auc_score(y_val, y_pred_proba)
}
print("Final Model Performance:")
for metric, value in final_metrics.items():
    print(f"{metric}: {value:.4f}")
