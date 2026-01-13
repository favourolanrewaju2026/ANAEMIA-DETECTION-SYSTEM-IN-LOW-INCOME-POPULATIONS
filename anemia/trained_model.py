# Anemia Detection - Model Training Script


# Import Required Libraries
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, recall_score


# Loading the Dataset 
Data = pd.read_csv("C:\\anemia.csv")


# Basic Data Cleaning
Data = Data.drop_duplicates()


# Splitting Features and Target
# X = Data.drop(columns=["Result"])
X = Data[["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"]]

y = Data["Result"]

# # Ensure target is numeric
# Data["Result"] = Data["Result"].map({
#     "Not Anemic": 0,
#     "Anemic": 1
# })


# Train-Test Split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    train_size=0.8,
    random_state=42,
    stratify=y
)


# Identifying Numerical & Categorical Columns

num_features = X.select_dtypes(include=["number"]).columns
# cat_features = X.select_dtypes(include=["object"]).columns


# Preprocessing Pipelines
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# cat_transformer = Pipeline(steps=[
#     ("imputer", SimpleImputer(strategy="most_frequent")),
#     ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
# ])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_features),
    ]
)

# ("cat", cat_transformer, cat_features)

# Model Pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("forest", RandomForestClassifier(random_state=42))
])


# Hyperparameter Tuning
param_grid = {
    "forest__n_estimators": [100, 200],
    "forest__max_depth": [None, 10, 20],
    "forest__min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)


# Training
grid_search.fit(Xtrain, ytrain)

best_model = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)


# Evaluation
y_pred = best_model.predict(Xtest)
y_prob = best_model.predict_proba(Xtest)[:, 1]

print("Accuracy:", accuracy_score(ytest, y_pred))
print("Recall (Sensitivity):", recall_score(ytest, y_pred))
print("ROC-AUC:", roc_auc_score(ytest, y_prob))
print("\nClassification Report:\n", classification_report(ytest, y_pred))


# Saving the Model (FOR FLASK DEPLOYMENT)
with open("model/anemia_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("✅ Model trained and saved successfully!")
