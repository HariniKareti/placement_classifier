import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# ===== LOAD DATA =====
df = pd.read_csv("student_data.csv")

# Features & Target
X = df.drop(columns=["Recommend", "Name"])
y = df["Recommend"]

# Categorical columns
ordinal_cols = ["OverallGrade"]
ordinal_mapping = [["E", "D", "C", "B", "A"]]  # Worst to Best
onehot_cols = ["Obedient"]

# Numeric columns
numeric_cols = ["ResearchScore", "ProjectScore"]

# ===== PREPROCESSING =====
ordinal_encoder = OrdinalEncoder(categories=ordinal_mapping)
onehot_encoder = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ("ordinal", ordinal_encoder, ordinal_cols),
        ("onehot", onehot_encoder, onehot_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# ===== MODEL PIPELINE =====
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, "student_recommend_model.joblib")
print("✅ Model saved as student_recommend_model.joblib")
