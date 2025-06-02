import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
file_path = "custom_suspicious_transactions.csv"
df = pd.read_csv(file_path)

# Encode categorical variables
mapping_yes_no = {'Yes': 1, 'No': 0}
mapping_transaction_type = {'POS': 0, 'ATM': 1, 'Online': 2, 'Wire': 3}

df['Geo_Location_Mismatched'] = df['Geo_Location_Mismatched'].map(mapping_yes_no)
df['Same_Bank'] = df['Same_Bank'].map(mapping_yes_no)
df['Transaction_Type'] = df['Transaction_Type'].map(mapping_transaction_type)
df['Receiver_New_Balance_Wrong'] = df['Receiver_New_Balance_Wrong'].map(mapping_yes_no)

# Encode target variable
label_encoder = LabelEncoder()
df['Suspicious_Label'] = label_encoder.fit_transform(df['Suspicious_Label'])
joblib.dump(label_encoder, "label_encoder.pkl")  # Save label encoder

# Define features and target
X = df.drop(columns=['Suspicious_Label'])
y = df['Suspicious_Label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")  # Save scaler

# Define models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear', probability=True),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    "LogisticRegression": LogisticRegression(max_iter=500),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# Train and evaluate models
best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Save the best model
model_path = "best_suspicious_model.pkl"
joblib.dump(best_model, model_path)
print(f"Best model saved as {model_path} with accuracy {best_accuracy:.4f}")
