import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel('Rice2024_cleaned.xlsx', engine='openpyxl')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

model = RandomForestClassifier(
    n_estimators= 100,
    max_depth= 10,
    min_samples_split= 10,
    min_samples_leaf= 1,
    max_features= 'sqrt'
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác mô hình: {accuracy*100:.3f} %")
joblib.dump(model, 'random_forest_model_rice.pkl')
print("Save model successfully: random_forest_model_rice.pkl")