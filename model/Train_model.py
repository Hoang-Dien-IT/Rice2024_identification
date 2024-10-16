import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

scale = MinMaxScaler()

df = pd.read_excel('../data_processing/xoa-2cot-rice.xlsx', engine='openpyxl')

X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

print("Dư liệu X:", X)
print("Dư liêu y", y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=50)

X_train_scaled = scale.fit_transform(X_train)
X_test_scaled = scale.transform(X_test)

print(X_train_scaled)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=1,
    max_features='sqrt'
)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
confusion_matrix = confusion_matrix(y_test, y_pred)

print(f"Độ chính xác mô hình accuracy: {accuracy*100:.3f} %")
print(f"Độ chính xác mô hình precision: {precision:.3f}")
print(f"Độ chính xác mô hình recall: {accuracy:.3f}")
print(f"ma trận nhầm lẫn: \n", confusion_matrix)

metrics = ['Accuracy', 'Precision', 'Recall']
values = [accuracy*100, precision * 100, recall * 100]

plt.figure(figsize=(8, 5))
plt.bar(metrics, values, color=['blue', 'green', 'orange'])

plt.title('Biểu đồ so sánh độ chính xác, precision và recall của mô hình', fontsize=14)
plt.ylabel('Giá trị (%)', fontsize=12)
plt.ylim([0, 100])
for i, value in enumerate(values):
    plt.text(i, value + 1, f'{value:.2f}%', ha='center', fontsize=12)
plt.show()

joblib.dump(model, 'random_forest_model_rice.pkl')
print("Save model successfully: random_forest_model_rice.pkl")

plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
