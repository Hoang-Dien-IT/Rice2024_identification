import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt

df = pd.read_excel('Rice2024_cleaned.xlsx', engine='openpyxl')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

models = {
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'DecisionTree': DecisionTreeClassifier(max_depth=100),
    'NaiveBayes': GaussianNB()
}

metrics = {
    'accuracy': {'RandomForest': 0, 'DecisionTree': 0, 'NaiveBayes': 0},
    'precision': {'RandomForest': 0, 'DecisionTree': 0, 'NaiveBayes': 0},
    'recall': {'RandomForest': 0, 'DecisionTree': 0, 'NaiveBayes': 0}
}

# Vòng lặp huấn luyện và tính toán kết quả 20 lần
for i in range(20):
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics['accuracy'][name] += accuracy_score(y_test, y_pred)
        metrics['precision'][name] += precision_score(y_test, y_pred, average='weighted')
        metrics['recall'][name] += recall_score(y_test, y_pred, average='weighted')

# Tính giá trị trung bình cho mỗi mô hình
for metric in metrics:
    for name in metrics[metric]:
        metrics[metric][name] /= 20

for model_name in models:
    print(f"Accuracy trung bình 20 lần {model_name}: {metrics['accuracy'][model_name]*100:.2f} %")
    print(f"Precision trung bình 20 lần {model_name}: {metrics['precision'][model_name]:.2f}")
    print(f"Recall trung bình 20 lần {model_name}: {metrics['recall'][model_name]:.2f}")

"""Biểu đồ trực qua giá trị trung bình 20 lần của accuracy"""
models_list = list(models.keys())
accuracies = [metrics['accuracy'][name] for name in models_list]

# Tạo biểu đồ cột
plt.figure(figsize=(10, 6))
plt.bar(models_list, accuracies, color=['blue', 'green', 'red'])

# Thêm tiêu đề và nhãn trục
plt.title('Độ chính xác trung bình của các mô hình phân loại')
plt.xlabel('Mô hình')
plt.ylabel('Độ chính xác')

# Hiển thị giá trị trên cột
for i, value in enumerate(accuracies):
    plt.text(i, value + 0.01, f'{value*100:.2f}%', ha='center', va='bottom')

plt.show()

# Tạo dữ liệu cho Precision và Recall
precisions = [metrics['precision'][name] for name in models_list]
recalls = [metrics['recall'][name] for name in models_list]

# Thiết lập vị trí cho các nhóm cột
bar_width = 0.35
index = np.arange(len(models_list))

# Tạo biểu đồ
plt.figure(figsize=(12, 6))
bar1 = plt.bar(index, precisions, bar_width, label='Precision', color='skyblue')
bar2 = plt.bar(index + bar_width, recalls, bar_width, label='Recall', color='lightcoral')

plt.title('Precision và Recall trung bình của các mô hình phân loại')
plt.xlabel('Mô hình')
plt.ylabel('Giá trị')
plt.xticks(index + bar_width / 2, models_list)
plt.legend()

for i, (p, r) in enumerate(zip(precisions, recalls)):
    plt.text(i, p + 0.01, f'{p:.2f}', ha='center', va='bottom')
    plt.text(i + bar_width, r + 0.01, f'{r:.2f}', ha='center', va='bottom')

plt.show()