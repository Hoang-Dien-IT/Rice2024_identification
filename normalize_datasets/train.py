import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import os
os.system('cls')

# Đọc dữ liệu từ file CSV
df = pd.read_csv('./standardized-rice.csv')

# Chia dữ liệu thành đặc trưng và mục tiêu
target_column = 'Class'  # Thay đổi tên cột mục tiêu theo dữ liệu của bạn
X = df.drop(columns=[target_column])
y = df[target_column]
print(X)
print(y)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Huấn luyện và đánh giá mô hình

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print(classification_report(y_test, y_pred_log_reg))

# Dự đoán cho hàng đầu tiên trong tập dữ liệu
first_row = X.iloc[0].values.reshape(1, -1)  # Chọn hàng đầu tiên và chuyển đổi thành định dạng (1, -1)
predicted_value = log_reg.predict(first_row)
print(f"Dự đoán cho hàng đầu tiên: {predicted_value[0]}")

print("____________________________________________________")
# Support Vector Machine
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("\nSupport Vector Machine:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# Dự đoán cho hàng đầu tiên trong tập dữ liệu
first_row = X.iloc[0].values.reshape(1, -1)  # Chọn hàng đầu tiên và chuyển đổi thành định dạng (1, -1)
predicted_value = svm.predict(first_row)
print(f"Dự đoán cho hàng đầu tiên: {predicted_value[0]}")

# Random Forest
print("____________________________________________________")
random_forest = RandomForestClassifier(n_estimators=100, max_features='sqrt', min_samples_split=10, min_samples_leaf=1, max_depth=10)
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)
print("\nRandom Forest:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
# Dự đoán cho hàng đầu tiên trong tập dữ liệu
first_row = X.iloc[0].values.reshape(1, -1)  # Chọn hàng đầu tiên và chuyển đổi thành định dạng (1, -1)
predicted_value = random_forest.predict(first_row)
print(f"Dự đoán cho hàng đầu tiên: {predicted_value[0]}")


# Support Vector Machine
print("____________________________________________________")
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("\nSupport Vector Machine:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))


# Dự đoán cho hàng đầu tiên trong tập dữ liệu
first_row = X.iloc[0].values.reshape(1, -1)  # Chọn hàng đầu tiên và chuyển đổi thành định dạng (1, -1)
predicted_value = svm.predict(first_row)
print(f"Dự đoán cho hàng đầu tiên: {predicted_value[0]}")