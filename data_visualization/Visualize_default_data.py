import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Đọc tập dữ liệu từ file CSV (giả định rằng bạn có file rice_dataset.csv)
df = pd.read_excel('Rice2024.xlsx', engine='openpyxl')

# Hiển thị vài dòng đầu tiên của tập dữ liệu
print(df.head())

# Kiểm tra thông tin dữ liệu
print(df.info())

# Thống kê cơ bản về các cột số
print(df.describe())

df = df[df['Perimeter'] != '429,,,,0830078125']

# Vẽ biểu đồ phân phối cho các thuộc tính số
df.hist(bins=30, figsize=(12, 10))
plt.show()

# Kiểm tra ngoại lệ với biểu đồ boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title('Biểu đồ Boxplot kiểm tra ngoại lệ cho tập dữ liệu Rice', fontsize=16)
plt.xlabel('Thuộc tính', fontsize=9)
plt.ylabel('Giá trị', fontsize=12)
plt.show()

# Loại bỏ cột 'Class' hoặc bất kỳ cột nào không phải dạng số
numeric_df = df.select_dtypes(include=[np.number])
# Tính toán ma trận tương quan chỉ với các cột dạng số
corr_matrix = numeric_df.corr()
# Vẽ heatmap của ma trận tương quan
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.show()

# Vẽ biểu đồ tán xạ giữa Area và Perimeter
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Area', y='Perimeter', hue='Class', data=df)
plt.title('Scatter Plot between Area and Perimeter of Rice Grains')
plt.xlabel('Area (pixels)')
plt.ylabel('Perimeter (pixels)')
plt.show()


#Countplot - Biểu đồ đếm số lượng mẫu giữa các lớp
plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=df, palette='viridis')  # Sử dụng bảng màu 'viridis' cho màu sắc khác nhau
plt.title('Countplot of Rice Grain Classes')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

#Biểu đồ Eccentricity và Extent
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Eccentricity', y='Extent', hue='Class', data=df)
plt.title('Biểu đồ Eccentricity và Extent')
plt.show()

#Biểu đồ thể hiện số lượng dữ liệu thiếu
print(df.isnull().sum())

missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]
plt.figure(figsize=(10, 6))
missing_values.plot(kind='bar', color='orange')
plt.title('Biểu đồ số lượng dữ liệu thiếu của từng thuộc tính')
plt.ylabel('Number of Missing Values')
plt.show()






