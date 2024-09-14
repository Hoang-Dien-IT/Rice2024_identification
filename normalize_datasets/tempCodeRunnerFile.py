# from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
os.system('cls')

df = pd.read_csv('./Rice2024.csv')

# xóa cột 'Id', 'Nickname'
df = df.drop(['Id', 'Nickname'], axis=1)

# In ra các cột có dữ liệu bị thiếu
print("Số lượng dữ liệu bị thiếu trong mỗi cột:")
missing_data = df.isnull().sum()
print(missing_data)

# Xóa các hàng có dữ liệu thiếu
print("Xóa các hàng thiếu dữ liệu")
df_cleaned = df.dropna()

missing_data = df_cleaned.isnull().sum()
print("Số lượng dữ liệu bị thiếu trong mỗi cột:")
print(missing_data)

print(df_cleaned.head())

# Kiểm tra toàn bộ dữ liệu có bị thiếu không
if df_cleaned.isnull().values.any():
    print("\nCó dữ liệu bị thiếu.")
else:
    print("\nKhông có dữ liệu bị thiếu.")

# print(df_cleaned.head())

#_____________________________________________________________________
# Vẽ biểu đồ phân phối cho cột 'Area'
plt.figure(figsize=(10, 6))
plt.hist(df_cleaned['Area'], bins=20, edgecolor='k', alpha=0.7)
plt.title('Phân phối của cột Area')
plt.xlabel('Area')
plt.ylabel('Số lượng')
plt.grid(True)
plt.show()


# #_____________________________________________________________________
column_name = 'Perimeter'  # Thay đổi tên cột theo dữ liệu của bạn

# Tìm giá trị lớn nhất và nhỏ nhất
max_value = df_cleaned[column_name].max()
min_value = df_cleaned[column_name].min()

# Tìm chỉ số của giá trị lớn nhất và nhỏ nhất
max_index = df_cleaned[column_name].idxmax()
min_index = df_cleaned[column_name].idxmin()
print(column_name)
print("max", max_value)
print("min", min_value)

#xóa hàng 429,,,,0830078125
df_cleaned = df_cleaned[df_cleaned[column_name] != '429,,,,0830078125']

# #_____________________________________________________________________
# column_name = 'Major_Axis_Length'  # Thay đổi tên cột theo dữ liệu của bạn

# # Tìm giá trị lớn nhất và nhỏ nhất
# max_value = df_cleaned[column_name].max()
# min_value = df_cleaned[column_name].min()

# # Tìm chỉ số của giá trị lớn nhất và nhỏ nhất
# max_index = df_cleaned[column_name].idxmax()
# min_index = df_cleaned[column_name].idxmin()
# print(column_name)
# print("max", max_value)
# print("min", min_value)

# Vẽ biểu đồ phân phối cho cột 'Major_Axis_Length'
plt.figure(figsize=(10, 6))
plt.hist(df_cleaned['Major_Axis_Length'], bins=20, edgecolor='k', alpha=0.7)
plt.title('Phân phối của cột Major_Axis_Length')
plt.xlabel('Major_Axis_Length')
plt.ylabel('Số lượng')
plt.grid(True)
plt.show()

# Xóa hàng dữ liệu nhiễu Major_Axis_Length
print("Tìm Major_Axis_Length nhiễu: ", df_cleaned['Major_Axis_Length'].max())
print("Đã xáo dữ liệu Major_Axis_Length nhiễu")
df_cleaned = df_cleaned.drop(df_cleaned['Major_Axis_Length'].idxmax())

print("Tìm Major_Axis_Length nhiễu: ", df_cleaned['Major_Axis_Length'].max())
print("Đã xáo dữ liệu Major_Axis_Length nhiễu")
df_cleaned = df_cleaned.drop(df_cleaned['Major_Axis_Length'].idxmax())

# Vẽ biểu đồ phân phối cho cột 'Major_Axis_Length'
plt.figure(figsize=(10, 6))
plt.hist(df_cleaned['Major_Axis_Length'], bins=20, edgecolor='k', alpha=0.7)
plt.title('Phân phối của cột Major_Axis_Length')
plt.xlabel('Major_Axis_Length')
plt.ylabel('Số lượng')
plt.grid(True)
plt.show()