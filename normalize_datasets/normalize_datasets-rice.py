from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
# os.system('cls')

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

print(df_cleaned.head())

# _____________________________________________________________________
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
print("max ", max_value)
print("min ", min_value)

# xóa hàng 429,,,,0830078125
print("Xóa hàng Perimeter = 429,,,,0830078125")
df_cleaned = df_cleaned[df_cleaned[column_name] != '429,,,,0830078125']

# # Vẽ biểu đồ phân phối cho cột 'Perimeter'
# plt.figure(figsize=(10, 6))
# plt.hist(df_cleaned['Perimeter'], bins=10, edgecolor='k', alpha=0.7)
# plt.title('Phân phối của cột Perimeter')
# plt.xlabel('Perimeter')
# plt.ylabel('Số lượng')
# plt.grid(True)
# plt.show()


# _____________________________________________________________________
column_name = 'Major_Axis_Length'  # Thay đổi tên cột theo dữ liệu của bạn

# Tìm giá trị lớn nhất và nhỏ nhất
max_value = df_cleaned[column_name].max()
min_value = df_cleaned[column_name].min()
# Tìm chỉ số của giá trị lớn nhất và nhỏ nhất
max_index = df_cleaned[column_name].idxmax()
min_index = df_cleaned[column_name].idxmin()
print(column_name)
print("max", max_value)
print("min", min_value)

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
print("Đã xóa dữ liệu Major_Axis_Length nhiễu")
df_cleaned = df_cleaned.drop(df_cleaned['Major_Axis_Length'].idxmax())

print("Tìm Major_Axis_Length nhiễu: ", df_cleaned['Major_Axis_Length'].max())
print("Đã xóa dữ liệu Major_Axis_Length nhiễu")
df_cleaned = df_cleaned.drop(df_cleaned['Major_Axis_Length'].idxmax())

# Vẽ biểu đồ phân phối cho cột 'Major_Axis_Length'
plt.figure(figsize=(10, 6))
plt.hist(df_cleaned['Major_Axis_Length'], bins=20, edgecolor='k', alpha=0.7)
plt.title('Phân phối của cột Major_Axis_Length')
plt.xlabel('Major_Axis_Length')
plt.ylabel('Số lượng')
plt.grid(True)
plt.show()

# #_____________________________________________________________________
column_name = 'Minor_Axis_Length'  # Thay đổi tên cột theo dữ liệu của bạn

# Tìm giá trị lớn nhất và nhỏ nhất
max_value = df_cleaned[column_name].max()
min_value = df_cleaned[column_name].min()

# Tìm chỉ số của giá trị lớn nhất và nhỏ nhất
max_index = df_cleaned[column_name].idxmax()
min_index = df_cleaned[column_name].idxmin()
print(column_name)
print("max", max_value)
print("min", min_value)


# Vẽ biểu đồ phân phối cho cột 'Minor_Axis_Length'
plt.figure(figsize=(10, 6))
plt.hist(df_cleaned['Minor_Axis_Length'], bins=20, edgecolor='k', alpha=0.7)
plt.title('Phân phối của cột Minor_Axis_Length')
plt.xlabel('Minor_Axis_Length')
plt.ylabel('Số lượng')
plt.grid(True)
plt.show()

print("Tìm Minor_Axis_Length nhiễu: ", df_cleaned['Minor_Axis_Length'].max())
print("Đã xáo dữ liệu Minor_Axis_Length nhiễu")
df_cleaned = df_cleaned.drop(df_cleaned['Minor_Axis_Length'].idxmax())

# Vẽ biểu đồ phân phối cho cột 'Minor_Axis_Length'
plt.figure(figsize=(10, 6))
plt.hist(df_cleaned['Minor_Axis_Length'], bins=20, edgecolor='k', alpha=0.7)
plt.title('Phân phối của cột Minor_Axis_Length')
plt.xlabel('Minor_Axis_Length')
plt.ylabel('Số lượng')
plt.grid(True)
plt.show()
# #_____________________________________________________________________
column_name = 'Eccentricity'  # Thay đổi tên cột theo dữ liệu của bạn

# Tìm giá trị lớn nhất và nhỏ nhất
max_value = df_cleaned[column_name].max()
min_value = df_cleaned[column_name].min()

# Tìm chỉ số của giá trị lớn nhất và nhỏ nhất
max_index = df_cleaned[column_name].idxmax()
min_index = df_cleaned[column_name].idxmin()
print(column_name)
print("max", max_value)
print("min", min_value)

# Vẽ biểu đồ phân phối cho cột 'Eccentricity'
plt.figure(figsize=(10, 6))
plt.hist(df_cleaned['Eccentricity'], bins=20, edgecolor='k', alpha=0.7)
plt.title('Phân phối của cột Eccentricity')
plt.xlabel('Eccentricity')
plt.ylabel('Số lượng')
plt.grid(True)
plt.show()

# #_____________________________________________________________________
column_name = 'Convex_Area'  # Thay đổi tên cột theo dữ liệu của bạn

# Tìm giá trị lớn nhất và nhỏ nhất
max_value = df_cleaned[column_name].max()
min_value = df_cleaned[column_name].min()

# Tìm chỉ số của giá trị lớn nhất và nhỏ nhất
max_index = df_cleaned[column_name].idxmax()
min_index = df_cleaned[column_name].idxmin()
print(column_name)
print("max", max_value)
print("min", min_value)

# Vẽ biểu đồ phân phối cho cột 'Convex_Area'
plt.figure(figsize=(10, 6))
plt.hist(df_cleaned['Convex_Area'], bins=20, edgecolor='k', alpha=0.7)
plt.title('Phân phối của cột Convex_Area')
plt.xlabel('Convex_Area')
plt.ylabel('Số lượng')
plt.grid(True)
plt.show()


# #_____________________________________________________________________
column_name = 'Extent'  # Thay đổi tên cột theo dữ liệu của bạn

# Tìm giá trị lớn nhất và nhỏ nhất
max_value = df_cleaned[column_name].max()
min_value = df_cleaned[column_name].min()

# Tìm chỉ số của giá trị lớn nhất và nhỏ nhất
max_index = df_cleaned[column_name].idxmax()
min_index = df_cleaned[column_name].idxmin()
print(column_name)
print("max", max_value)
print("min", min_value)

# Vẽ biểu đồ phân phối cho cột 'Extent'
plt.figure(figsize=(10, 6))
plt.hist(df_cleaned['Extent'], bins=20, edgecolor='k', alpha=0.7)
plt.title('Phân phối của cột Extent')
plt.xlabel('Extent')
plt.ylabel('Số lượng')
plt.grid(True)
plt.show()


# #_____________________________________________________________________

# Vẽ biểu đồ phân phối cho cột 'Class'
plt.figure(figsize=(10, 6))
plt.hist(df_cleaned['Class'], bins=20, edgecolor='k', alpha=0.7)
plt.title('Phân phối của cột Class')
plt.xlabel('Class')
plt.ylabel('Số lượng')
plt.grid(True)
plt.show()


print("c -> C, o -> O")
df_cleaned['Class'] = df_cleaned['Class'].replace({'c': 'C', 'o': 'O'})


# Vẽ biểu đồ phân phối cho cột 'Class'
plt.figure(figsize=(10, 6))
plt.hist(df_cleaned['Class'], bins=20, edgecolor='k', alpha=0.7)
plt.title('Phân phối của cột Class')
plt.xlabel('Class')
plt.ylabel('Số lượng')
plt.grid(True)
plt.show()

# #_____________________________________________________________________


# scale
# Thực hiện Label Encoding cho các cột 'Class'
df_cleaned.loc[:, 'Class'] = df_cleaned['Class'].map({'C': 0, 'O': 1})

# #_____________________________________________________________________

# chuẩn hóa min-max
scaler = MinMaxScaler(feature_range=(0, 1))
# Chọn các cột số để chuẩn hóa
columns_to_scale = ['Area', 'Perimeter',
                    'Major_Axis_Length', 'Minor_Axis_Length', 'Eccentricity', 'Extent', 'Convex_Area']
# Tạo một bản sao của DataFrame để chuẩn hóa
df_scaled = df_cleaned.copy()
# Áp dụng chuẩn hóa Min-Max
df_scaled[columns_to_scale] = scaler.fit_transform(
    df_cleaned[columns_to_scale])

# In ra DataFrame sau khi chuẩn hóa
print("5 dòng đầu của dữ liệu sau khi chuẩn hóa Min-Max:")
print(df_scaled.head())

# lưu scale
name_file = "standardized-rice.csv"
print(f"Đã lưu {name_file}")
# df_scaled.to_csv(name_file, index=False)
