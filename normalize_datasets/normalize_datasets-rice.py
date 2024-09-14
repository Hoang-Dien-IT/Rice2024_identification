from sklearn.preprocessing import MinMaxScaler
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

# # Vẽ biểu đồ phân phối cho cột 'Perimeter'
# plt.figure(figsize=(10, 6))
# plt.hist(df_cleaned['Perimeter'], bins=20, edgecolor='k', alpha=0.7)
# plt.title('Phân phối của cột Perimeter')
# plt.xlabel('Perimeter')
# plt.ylabel('Số lượng')
# plt.grid(True)
# plt.show()

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

# # Vẽ biểu đồ phân phối cho cột 'Minor_Axis_Length'
# plt.figure(figsize=(10, 6))
# plt.hist(df_cleaned['Minor_Axis_Length'], bins=20, edgecolor='k', alpha=0.7)
# plt.title('Phân phối của cột Minor_Axis_Length')
# plt.xlabel('Minor_Axis_Length')
# plt.ylabel('Số lượng')
# plt.grid(True)
# plt.show()


print("Tìm Minor_Axis_Length nhiễu: ", df_cleaned['Minor_Axis_Length'].max())
print("Đã xáo dữ liệu Minor_Axis_Length nhiễu")
df_cleaned = df_cleaned.drop(df_cleaned['Minor_Axis_Length'].idxmax())

# # Vẽ biểu đồ phân phối cho cột 'Minor_Axis_Length'
# plt.figure(figsize=(10, 6))
# plt.hist(df_cleaned['Minor_Axis_Length'], bins=20, edgecolor='k', alpha=0.7)
# plt.title('Phân phối của cột Minor_Axis_Length')
# plt.xlabel('Minor_Axis_Length')
# plt.ylabel('Số lượng')
# plt.grid(True)
# plt.show()
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

# # Vẽ biểu đồ phân phối cho cột 'Eccentricity'
# plt.figure(figsize=(10, 6))
# plt.hist(df_cleaned['Eccentricity'], bins=20, edgecolor='k', alpha=0.7)
# plt.title('Phân phối của cột Eccentricity')
# plt.xlabel('Eccentricity')
# plt.ylabel('Số lượng')
# plt.grid(True)
# plt.show()

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

# # Vẽ biểu đồ phân phối cho cột 'Convex_Area'
# plt.figure(figsize=(10, 6))
# plt.hist(df_cleaned['Convex_Area'], bins=20, edgecolor='k', alpha=0.7)
# plt.title('Phân phối của cột Convex_Area')
# plt.xlabel('Convex_Area')
# plt.ylabel('Số lượng')
# plt.grid(True)
# plt.show()


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

# # Vẽ biểu đồ phân phối cho cột 'Extent'
# plt.figure(figsize=(10, 6))
# plt.hist(df_cleaned['Extent'], bins=20, edgecolor='k', alpha=0.7)
# plt.title('Phân phối của cột Extent')
# plt.xlabel('Extent')
# plt.ylabel('Số lượng')
# plt.grid(True)
# plt.show()


# #_____________________________________________________________________

# # Vẽ biểu đồ phân phối cho cột 'Class'
# plt.figure(figsize=(10, 6))
# plt.hist(df_cleaned['Class'], bins=20, edgecolor='k', alpha=0.7)
# plt.title('Phân phối của cột Class')
# plt.xlabel('Class')
# plt.ylabel('Số lượng')
# plt.grid(True)
# plt.show()


print("c -> C, o -> O")
df_cleaned['Class'] = df_cleaned['Class'].replace({'c': 'C', 'o': 'O'})


# # Vẽ biểu đồ phân phối cho cột 'Class'
# plt.figure(figsize=(10, 6))
# plt.hist(df_cleaned['Class'], bins=20, edgecolor='k', alpha=0.7)
# plt.title('Phân phối của cột Class')
# plt.xlabel('Class')
# plt.ylabel('Số lượng')
# plt.grid(True)
# plt.show()

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
# Tạo các biến để lưu giá trị nhỏ nhất và lớn nhất cho từng cột
Area = [df_cleaned['Area'].min(), df_cleaned['Area'].max()]
Perimeter = [df_cleaned['Perimeter'].min(), df_cleaned['Perimeter'].max()]
Major_Axis_Length = [
    df_cleaned['Major_Axis_Length'].min(), df_cleaned['Major_Axis_Length'].max()]
Minor_Axis_Length = [
    df_cleaned['Minor_Axis_Length'].min(), df_cleaned['Minor_Axis_Length'].max()]
Eccentricity = [df_cleaned['Eccentricity'].min(), df_cleaned['Eccentricity'].max()]
Convex_Area = [df_cleaned['Convex_Area'].min(), df_cleaned['Convex_Area'].max()]
Extent = [df_cleaned['Extent'].min(), df_cleaned['Extent'].max()]

# Kiểm tra kết quả
print("Area:", Area)
print("Perimeter:", Perimeter)
print("Major_Axis_Length:", Major_Axis_Length)
print("Minor_Axis_Length:", Minor_Axis_Length)
print("Eccentricity:", Eccentricity)
print("Convex_Area:", Convex_Area)
print("Extent:", Extent)

a = [Area, Perimeter, Major_Axis_Length,
     Minor_Axis_Length, Eccentricity, Convex_Area, ]
df_scaled.to_csv('standardized-rice.csv', index=False)

        
        
        
        