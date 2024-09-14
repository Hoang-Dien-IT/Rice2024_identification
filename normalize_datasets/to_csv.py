import pandas as pd
 


# Đọc file xlsx
df = pd.read_excel('./Rice2024.xlsx', engine='openpyxl')

# Ghi vào file csv
df.to_csv('Rice2024.csv', index=False)
