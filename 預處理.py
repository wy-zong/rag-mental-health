import pandas as pd 
from sklearn.model_selection import train_test_split
# 讀取資料集，只讀取指定欄位
data = (
    pd.read_csv('/home/aiuser/wy/csv/Combined_Data_Balanced.csv')
    .sample(frac=0.1)
    )

print("原始資料狀態：")
print(f"總行數：{data.shape[0]}")
print(f"類別分佈：\n{data['status'].value_counts()}")

# 找出最小的類別數量
min_count = data['status'].value_counts().min()
print(f"\n每個類別將被下採樣到 {min_count} 筆資料。")

# 對每個類別進行下採樣
data_balanced = data.groupby('status').apply(lambda x: x.sample(n=min_count, random_state=42)).reset_index(drop=True)

print("原始資料狀態：")
print(f"總行數：{data_balanced.shape[0]}")
print(f"類別分佈：\n{data_balanced['status'].value_counts()}")

data_balanced.to_csv('/home/aiuser/wy/csv/Combined_Data_Balanced十分之一.csv')
#
train_df, test_df = train_test_split(
        data_balanced, test_size=0.2, random_state=42, stratify=data_balanced['status']
    )
print(f"\n訓練集大小: {train_df.shape}, 測試集大小: {test_df.shape}")
train_df.to_csv('/home/aiuser/wy/csv/Combined_Data_Balanced十分之一_train.csv')
test_df.to_csv('/home/aiuser/wy/csv/Combined_Data_Balanced十分之一_test.csv')


