import os
import pandas as pd

# 指定目录路径
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

directory = '../data/AndMal2020-Dynamic-BeforeAndAfterReboot'

# 存储所有CSV文件数据的列表
merge_data = pd.DataFrame()

# 遍历目录下的所有文件
for filename in os.listdir(directory):
    if filename.endswith('reboot_Cat.csv'):
        file_path = os.path.join(directory, filename)
        # 使用Pandas的read_csv函数读取CSV文件


        df = pd.read_csv(file_path)
        memory_columns = df.filter(like='Memory')
        # 使用shape属性获取行数和列数
        num_rows, num_columns = memory_columns.shape

        if 'before_reboot_Cat' in filename:
            memory_columns['Class'] = ['Benign' for i in range(num_rows)]
            memory_columns['Category'] = ['Benign' for i in range(num_rows)]
        if 'after_reboot_Cat' in filename:
            memory_columns['Class'] = ['Malware' for i in range(num_rows)]
            Category = filename.split('_after_')[0]
            memory_columns['Category'] = [Category for i in range(num_rows)]


        merge_data = pd.concat([merge_data, memory_columns], axis=0)
        # csv_data.append(df)
print(merge_data)


# 将DataFrame保存为CSV文件
merge_data.to_csv('../data/AndMal2020.csv', index=False)  # index=False表示不保存行索引
#
# # 合并所有CSV文件的数据
# combined_data = pd.concat(csv_data, ignore_index=True)
#
# combined_data = combined_data.drop('Hash', axis=1)
# combined_data = combined_data.drop('Family', axis=1)
#
# # 创建一个LabelEncoder对象
# le = LabelEncoder()
# # 对标签进行编码
# combined_data['Category'] = le.fit_transform(combined_data['Category'])
#
# combined_data_col = combined_data.columns
# combined_data_col = combined_data_col.drop('Category')
# scaler = MinMaxScaler()
# combined_data[combined_data_col] = scaler.fit_transform(combined_data[combined_data_col])
#
# print(f"Number of occurrences for each malware class: \n{combined_data['Category'].value_counts()}")
#
#
# combined_data.to_csv('./andMal.csv')
# 使用SMOTE算法对数据进行平衡处理
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(combined_data[combined_data_col], combined_data['Category'])
# print(f"Number of occurrences for each malware class after balancing: \n{y_resampled.value_counts()}")
# df = pd.DataFrame(X_resampled, y_resampled)
# df.to_csv('./andMal.csv')