import pandas as pd

# 读取Excel文件
file_path = r'C:\Users\ThinkPad\Desktop\中化\2、数据资料\气象数据统计\日喀则气象站-采集数据.xlsx'
df = pd.read_excel(file_path)

# 转换时间列为datetime类型
df['collect_time'] = pd.to_datetime(df['collect_time'])

# 创建日期列
df['date'] = df['collect_time'].dt.date

# 定义聚合函数
agg_functions = {
    'temperature': ['mean', 'min'],
    'solar_radiation': 'mean',
    'wind_speed': 'mean',
    'rainfall': 'sum',
}

# 计算日长（日照时间）
def calculate_daylight(series):
    return (series > 0).sum() * 5 / 60  # 转换为小时

# 分组计算每日数据
daily_df = df.groupby('date').agg(agg_functions)
daily_df['daylight_hours'] = df.groupby('date')['solar_radiation'].apply(calculate_daylight)

# 重置索引并重命名列
daily_df = daily_df.reset_index()
daily_df.columns = [
    '日期',
    '平均温度(℃)',
    '最低温度(℃)',
    '平均太阳辐射(W/m²)',
    '平均风速(m/s)',
    '累计降雨量(mm)',
    '日照时长(小时)'
]

# 保存结果
output_path = r'C:\Users\ThinkPad\Desktop\中化\2、数据资料\气象数据统计\日喀则气象站-逐日数据.xlsx'
daily_df.to_excel(output_path, index=False)

print("数据处理完成，结果已保存至:", output_path)