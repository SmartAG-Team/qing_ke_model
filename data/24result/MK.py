import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.stats import norm

# 设置字体：宋体（中文）+ Times New Roman（默认西文）
simsun_font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=10.5)

file_path = r'E:\Workshop\highland barley\日喀则气象站-逐日数据.xlsx'
date_col = '日期'

# 读取 Excel 并按日期排序
df = pd.read_excel(file_path, parse_dates=[date_col])
df.sort_values(date_col, inplace=True)

# 自动检测主要参数列
keywords = ['温度', '辐射', '降雨', '日照']
parameters = [col for col in df.columns if any(k in col for k in keywords)]

# 定义 Mann-Kendall 检验函数
def mann_kendall_test(data, alpha=0.05):
    n = len(data)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(data[j] - data[i])
    
    unique_x, counts = np.unique(data, return_counts=True)
    tie_sum = sum(c * (c - 1) * (2 * c + 5) for c in counts if c > 1)
    
    var = (n * (n - 1) * (2 * n + 5) - tie_sum) / 18
    if var == 0:
        return 0, 1.0, 'no trend'
    
    z = (s - 1) / np.sqrt(var) if s > 0 else (s + 1) / np.sqrt(var) if s < 0 else 0
    p = 2 * (1 - norm.cdf(abs(z)))
    trend = 'increasing' if (p <= alpha and z > 0) else 'decreasing' if (p <= alpha and z < 0) else 'no trend'
    
    return z, p, trend

# 执行 MK 检验
results = {}
for param in parameters:
    series = df[param].values
    z, p, trend = mann_kendall_test(series)
    results[param] = {'Z': z, 'p': p, 'trend': trend}

# 设置输出目录
output_dir = r'E:\Workshop\highland barley\codes\output'
os.makedirs(output_dir, exist_ok=True)

# 厘米转英寸
def cm_to_inch(cm): return cm / 2.54

# 绘图
for param in parameters:
    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(10)))  

    # 原始数据曲线
    plt.plot(df[date_col], df[param], label='', color='steelblue')

    title_info = (
        f"Mann-Kendall Test: {param}\n"
        f"Z = {results[param]['Z']:.2f}, "
        f"p = {results[param]['p']:.3f}\n"
        f"Trend: {results[param]['trend']}"
    )
    plt.title(title_info, fontproperties=simsun_font)
    plt.xlabel("日期", fontproperties=simsun_font)
    plt.ylabel(param, fontproperties=simsun_font)

    # 日期横坐标旋转横向
    plt.xticks(rotation=0, fontproperties=simsun_font)
    plt.yticks(fontproperties=simsun_font)
    plt.grid(alpha=0.3)

    # 安全处理文件名
    safe_param = param.translate(str.maketrans({
        "\\": "_", "/": "_", ":": "_", "*": "_",
        "?": "_", "\"": "_", "<": "_", ">": "_", "|": "_"
    }))

    save_path = os.path.join(output_dir, f'{safe_param}_MK_test.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

print("✅ Mann-Kendall 检验与可视化完成，图像已保存至：", output_dir)
