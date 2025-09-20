import emcee
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
import seaborn as sns
from datetime import datetime, timedelta
from multiprocessing import Pool
from sklearn.metrics import r2_score, mean_squared_error
from collections import defaultdict
from matplotlib import cm
from matplotlib import rcParams
from matplotlib import font_manager as fm
from matplotlib.font_manager import FontProperties


script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(script_dir)

def load_data():
    # 加载气象数据并进行插值处理
    weather = pd.read_excel("../日喀则气象站-逐日数据.xlsx")

    weather['日期'] = pd.to_datetime(weather['日期'])
    # 设置日期为索引
    weather.set_index('日期', inplace=True)
    
    # 处理异常值和缺失值
    weather['平均温度(℃)'] = weather['平均温度(℃)'].clip(-10, 40)
    weather['日照时长(小时)'] = weather['日照时长(小时)'].replace(0, np.nan).interpolate(method='time').bfill().ffill()
    weather['累计降雨量(mm)'] = weather['累计降雨量(mm)'].fillna(0)
    weather.to_excel('../weather_data.xlsx')

    # 加载试验数据
    trials = pd.read_excel("../日喀则2024青稞播期试验数据.xlsx")
    trials['日期'] = pd.to_datetime(trials['日期'])
    trials['播期'] = pd.to_datetime(trials['播期'])

    # 生育期映射与数据校验
    stage_mapping = {
        '出苗期': 'emergence',
        '分蘖期': 'tillering',
        '拔节期': 'jointing',
        '抽穗期': 'heading',
        '开花期': 'flowering',
        '灌浆期': 'grain_filling',
        '乳熟期': 'milking',
        '成熟期': 'maturity'
    }

    processed = []
    for (sowing_date, variety), group in trials.groupby(['播期', '品种']):
        stages = {}
        for _, row in group.iterrows():
            stage = stage_mapping.get(row['生育期'].strip())
            if stage:
                stages[stage] = row['日期']
        # 验证关键生育期数据
        if not {'emergence', 'heading', 'maturity'}.issubset(stages.keys()):
            print(f"警告：{sowing_date} {variety} 缺失关键生育期")
            continue
        processed.append({'播期': sowing_date, '品种': variety, **stages})
    
    pd.DataFrame(processed).to_excel('../obs_pheno_data.xlsx')

class EnhancedBarleyModel:
    def __init__(self):
        self.params = self.load_params_data()
        self.weather = pd.read_excel("../日喀则气象站-逐日数据.xlsx")

    def load_params_data(self):
        p_data = pd.read_excel("../parameters.xlsx")
        return p_data
    
    def temperature_response(self, T, params):
        """温度响应函数"""
        T_base = params['T_base']
        T_max = params['T_max']
        
        if T < T_base or T > T_max:
            return 0.0
        else:
            return T - T_base
    
    def photoperiod_response(self, DL, params):
        """基于阈值的日长响应函数"""
        critical_photoperiod = params['P_critical']
        sensitivity = params['P_sensitivity']
        DL = np.clip(DL, 8.0, 16.0)
        return 1 + sensitivity * (DL - critical_photoperiod) if DL > critical_photoperiod else 1.0
    
    def calculate_gdd(self, df, params):
        """结合温度和光周期响应的GDD累积计算"""
        gdd = 0.0
        cumulative = []
        for _, row in df.iterrows():
            T = row['平均温度(℃)']
            DL = row['日照时长(小时)']
            temp_resp = self.temperature_response(T, params)
            photo_resp = self.photoperiod_response(DL, params)
            daily_gdd = temp_resp * photo_resp
            gdd += daily_gdd
            cumulative.append(gdd)
        return np.array(cumulative)

    def predict_stages(self, cumulative_gdd, dates, thresholds, sowing_date):
        """根据累积GDD预测生育期"""
        stages = {}
        for stage, threshold in thresholds.items():
            idx = np.argmax(cumulative_gdd >= threshold)
            if idx > 0:
                stages[stage] = (dates.iloc[idx] - pd.to_datetime(sowing_date)).days
            else:
                stages[stage] = (dates.iloc[-1] - pd.to_datetime(sowing_date)).days  
        return stages
    
    def predict_phenology(self, variety, sowing_date):
        params = self.params[self.params['品种'] == variety]
        if params.empty:
            return {}

        params = params.iloc[0].to_dict()

        # 筛选播种后的气象数据
        df = self.weather[self.weather['日期'] >= sowing_date].copy()
        if df.empty:
            return {}

        # 计算带光周期调节的 GDD
        base_gdd = self.calculate_gdd(df, params)

        # 生育期GDD阈值
        stage_thresholds = {
            'emergence': params['GDD_emergence'],
            'tillering': params['GDD_tillering'],
            'jointing': params['GDD_jointing'],
            'heading': params['GDD_heading'],
            'flowering': params['GDD_flowering'],
            'grain_filling': params['GDD_grain_filling'],
            'milking': params['GDD_milking'],
            'maturity': params['GDD_maturity']
        }

        return self.predict_stages(base_gdd, df['日期'], stage_thresholds, sowing_date)


# 误差
def calculate_average_error(predicted, observed):
    errors = defaultdict(list)
    squared_errors = defaultdict(list)
    
    for sowing_date, stages in predicted.items():
        obs_row = observed[(observed['播期'] == sowing_date) & (observed['品种'] == stages['品种'])]

        if obs_row.empty:
            continue

        for stage in ['emergence', 'tillering', 'jointing', 'heading', 'flowering', 'grain_filling', 'milking', 'maturity']:
            pred = stages.get(stage)
            obs = obs_row[stage].values[0]
            if pd.notna(obs):
                delta = abs(pred - (pd.to_datetime(obs) - sowing_date).days)
                errors[stage].append(delta)
                squared_errors[stage].append(delta ** 2)

    # 计算 MAE 和 RMSE
    avg_errors = {stage: round(np.mean(vals), 1) for stage, vals in errors.items()}
    std_errors = {stage: round(np.std(vals), 1) for stage, vals in errors.items()}
    rmse_errors = {stage: round(np.sqrt(np.mean(sq_vals)), 1) for stage, sq_vals in squared_errors.items()}
    
    return avg_errors, std_errors, rmse_errors      

# 绘图
def get_color_gradient(base_color, n):
    """从给定base_color生成n个渐变色"""
    cmap = cm.get_cmap(base_color, n)
    return [cmap(i) for i in range(n)]

stages_en = ['emergence', 'tillering', 'jointing', 'heading', 'flowering', 'grain_filling', 'milking', 'maturity']
stages_cn = ['出苗', '分蘖', '拔节', '抽穗', '开花', '灌浆', '乳熟', '成熟']

mae_data = {
    '喜马拉22': pd.Series([2.5, 3.2, 5.0, 4.2, 6.0, 14.0, 9.7, 8.0], index=stages_en),
    '藏青2000': pd.Series([2.5, 3.5, 5.2, 4.5, 4.3, 13.0, 8.7, 6.8], index=stages_en),
    '藏青3000': pd.Series([2.5, 3.0, 4.8, 4.5, 5.0, 11.0, 11.0, 7.5], index=stages_en)
}

rmse_data = {
    '喜马拉22': pd.Series([3.2, 5.6, 6.7, 5.6, 7.5, 16.6, 15.1, 9.3], index=stages_en),
    '藏青2000': pd.Series([3.0, 5.2, 6.9, 5.9, 5.4, 13.6, 10.2, 11.3], index=stages_en),
    '藏青3000': pd.Series([3.0, 5.1, 6.7, 6.0, 6.8, 11.9, 11.8, 10.1], index=stages_en)
}

# 不同品种的同色系渐变色
gradient_colors = {
    '喜马拉22': plt.cm.Blues(np.linspace(0.4, 0.9, len(stages_cn))),
    '藏青2000': plt.cm.Greens(np.linspace(0.4, 0.9, len(stages_cn))),
    '藏青3000': plt.cm.Oranges(np.linspace(0.4, 0.9, len(stages_cn))),
}

# 图 (d)：统一颜色表示生育期
fixed_stage_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

def cm_to_inch(cm): return cm / 2.54
    
font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=10.5)  # 中文
times_new_roman = FontProperties(fname=r"C:\Windows\Fonts\times.ttf", size=10.5)  # 西文
times_style = {'fontproperties': font}

# 绘图函数（适配 MAE 或 RMSE）
def plot_individual_figures(metric_data, metric_name='RMSE'):
    # 每个品种一张图
    for variety in varieties:
        fig, ax = plt.subplots(figsize=(cm_to_inch(14.29), cm_to_inch(10.72)))

        colors = gradient_colors[variety]
        values = metric_data[variety].values
        ax.bar(stages_cn, values, color=colors, edgecolor='none')

        ax.set_title(f"{variety}", fontsize=10.5, fontproperties=times_new_roman)  
        ax.set_ylabel(metric_name, fontsize=10.5, fontproperties=times_new_roman)
        ax.set_ylim(0, max(values) * 1.25)

        ax.set_xticklabels(stages_cn, fontproperties=font)  
        ax.set_yticklabels([f"{y:.0f}" for y in ax.get_yticks()], fontproperties=times_new_roman)

        plt.tight_layout()
        plt.savefig(f"{metric_name}_{variety}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 第四张图：组合图，比较不同品种的所有生育期
    fig, ax_d = plt.subplots(figsize=(cm_to_inch(14.29), cm_to_inch(10.72)))
    num_stages = len(stages_en)
    bar_width = 0.2
    spacing = 0.05
    total_bar_width = num_stages * (bar_width + spacing)
    group_positions = np.arange(len(varieties)) * (total_bar_width + 0.2)

    for idx, stage in enumerate(stages_en):
        bar_pos = group_positions + idx * (bar_width + spacing)
        values = [metric_data[var][stage] for var in varieties]
        ax_d.bar(bar_pos, values, width=bar_width, color=fixed_stage_colors[idx],
                 edgecolor='none', label=stages_cn[idx])

    center_pos = group_positions + (total_bar_width - bar_width - spacing) / 2
    ax_d.set_xticks(center_pos)
    ax_d.set_xticklabels(varieties, fontsize=10.5, fontproperties=font)
    ax_d.set_ylabel(metric_name, fontsize=10.5, fontproperties=times_new_roman)

    legend = ax_d.legend(
        title="生育期", fontsize=10.5, title_fontsize=10.5,
        prop=font, bbox_to_anchor=(1.01, 1), loc='upper left'
    )
    legend.get_title().set_fontproperties(font)

    ax_d.set_yticklabels([f"{y:.0f}" for y in ax_d.get_yticks()], fontproperties=times_new_roman)

    plt.tight_layout()
    plt.savefig(f"{metric_name}_组合图.png", dpi=300, bbox_inches='tight')
    plt.close()
   

if __name__ == "__main__":
    test = EnhancedBarleyModel()
    observed_data = pd.read_excel('../obs_pheno_data.xlsx')
    varieties = observed_data['品种'].unique()
    all_errors = []
    all_mae_errors = []  
    all_rmse_errors = []  
    
    for variety in varieties:
        print(f"\n================ 品种 {variety} ================")
        sowing_dates = observed_data[observed_data['品种'] == variety]['播期'].dropna().unique()
        predicted_stages = {}

        for sowing_date in sowing_dates:
            sowing_date = pd.to_datetime(sowing_date)
            stages = test.predict_phenology(variety, sowing_date.strftime('%m/%d/%Y'))
            stages['品种'] = variety  
            predicted_stages[sowing_date] = stages
            print(f"播期 {sowing_date.date()} 模拟结果: {stages}")

        avg_error, std_error, rmse_error = calculate_average_error(predicted_stages, observed_data)
        for stage, err in avg_error.items():
            print(f"  {stage} - 平均绝对误差: {err} 天")
        for stage, rmse in rmse_error.items():
            print(f"  {stage} - 均方根误差: {rmse} 天")

        # 保存 MAE 数据
        mae_row = {'品种': variety}
        mae_row.update(avg_error)  # 平均绝对误差
        all_mae_errors.append(mae_row)

        # 保存 RMSE 数据
        rmse_row = {'品种': variety}
        rmse_row.update(rmse_error)  # 均方根误差
        all_rmse_errors.append(rmse_row)

    # 保存 MAE 数据到 CSV 文件
    mae_df = pd.DataFrame(all_mae_errors)
    mae_df.to_csv("各品种生育期平均绝对误差.csv", index=False, encoding='utf-8-sig')
    print("平均绝对误差表已保存为 '各品种生育期平均绝对误差.csv'")

    # 保存 RMSE 数据到 CSV 文件
    rmse_df = pd.DataFrame(all_rmse_errors)
    rmse_df.to_csv("各品种生育期均方根误差.csv", index=False, encoding='utf-8-sig')
    print("均方根误差表已保存为 '各品种生育期均方根误差.csv'")

    # 绘制误差图
    plot_individual_figures(mae_data, metric_name='MAE')
    plot_individual_figures(rmse_data, metric_name='RMSE')
