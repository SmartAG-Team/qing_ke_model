import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
from phen_model import EnhancedBarleyModel
from matplotlib.font_manager import FontProperties

class SowingWindowModel:
    def __init__(self, t_freeze, t_grain_threshold):
        self.model = EnhancedBarleyModel()
        self.weather = self.model.weather.copy()  
        self.weather['日期'] = pd.to_datetime(self.weather['日期'])  
        self.t_freeze = t_freeze
        self.t_grain_threshold = t_grain_threshold

    def compute_freeze_risk_index(self, sowing_date, tillering_days):
        """
        从播种日至分蘖日前，统计最低温度低于 t_freeze 的天数
        """
        start_date = pd.to_datetime(sowing_date)
        end_date = start_date + pd.Timedelta(days=tillering_days)

        window = self.weather[(self.weather['日期'] >= start_date) & 
                            (self.weather['日期'] < end_date)]

        if window.empty or '最低温度(℃)' not in window.columns:
            return None

        window = window.dropna(subset=['最低温度(℃)'])
        risk_days = (window['最低温度(℃)'] < self.t_freeze).sum()

        return int(risk_days)


    def check_grain_filling_freeze(self, sowing_date, grain_days, milking_days):
        """
        判断从灌浆期到成熟期间是否有气温低于 t_grain_threshold，若有则视为不能成熟
        """
        grain_start = pd.to_datetime(sowing_date) + pd.Timedelta(days=grain_days)
        grain_end = pd.to_datetime(sowing_date) + pd.Timedelta(days=milking_days)

        window = self.weather[(self.weather['日期'] >= grain_start) &
                              (self.weather['日期'] <= grain_end)]

        if window.empty or '最低温度(℃)' not in window:
            return None

        # 若灌浆期内任一日低于阈值，则不能成熟
        if (window['最低温度(℃)'] < self.t_grain_threshold).any():
            return '否'
        return '是'

    def simulate_variety_sowing_range(self, variety, start_date, end_date):
        date_range = pd.date_range(start=start_date, end=end_date, freq='1D')
        records = []

        for sowing_date in date_range:
            sowing_str = sowing_date.strftime('%m/%d/%Y')
            phen = self.model.predict_phenology(variety, sowing_str)

            if not phen:
                continue
            emergence_days = phen.get('emergence', np.nan)
            tillering_days = phen.get('tillering', np.nan)
            grain_days = phen.get('grain_filling', np.nan)
            maturity_days = phen.get('maturity', np.nan)

            if any(pd.isna(v) for v in [emergence_days, tillering_days, grain_days, maturity_days]):
                continue

            freeze_index = self.compute_freeze_risk_index(sowing_date, tillering_days)
            if freeze_index is None:
                continue

            mature_flag = self.check_grain_filling_freeze(sowing_date, grain_days, maturity_days)
            if mature_flag is None:
                continue

            records.append({
                '品种': variety,
                '播期': sowing_date.strftime('%Y-%m-%d'),
                '出苗期 (天)': emergence_days,
                '成熟期 (天)': maturity_days,
                '冻害风险天数': freeze_index,
                '能否成熟': mature_flag
            })

        df = pd.DataFrame(records)
        if '播期' in df.columns:
            return df.sort_values('播期')
        else:
            return df

    def recommend_best_dates(self, df):
        """
        推荐冻害风险最小、能成熟的播期
        """
        if df.empty:
            return pd.DataFrame()

        df_valid = df[df['能否成熟'] == '是']
        if df_valid.empty:
            return pd.DataFrame()

        min_risk = df_valid['冻害风险天数'].min()
        best_df = df_valid[df_valid['冻害风险天数'] == min_risk]
        return best_df
    
    def recommend_optimal_range(self, df):
        """
        综合考虑冻害风险最低 + 生育期最长的最优播期范围
        """
        if df.empty or '能否成熟' not in df.columns:
            return pd.DataFrame()

        df_valid = df[df['能否成熟'] == '是']
        if df_valid.empty:
            return pd.DataFrame()

        min_risk = df_valid['冻害风险天数'].min()
        df_risk = df_valid[df_valid['冻害风险天数'] == min_risk]

        max_duration = df_risk['成熟期 (天)'].max()
        optimal_range = df_risk[df_risk['成熟期 (天)'] == max_duration]

        return optimal_range

def cm_to_inch(cm): return cm / 2.54
    
font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=10.5)  # 中文
times_new_roman = FontProperties(fname=r"C:\Windows\Fonts\times.ttf", size=10.5)  # 西文
times_style = {'fontproperties': font}

def plot_maturity_and_risk(df, variety):
    """
    绘制播期-成熟期 + 冻害风险指数的双轴图（优化版本）
    """
    if df.empty or '播期' not in df.columns:
        print(f"{variety}：数据为空或缺少 '播期' 列，无法绘图")
        return

    df_plot = df[df['能否成熟'] == '是'].copy()
    df_plot['播期'] = pd.to_datetime(df_plot['播期'])

    fig, ax1 = plt.subplots(figsize=(cm_to_inch(16), cm_to_inch(10)))

    # 左轴：成熟期（蓝色）
    ax1.set_xlabel('播种日期', fontsize=10.5, fontproperties=font)
    ax1.set_ylabel('成熟期（d）', color='tab:blue', fontsize=10.5, fontproperties=font)
    l1, = ax1.plot(df_plot['播期'], df_plot['成熟期 (天)'], color='tab:blue', marker='o', markersize=4, label='成熟期')
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=10.5)
    for label in ax1.get_yticklabels():
        label.set_fontproperties(times_new_roman)
    ax1.tick_params(axis='x', labelsize=10.5)
    for label in ax1.get_xticklabels():
        label.set_fontproperties(times_new_roman)
    ax1.set_ylim(113, 136) 

    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    # 右轴：冻害风险天数（红色）
    ax2 = ax1.twinx()
    ax2.set_ylabel('冻害风险天数（d）', color='tab:red', fontsize=10.5, fontproperties=font)
    l2, = ax2.plot(df_plot['播期'], df_plot['冻害风险天数'], color='tab:red', marker='s', markersize=4, linestyle='--', label='冻害风险')
    ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=10.5)
    for label in ax2.get_yticklabels():
        label.set_fontproperties(times_new_roman)
    ax2.set_ylim(0, 4) 

    # 图例：整合双轴图例
    lines = [l1, l2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper center', fontsize=10.5, ncol=2, prop=font)

    fig.tight_layout()

    # 保存图像
    img_path = f'{variety}_播期_成熟期_冻害图.png'
    plt.savefig(img_path, dpi=300)
    plt.close()
    print(f"双轴图已保存：{img_path}")


if __name__ == "__main__":
    model = SowingWindowModel(
        t_freeze=-1,
        t_grain_threshold=-2
    )

    variety = '喜马拉22'
    start = '2024-05-01'
    end = '2024-06-30'

    print(f"\n正在模拟：{variety} 在 {start} 到 {end} 间的播期……")
    df_all = model.simulate_variety_sowing_range(variety, start, end)

    df_all.to_excel('df_all.xlsx')

    if not df_all.empty:
        output_file = f'{variety}_播期模拟结果.xlsx'
        df_all.to_excel(output_file, index=False)
        print(f"模拟完成，结果保存为 {output_file}")

        best_df = model.recommend_best_dates(df_all)
        print("\n推荐最佳播期（冻害风险最小，且能成熟）：")
        print(best_df)
        
        optimal_range_df = model.recommend_optimal_range(df_all)
        print("\n推荐最优播期范围（能成熟 & 冻害风险最小 & 生育期最长）：")
        print(optimal_range_df)
    else:
        print("无有效模拟结果，请检查模型或数据。")

    plot_maturity_and_risk(df_all, variety)
