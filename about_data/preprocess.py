############preprocess##########
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.signal import butter, filtfilt, find_peaks

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 设置为支持中文的字体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置参数
fs = 1000  # 采样率
window = int(0.4 * fs)  # 每个心搏窗口长度（400ms）；待定！！！
lead_for_rpeak = "II"  # 用于R波检测的导联
hf_band = (150, 250)  # 高频带通滤波范围

# 患者数据路径列表
xml_filepath = "12leads_data/"   # 存放所有心电xml文件
output_path = "preprocessed_avg_beat"     # 输出csv文件夹

# 创建输出目录（如果不存在）
if not os.path.exists(output_path):
    os.makedirs(output_path)

input_files = [f for f in os.listdir(xml_filepath) if f.endswith('.csv')]  # 只处理CSV文件

# 滤波器设计函数
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)

# R波检测函数
def detect_r_peaks(signal, fs):
    signal = signal - np.mean(signal)
    signal = signal / np.max(np.abs(signal))
    peaks, _ = find_peaks(signal, distance=int(0.4 * fs), prominence=0.5)
    return peaks

# 记录每个患者处理状态的列表
log_data = []

# 遍历处理
for file in input_files:
    name = os.path.basename(file)
    print(f"\n正在处理: {name}")

    try:
        # 读取CSV文件
        df = pd.read_csv(os.path.join(xml_filepath, file))

        # 检查导联
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        missing_leads = [lead for lead in lead_names if lead not in df.columns]
        
        if missing_leads:
            print(f"缺少导联: {missing_leads}, 跳过")
            log_data.append([name, f'缺少导联{missing_leads}', '未处理'])
            continue

        # R波检测
        signal_r = df[lead_for_rpeak].values
        r_peaks = detect_r_peaks(signal_r, fs)
        print(f"检测到 R 波数量: {len(r_peaks)}")

        if len(r_peaks) < 10:
            print("R 波数量太少，可能不稳定，跳过")
            log_data.append([name, 'R波数量太少', '未处理'])
            continue

        # 1. 对12导联分别进行滤波和叠加
        all_beats = []  # 存储所有导联的beat片段

        for lead in lead_names:
            # 对每个导联进行带通滤波
            lead_signal = bandpass_filter(df[lead].values, *hf_band, fs)
    
            # 截取每个心搏片段
            lead_beats = []
            for r in r_peaks:
                start = r - window // 3
                end = r + window * 2 // 3
                if start >= 0 and end <= len(lead_signal):
                    lead_beats.append(lead_signal[start:end])
    
            all_beats.append(lead_beats)

        # 2. 计算各导联的平均心搏（相干平均）
        avg_beats = []
        valid_lead_names = []
        for lead, beats in zip(lead_names, all_beats):
            if len(beats) >= 5:
                avg_beats.append(np.mean(beats, axis=0))
                valid_lead_names.append(lead)

        if len(avg_beats) < 12:
            print(f"警告：只有 {len(avg_beats)} 个导联有足够心搏")
    
        # 3. 将有效导联平均心搏堆叠为矩阵
        stacked_beats = np.column_stack(avg_beats)

        # 4. 保存为CSV
        time_ms = np.arange(stacked_beats.shape[0]) / fs * 1000
        output_df = pd.DataFrame(stacked_beats, columns=valid_lead_names)
        output_df.insert(0, 'Time(ms)', time_ms)
        
        # 修正保存路径
        output_filename = f"{os.path.splitext(name)[0]}.csv"
        full_output_path = os.path.join(output_path, output_filename)
        output_df.to_csv(full_output_path, index=False)
        print(f"已保存到: {full_output_path}")
        # 在原有代码的循环内添加以下内容（放在保存CSV之后）
        """
        # 5. 导出V5导联图像
        if 'V5' in valid_lead_names:
            v5_index = valid_lead_names.index('V5')
            v5_signal = avg_beats[v5_index]  # 获取V5导联平均心搏
    
            plt.figure(figsize=(12, 6))
            plt.plot(time_ms, v5_signal, 'b-', linewidth=2, label='V5导联平均心搏')
    
            # 标记关键点
            r_peak_pos = np.argmax(v5_signal)  # 假设R波在窗口内最大值
            plt.plot(time_ms[r_peak_pos], v5_signal[r_peak_pos], 'ro', label='R波峰值')
    
            # 标记ST段（R波后80-120ms）
            st_start = r_peak_pos + int(0.08 * fs)
            st_end = r_peak_pos + int(0.12 * fs)
            plt.plot(time_ms[st_start:st_end], v5_signal[st_start:st_end], 
                'g-', linewidth=3, label='ST段')
    
            # 图像修饰
            plt.title(f'{os.path.splitext(name)[0]} - V5导联高频成分', fontsize=14)
            plt.xlabel('时间 (ms)', fontsize=12)
            plt.ylabel('振幅 (μV)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=10)
    
            # 保存图像
            img_filename = f"{os.path.splitext(name)[0]}_V5_waveform.png"
            img_path = os.path.join(output_path, img_filename)
            plt.savefig(img_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"已保存V5导联图像到: {img_path}")
        """
    except Exception as e:
        print(f"处理{name}时出错: {str(e)}")
        log_data.append([name, f'处理错误: {str(e)}', '未处理'])

# 保存处理日志
log_df = pd.DataFrame(log_data, columns=['文件名', '状态', '处理结果'])
log_df.to_csv(os.path.join(output_path, 'processing_log.csv'), 
             index=False, 
             encoding='utf_8_sig')