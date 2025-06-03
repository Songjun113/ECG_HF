import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.signal import butter, filtfilt, find_peaks
import random
from collections import Counter

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 参数设置
fs = 1000  # 采样率
window_size = int(0.4 * fs)  # 400ms 对称窗口（前后各200ms）
hf_cut = 100  # 保持100Hz高通滤波阈值
max_beats_per_person = 100  # 每位患者最多心搏数量
rpeak_detection_leads = ['II', 'V5', 'V2']  # 用于检测R波的多个导联

# 路径设置
base_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
input_folder = os.path.join(base_dir, "csv格式数据")
output_folder = os.path.join(base_dir, "preprocessed_beats_100hz_highpass")

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# 导联名称
lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
log_data = []


# 高通滤波函数（保持100Hz）
def highpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='high')
    return filtfilt(b, a, data)


# 改进的多导联R波检测函数
def detect_r_peaks_multilead(df, detection_leads, fs):
    all_peaks = []
    lead_peaks = {}  # 存储每个导联检测到的R波

    for lead in detection_leads:
        try:
            signal = df[lead].values
            signal = signal - np.mean(signal)
            signal = signal / (np.max(np.abs(signal)) + 1e-6)  # 避免除以零

            # 检测R波 - 调整参数以适应100Hz滤波后的信号
            peaks, properties = find_peaks(
                signal,
                distance=int(0.5 * fs),  # 最小R-R间期500ms (120bpm)
                prominence=0.3,  # 降低 prominence 阈值
                width=int(0.05 * fs)  # QRS波宽度约50ms
            )
            lead_peaks[lead] = peaks
            all_peaks.extend(peaks)
        except Exception as e:
            print(f"⚠ 导联 {lead} R波检测出错: {str(e)}")
            continue

    # 如果没有检测到任何R波，返回空数组
    if not all_peaks:
        return np.array([])

    # 统计所有导联检测到的R波位置
    peak_counter = Counter(all_peaks)

    # 设置R波确认阈值（至少被N/2个导联检测到，N>=2）
    min_detections = max(1, len(detection_leads) // 2)
    confirmed_peaks = [peak for peak, count in peak_counter.items() if count >= min_detections]

    # 对确认的R波进行聚类，消除邻近重复检测
    final_peaks = []
    if confirmed_peaks:
        confirmed_peaks_sorted = sorted(confirmed_peaks)
        final_peaks.append(confirmed_peaks_sorted[0])

        for peak in confirmed_peaks_sorted[1:]:
            if peak - final_peaks[-1] > int(0.3 * fs):  # 最小间隔300ms
                final_peaks.append(peak)

    return np.array(final_peaks)


# 遍历主文件夹及其子文件夹中的所有 CSV 文件
input_files = []
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith('.csv'):
            input_files.append(os.path.join(root, file))

for file in input_files:
    name = os.path.basename(file)
    print(f"\n▶ 正在处理: {name}")

    try:
        df = pd.read_csv(os.path.join(input_folder, file))

        # 检查导联完整性
        missing_leads = [lead for lead in lead_names if lead not in df.columns]
        if missing_leads:
            print(f"⚠ 缺少导联: {missing_leads}, 跳过")
            log_data.append([name, f'缺少导联{missing_leads}', '未处理'])
            continue

        # 多导联R波检测
        r_peaks = detect_r_peaks_multilead(df, rpeak_detection_leads, fs)
        print(f"✔ 检测到 R 波数量: {len(r_peaks)} (使用导联: {rpeak_detection_leads})")

        if len(r_peaks) < 10:
            print("⚠ R波数量过少，跳过")
            log_data.append([name, 'R波数量过少', '未处理'])
            continue

        # 随机选取最多100个R波
        if len(r_peaks) > max_beats_per_person:
            r_peaks = sorted(random.sample(list(r_peaks), max_beats_per_person))

        # 每个导联滤波并提取心搏（对称窗口）
        beat_dict = {lead: [] for lead in lead_names}
        for lead in lead_names:
            raw_signal = df[lead].values
            filtered_signal = highpass_filter(raw_signal, hf_cut, fs)

            for r in r_peaks:
                start = r - window_size // 2  # 对称窗口，前后各200ms
                end = r + window_size // 2
                if start >= 0 and end <= len(filtered_signal):
                    beat_dict[lead].append(filtered_signal[start:end])

        # 汇总导联数据，确认是否有效
        beat_arrays = []
        valid_leads = []
        for lead in lead_names:
            beats = beat_dict[lead]
            if len(beats) >= 5:  # 至少5个有效心搏
                beat_arrays.append(np.stack(beats))  # shape: (n_beats, window)
                valid_leads.append(lead)

        if not beat_arrays:
            print("⚠ 无有效导联，跳过")
            log_data.append([name, '无有效导联', '未处理'])
            continue

        # 合并为 (n_beats, window, n_leads)
        beat_matrix = np.stack(beat_arrays, axis=-1)
        n_beats = beat_matrix.shape[0]
        time_ms = np.arange(window_size) / fs * 1000 - (window_size // 2)  # 时间轴以R波为中心(0ms)

        # 保存每个beat为单独csv
        for i in range(n_beats):
            beat_df = pd.DataFrame(beat_matrix[i], columns=valid_leads)
            beat_df.insert(0, 'Time(ms)', time_ms)
            out_file = os.path.join(output_folder, f"{os.path.splitext(name)[0]}_beat_{i + 1}.csv")
            beat_df.to_csv(out_file, index=False)

        print(f"✅ 已保存 {n_beats} 个心搏片段至：{output_folder}")
        log_data.append([name, f'成功提取 {n_beats} 个片段', '已处理'])

    except Exception as e:
        print(f"❌ 处理 {name} 时出错: {str(e)}")
        log_data.append([name, f'处理错误: {str(e)}', '未处理'])

# 保存日志
log_df = pd.DataFrame(log_data, columns=['文件名', '状态', '处理结果'])
log_df.to_csv(os.path.join(output_folder, 'processing_log.csv'), index=False, encoding='utf_8_sig')