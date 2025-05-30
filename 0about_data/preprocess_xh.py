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
window = int(0.6 * fs)  # 每个心搏窗口长度（600ms）
lead_for_rpeak = "II"  # 用于R波检测的导联
hf_band = (150, 250)  # 高频带通滤波范围

# 患者数据路径列表
input_files = [
    r"D:\professional_design\xml_csv\韩玉先-20230719-170248.csv",
    r"D:\professional_design\xml_csv\黄刚志-20231119-223130.csv",
    r"D:\professional_design\xml_csv\蒋小桃-20230721-154745.csv",
    r"D:\professional_design\xml_csv\李丙印-20231115-221723.csv",
    r"D:\professional_design\xml_csv\李传光-20230310-155623.csv",
    r"D:\professional_design\xml_csv\李跃武-20231115-213925.csv",
    r"D:\professional_design\xml_csv\刘玉贤-20230719-155458.csv",
    r"D:\professional_design\xml_csv\聂大玉-20230719-152103.csv",
    r"D:\professional_design\xml_csv\石皖枝-20230721-160554.csv",
    r"D:\professional_design\xml_csv\熊传河-20231026-101410.csv",
    r"D:\professional_design\xml_csv\徐长胜-20230719-164533.csv",
    r"D:\professional_design\xml_csv\杨传辉-20230719-150303.csv",
    r"D:\professional_design\xml_csv\叶张·鹏-20231112-222016.csv",
    r"D:\professional_design\xml_csv\张永英-20231215-112549.csv",
    r"D:\professional_design\xml_csv\赵立芳-20230308-171340.csv",
    r"D:\professional_design\xml_csv\朱先芳-20230308-165437.csv",
]

# 滤波器设计函数
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)

# R波检测函数
def detect_r_peaks(signal, fs):
    signal = signal - np.mean(signal)
    signal = signal / np.max(np.abs(signal))
    peaks, _ = find_peaks(signal, distance=int(0.6 * fs), prominence=0.5)
    return peaks

# 记录每个患者处理状态的列表
log_data = []

# 遍历处理
for file in input_files:
    name = os.path.basename(file)
    print(f"\n📁 正在处理: {name}")

    try:
        df = pd.read_csv(file)

        # 检查导联
        if lead_for_rpeak not in df.columns:
            print(f"❌ 缺少用于 R 波检测的导联 '{lead_for_rpeak}'，跳过")
            log_data.append([name, '缺少R波检测导联', '未处理'])
            continue
        if "V5" not in df.columns:
            print(f"❌ 缺少高频分析所需的导联 'V5'，跳过")
            log_data.append([name, '缺少V5导联', '未处理'])
            continue

        signal_r = df[lead_for_rpeak].values
        r_peaks = detect_r_peaks(signal_r, fs)
        print(f"✅ 检测到 R 波数量: {len(r_peaks)}")

        if len(r_peaks) < 10:
            print("⚠️ R 波数量太少，可能不稳定，跳过")
            log_data.append([name, 'R波数量太少', '未处理'])
            continue

        hf_signal = bandpass_filter(df["V5"].values, *hf_band, fs)
        beats = []
        for r in r_peaks:
            start = r - window // 3
            end = r + window * 2 // 3
            if start < 0 or end > len(hf_signal):
                continue
            beats.append(hf_signal[start:end])

        if len(beats) < 5:
            print("⚠️ 有效心搏片段太少，跳过")
            log_data.append([name, '有效心搏片段太少', '未处理'])
            continue

        avg_beat = np.mean(beats, axis=0)
        time = np.arange(len(avg_beat)) / fs * 1000  # 单位：ms

        # 画图保存
        plt.figure(figsize=(8, 4))
        plt.plot(time, avg_beat, label='平均高频心搏', color='blue')
        plt.title(f"{name} 高频心搏平均波形")
        plt.xlabel("时间 (ms)")
        plt.ylabel("电压 (μV)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"avg_beat_{name}.png", dpi=300)
        plt.close()
        print("✅ 已生成图像")

        # 保存为 .npy 文件
        npy_name = f"avg_beat_{os.path.splitext(name)[0]}.npy"
        np.save(npy_name, avg_beat)
        print(f"✅ 已保存 npy 文件: {npy_name}")

        # 记录日志
        log_data.append([name, '处理成功', '已保存图像和npy'])

    except Exception as e:
        print(f"❌ 处理异常: {e}")
        log_data.append([name, f'处理异常: {e}', '未处理'])

# 将日志数据保存为 .csv 文件
log_df = pd.DataFrame(log_data, columns=['患者', '状态', '结果'])
log_df.to_csv("处理日志.csv", index=False, encoding='utf-8-sig')

print("\n✅ 所有处理完成，日志已保存！")
