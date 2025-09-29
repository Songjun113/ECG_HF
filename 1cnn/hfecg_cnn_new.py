import sys
print("Python version:", sys.version)
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
from glob import glob

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    LearningRateScheduler
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit

# ====================== 结果保存：基于时间戳的子目录 ======================
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join("results", timestamp)
os.makedirs(results_dir, exist_ok=True)
summary_path = os.path.join(results_dir, "training_summary.txt")

# ====================== 数据加载与预处理函数（新） ======================
def preprocess_data(all_files, labels_df):
    """
    接收所有文件路径及标签表，返回归一化后的 ECG 样本、标签编码及患者 ID 列表
    all_files: 包含“*_beat_*.csv”格式文件的完整路径列表
    labels_df: 包含 name 与 labels 列的 DataFrame
    """
    ecg_samples = []
    labels = []
    patient_ids = []

    # 构建姓名到标签的映射，避免循环内频繁检索
    name_to_label = dict(zip(labels_df['name'], labels_df['labels']))

    for file_path in all_files:
        base_name = os.path.basename(file_path)
        # 文件名格式示例：ID123_beat_001.csv，从“_beat_”前截取为患者 ID
        name = base_name.split('_beat_')[0]

        if name not in name_to_label:
            # 若无对应标签则跳过
            continue

        try:
            df = pd.read_csv(file_path)
            # 提取第 2~13 列（12 导联信号）
            ecg_data = df.iloc[:, 1:13].values
            # 归一化到 [-1,1]
            max_val = np.max(np.abs(ecg_data))
            ecg_data = ecg_data / (max_val if max_val != 0 else 1.0)

            ecg_samples.append(ecg_data)
            labels.append(name_to_label[name])
            patient_ids.append(name)

        except FileNotFoundError:
            print(f"Warning: File not found - {file_path}")
            continue

    # 将字符串标签编码为整数（0/1）
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    return np.array(ecg_samples), np.array(labels), np.array(patient_ids)

# ====================== 模型构建函数（同原有） ======================
def build_cnn_model(input_shape):
    model = models.Sequential([
        layers.Conv1D(32, 5, padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.2),

        layers.Conv1D(64, 5, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(2),

        layers.Conv1D(128, 3, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.GlobalAveragePooling1D(),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# ====================== 可视化函数（同原有） ======================
def plot_history(history, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history.history['accuracy'], label='Train Acc')
    axes[0].plot(history.history['val_accuracy'], label='Val Acc')
    axes[0].set_xlabel('Epochs'); axes[0].set_ylabel('Accuracy'); axes[0].legend()
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_xlabel('Epochs'); axes[1].set_ylabel('Loss'); axes[1].legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "training_history.png")
    plt.savefig(path); plt.close()
    print(f"训练曲线图已保存至 {path}")

def plot_learning_rate(history, save_dir):
    if 'lr' not in history.history:
        print("Warning: history 中不包含 'lr'，请添加 LearningRateScheduler 回调")
        return
    epochs = range(1, len(history.history['lr']) + 1)
    plt.figure(figsize=(6,4))
    plt.plot(epochs, history.history['lr'], label='Learning Rate')
    plt.xlabel('Epochs'); plt.ylabel('Learning Rate'); plt.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "learning_rate.png")
    plt.savefig(path); plt.close()
    print(f"学习率变化图已保存至 {path}")

# ====================== 主流程 ======================
def main():
    # 1. 加载标签表
    labels_df = pd.read_csv(
        "labels.csv",
        names=['name', 'labels'],
        header=0,
        encoding='utf-8'
    )

    # 2. 获取所有 ECG 文件路径并预处理
    data_dir = "../preprocessed_data"
    all_files = glob(f"{data_dir}/*_beat_*.csv")
    if not all_files:
        raise ValueError(f"No ECG files found in {data_dir}")

    X, y, patient_ids = preprocess_data(all_files, labels_df)
    print(f"原始数据: 样本数={X.shape[0]}, 正样本数={np.sum(y==1)}, 负样本数={np.sum(y==0)}")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"原始样本: X={X.shape}, y={y.shape}\n")
        f.write(f"标签分布: {{0: {np.sum(y==0)}, 1: {np.sum(y==1)}}}\n")

    # 3. 患者级分层划分（确保同一患者样本不跨训练/验证集）
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=88)
    train_idx, val_idx = next(gss.split(X, y, groups=patient_ids))
    X_train, X_test = X[train_idx], X[val_idx]
    y_train, y_test = y[train_idx], y[val_idx]
    with open(summary_path, 'a', encoding='utf-8') as f:
        f.write(f"训练集: X_train={X_train.shape}, 测试集: X_test={X_test.shape}\n")

    # 4. 构建并记录模型结构
    input_shape = X_train.shape[1:]
    model = build_cnn_model(input_shape)
    with open(summary_path, 'a', encoding='utf-8') as f:
        model.summary(print_fn=lambda s: f.write(s + '\n'))

    # 5. 配置回调（含学习率记录）
    lr_scheduler = LearningRateScheduler(lambda epoch, lr: lr, verbose=0)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(filepath=os.path.join(results_dir, 'best_model.h5'),
                        monitor='val_accuracy', save_best_only=True),
        lr_scheduler
    ]

    # 6. 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # 7. 测评并保存最终结果
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    with open(summary_path, 'a', encoding='utf-8') as f:
        f.write(f"测试集准确率: {test_acc:.4f}\n")
    print(f"Test accuracy: {test_acc:.4f}")

    acc_str = f"{test_acc:.4f}".replace('.', '_')
    model_path = os.path.join(results_dir, f"model_{timestamp}_acc_{acc_str}.h5")
    model.save(model_path)
    print(f"模型已保存至 {model_path}")

    # 8. 可视化结果
    plot_history(history, results_dir)
    plot_learning_rate(history, results_dir)

if __name__ == "__main__":
    main()
