import sys
print("Python version:", sys.version)
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    LearningRateScheduler
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ====================== 结果保存：基于时间戳的子目录 ======================
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join("results", timestamp)
os.makedirs(results_dir, exist_ok=True)
summary_path = os.path.join(results_dir, "training_summary.txt")

# ====================== 数据加载与预处理 ======================
def load_ecg_data(file_path):
    """加载单文件ECG数据并返回 12 导联数组"""
    df = pd.read_csv(file_path)
    return df.iloc[:, 1:13].values

def preprocess_data(data_dir, labels_df):
    """
    批量预处理 ECG 数据
    标准化至 [-1,1]，并返回样本数组与标签数组
    """
    ecg_samples, labels = [], []
    for name, label in zip(labels_df['name'], labels_df['labels']):
        file_path = os.path.join(data_dir, f"{name}.csv")
        try:
            ecg = load_ecg_data(file_path)
            ecg = ecg / np.max(np.abs(ecg))  # 归一化
            ecg_samples.append(ecg)
            labels.append(label)
        except FileNotFoundError:
            print(f"Warning: File not found - {file_path}")
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    return np.array(ecg_samples), np.array(labels)

# ====================== 模型构建 ======================
def build_cnn_model(input_shape):
    """
    构建简单一维 CNN 分类模型
    """
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

# ====================== 可视化函数 ======================
def plot_history(history, save_dir):
    """绘制并保存训练/验证准确率与损失曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Acc')
    axes[0].plot(history.history['val_accuracy'], label='Val Acc')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(save_dir, "training_history.png")
    plt.savefig(path)
    plt.close()
    print(f"训练曲线图已保存至 {path}")

def plot_learning_rate(history, save_dir):
    """
    绘制并保存学习率随 Epoch 变化曲线，
    需要在回调中使用 LearningRateScheduler 记录 lr 历史
    """
    if 'lr' not in history.history:
        print("Warning: history 中不包含 'lr'，请添加 LearningRateScheduler 回调")
        return
    epochs = range(1, len(history.history['lr']) + 1)
    plt.figure(figsize=(6,4))
    plt.plot(epochs, history.history['lr'], label='Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    path = os.path.join(save_dir, "learning_rate.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
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
    # 2. 预处理
    data_dir = "raw_data"
    X, y = preprocess_data(data_dir, labels_df)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"数据形状: X={X.shape}, y={y.shape}\n")
        f.write(f"标签分布: {dict(zip(*np.unique(y, return_counts=True)))}\n")

    # 3. 划分训练集/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=88, stratify=y
    )
    with open(summary_path, 'a', encoding='utf-8') as f:
        f.write(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}\n")

    # 4. 模型构建
    input_shape = X_train.shape[1:]
    model = build_cnn_model(input_shape)
    model.summary(print_fn=lambda s: None)  # 可选：将模型结构输出至控制台
    with open(summary_path, 'a', encoding='utf-8') as f:
        model.summary(print_fn=lambda s: f.write(s + '\n'))

    # 5. 回调配置（包含学习率记录）
    lr_scheduler = LearningRateScheduler(lambda epoch, lr: lr, verbose=0)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(
            filepath=os.path.join(results_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True
        ),
        lr_scheduler
    ]

    # 6. 模型训练
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # 7. 评估并保存结果
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    with open(summary_path, 'a', encoding='utf-8') as f:
        f.write(f"测试集准确率: {test_acc:.4f}\n")
    print(f"Test accuracy: {test_acc:.4f}")

    # 按时间戳和精度保存模型
    acc_str = f"{test_acc:.4f}".replace('.', '_')
    model_path = os.path.join(results_dir, f"model_{timestamp}_acc_{acc_str}.h5")
    model.save(model_path)
    print(f"模型已保存至 {model_path}")

    # 8. 可视化并保存
    plot_history(history, results_dir)
    plot_learning_rate(history, results_dir)

if __name__ == "__main__":
    main()
