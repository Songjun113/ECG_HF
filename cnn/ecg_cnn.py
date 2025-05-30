import sys
print("Python version:", sys.version)
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import datetime
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 创建结果保存目录
os.makedirs("results", exist_ok=True)

# 数据加载和预处理函数
def load_ecg_data(file_path):
    df = pd.read_csv(file_path)
    ecg_data = df.iloc[:, 1:13].values  # 提取12导联数据
    return ecg_data

def preprocess_data(data_dir, labels_df):
    ecg_samples = []
    labels = []
    for name, label in zip(labels_df['name'], labels_df['labels']):
        file_path = f"{data_dir}/{name}.csv"
        try:
            ecg_data = load_ecg_data(file_path)
            max_val = np.max(np.abs(ecg_data))
            ecg_data = ecg_data / max_val
            ecg_samples.append(ecg_data)
            labels.append(label)
        except FileNotFoundError:
            print(f"Warning: File not found - {file_path}")
            continue
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    return np.array(ecg_samples), np.array(labels)

def build_cnn_model(input_shape):
    model = models.Sequential([
        layers.Conv1D(32, kernel_size=5, activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),

        layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')

    plt.tight_layout()
    plt.savefig('results/training_history.png')
    plt.close()

def main():
    labels_df = pd.read_csv("labels.csv", names=['name', 'labels'], header=0, encoding='utf-8')
    data_dir = "raw_data"
    X, y = preprocess_data(data_dir, labels_df)

    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=88, stratify=y)

    input_shape = X_train[0].shape
    model = build_cnn_model(input_shape)
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    checkpoint = ModelCheckpoint('results/best_model.h5', monitor='val_loss', save_best_only=True)

    history = model.fit(X_train, y_train,
                        epochs=60,
                        batch_size=4,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping, reduce_lr, checkpoint])

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

    # 获取当前时间并格式化
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    acc_str = f"{test_acc:.4f}".replace('.', '_')
    model_name = f"results/model_{timestamp}_ac_{acc_str}.h5"

    # 保存模型
    model.save(model_name)
    print(f"模型保存至 {model_name}")

    # 绘图并保存
    plot_history(history)
    print("训练曲线图已保存至 results/training_history.png")

if __name__ == "__main__":
    main()