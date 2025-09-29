import sys
print("Python version:", sys.version)
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns  # 添加seaborn库用于混淆矩阵可视化
import os
import datetime
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score  # 添加更多评估指标

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
    
    # 获取所有数据文件并按姓名分组
    from glob import glob
    import os
    
    # 构建姓名到标签的映射字典
    name_to_label = dict(zip(labels_df['name'], labels_df['labels']))
    
    # 获取所有csv文件
    all_files = glob(f"{data_dir}/*_beat_*.csv")
    
    # 按姓名分组文件
    name_groups = {}
    for file_path in all_files:
        base_name = os.path.basename(file_path)
        name = base_name.split('_beat_')[0]
        if name not in name_groups:
            name_groups[name] = []
        name_groups[name].append(file_path)
    
    # 处理每个人的数据
    for name, files in name_groups.items():
        if name not in name_to_label:
            print(f"Warning: No label found for {name}")
            continue
            
        label = name_to_label[name]
        for file_path in files:
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
        layers.Conv1D(32, kernel_size=5, padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.2),

        layers.Conv1D(64, kernel_size=5, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(128, kernel_size=3, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.GlobalAveragePooling1D(),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
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

# 新增函数：绘制并保存混淆矩阵
def plot_confusion_matrix(y_true, y_pred, model_name, timestamp):
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - {model_name}')
    
    # 保存图像
    fig_path = f"results/{timestamp}_{model_name}_confusion_matrix.png"
    plt.savefig(fig_path)
    plt.close()
    
    print(f"混淆矩阵图保存至: {fig_path}")
    return cm

# 新增函数：保存评估结果
def save_evaluation_results(y_true, y_pred, y_prob, timestamp, model_name):
    # 计算各项指标
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    
    # 保存结果到文本文件
    result_path = f"results/{timestamp}_{model_name}_evaluation.txt"
    with open(result_path, 'w') as f:
        f.write(f"Model Evaluation Results - {model_name} ({timestamp})\n")
        f.write("="*50 + "\n\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"评估结果保存至: {result_path}")
    return result_path

def main():
    labels_df = pd.read_csv("labels.csv", names=['name', 'labels'], header=0, encoding='utf-8')
    data_dir = "../preprocessed_data"
    X, y = preprocess_data(data_dir, labels_df)

    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=88, stratify=y)

    input_shape = X_train[0].shape
    model = build_cnn_model(input_shape)
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
    checkpoint = ModelCheckpoint('results/best_model.h5', monitor='val_accuracy', save_best_only=True)

    history = model.fit(X_train, y_train,
                        epochs=256,
                        batch_size=32,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping, reduce_lr, checkpoint])

    # 在测试集上评估模型
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # 获取预测结果
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype("int32").flatten()
    
    # 获取当前时间并格式化
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    acc_str = f"{test_acc:.4f}".replace('.', '_')
    model_name = f"model_{timestamp}_ac_{acc_str}"
    model_file = f"results/{model_name}.h5"

    # 保存模型
    model.save(model_file)
    print(f"模型保存至 {model_file}")
    
    # 绘制并保存混淆矩阵
    cm = plot_confusion_matrix(y_test, y_pred, model_name, timestamp)
    print("混淆矩阵:")
    print(cm)
    
    # 保存评估结果
    save_evaluation_results(y_test, y_pred, y_pred_prob, timestamp, model_name)
    
    # 输出分类报告
    report = classification_report(y_test, y_pred)
    print("分类报告:")
    print(report)

    # 绘图并保存训练历史
    plot_history(history)
    print("训练曲线图已保存至 results/training_history.png")

if __name__ == "__main__":
    main()