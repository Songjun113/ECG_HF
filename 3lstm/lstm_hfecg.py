import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
from glob import glob  # 新增：用于批量获取文件列表
from sklearn.model_selection import train_test_split   # ← 添加此行，确保 train_test_split 可用
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from tensorflow.keras.utils import to_categorical
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# ====================== 配置模块 ======================
class Hyperparameters:
    """集中管理超参数"""
    def __init__(self):
        self.data_dir = "../preprocessed_data"   # 修改为当前数据目录
        self.labels_file = "labels.csv"
        self.num_classes = 2
        self.batch_size = 8
        self.epochs = 100
        self.learning_rate = 1e-3
        self.patience_es = 24
        self.patience_lr = 8
        self.factor_lr = 0.8
        self.min_lr = 1e-6
        self.random_state = 42
        self.lstm_units = 64
        self.dropout_rate = 0.3

# ====================== 数据处理模块（已重写） ======================
def preprocess_data(all_files, labels_df):
    """
    批量加载并预处理 ECG 片段
    all_files: list of 文件完整路径，格式形如：ID123_beat_001.csv
    labels_df: DataFrame，包含 'name'（患者 ID）与 'labels' 列
    返回：
      X: ndarray, 形状 (样本数, 时间步, 导联数)
      y: ndarray, 形状 (样本数,) 的标签编码
      patient_ids: ndarray, 形状 (样本数,) 的患者 ID
    """
    ecg_samples = []
    labels      = []
    patient_ids = []

    # 构建患者ID到标签的映射
    name_to_label = dict(zip(labels_df['name'], labels_df['labels']))

    for file_path in all_files:
        base_name = os.path.basename(file_path)
        # 从 “_beat_” 前截取，即为患者 ID
        pid = base_name.split('_beat_')[0]
        if pid not in name_to_label:
            # 若无标签则跳过该文件
            continue

        try:
            df = pd.read_csv(file_path)
            # 提取第2~13列，共12导联数据
            ecg = df.iloc[:, 1:13].values
            # 归一化到 [-1,1]
            max_val = np.max(np.abs(ecg))
            ecg = ecg / (max_val if max_val != 0 else 1.0)

            ecg_samples.append(ecg)
            labels.append(name_to_label[pid])
            patient_ids.append(pid)
        except FileNotFoundError:
            print(f"Warning: File not found - {file_path}")
            continue

    # 将标签映射为整数
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)

    return np.array(ecg_samples), np.array(y_encoded), np.array(patient_ids)

# ====================== 模型构建、可视化、训练评估等模块保持不变 ======================
def build_lstm_model(input_shape, hp):
    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(hp.lstm_units, return_sequences=False)(inputs)
    x = layers.Dropout(hp.dropout_rate)(x)
    act = 'softmax' if hp.num_classes > 1 else 'sigmoid'
    outputs = layers.Dense(hp.num_classes, activation=act)(x)
    return models.Model(inputs, outputs)

def export_model_architecture(model, filepath):
    with open(filepath, 'w') as f:
        model.summary(print_fn=lambda s: f.write(s + '\n'))
    print(f"模型架构已导出至: {filepath}")

def plot_training_history(history, results_dir):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend()
    plt.savefig(f'{results_dir}/training_accuracy.png'); plt.close()

def plot_confusion_matrix(cm, results_dir, class_names=['Class 0', 'Class 1']):
    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label'); plt.ylabel('True Label')
    plt.xticks([0,1], class_names); plt.yticks([0,1], class_names)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i,j]}", ha="center",
                     color="white" if cm[i,j]>thresh else "black")
    plt.colorbar(im, shrink=0.6); plt.tight_layout()
    plt.savefig(f'{results_dir}/confusion_matrix.png'); plt.close()

def calculate_class_weights(y):
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))

def evaluate_model(model, X_val, y_val, summary_path, num_classes):
    with open(summary_path, 'a', encoding='utf-8') as f:
        f.write("\n=== 评估结果 ===\n")
    y_prob = model.predict(X_val)
    if num_classes == 2:
        y_score = y_prob[:,1]
        auc = roc_auc_score(y_val, y_score)
        y_pred = (y_score > 0.5).astype(int)
        with open(summary_path, 'a', encoding='utf-8') as f:
            f.write(f"AUC: {auc:.4f}\n")
    else:
        y_pred = np.argmax(y_prob, axis=1)

    cm     = confusion_matrix(y_val, y_pred)
    report = classification_report(y_val, y_pred)
    with open(summary_path, 'a', encoding='utf-8') as f:
        f.write(f"混淆矩阵:\n{cm}\n")
        f.write(f"分类报告:\n{report}\n")
        f.write("===========================\n")
    return cm

# ====================== 主流程模块 ======================
def main():
    hp = Hyperparameters()
    # 创建结果目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, 'training_summary.txt')

    # 1. 加载标签表
    labels_df = pd.read_csv(hp.labels_file, names=['name','labels'], header=0, encoding='utf-8')

    # 2. 批量获取 ECG 文件并预处理
    all_files = glob(f"{hp.data_dir}/*_beat_*.csv")
    if not all_files:
        raise ValueError(f"No ECG files found in {hp.data_dir}")
    X, y, patient_ids = preprocess_data(all_files, labels_df)

    # 3. 记录基本信息
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"训练时间: {timestamp}\n")
        f.write(f"样本总数: {X.shape[0]}, 正样本: {np.sum(y==1)}, 负样本: {np.sum(y==0)}\n")

    # 4. 划分训练/验证集并处理类别不平衡
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=hp.random_state
    )
    rus = RandomUnderSampler(random_state=hp.random_state)
    X_tr_flat, y_tr = rus.fit_resample(
        X_train.reshape(X_train.shape[0], -1), y_train
    )
    X_tr = X_tr_flat.reshape(-1, X_train.shape[1], X_train.shape[2])
    y_tr_oh = to_categorical(y_tr, hp.num_classes)
    class_weights = calculate_class_weights(y_tr)

    # 5. 构建并编译模型
    model = build_lstm_model(X_train.shape[1:], hp)
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    export_model_architecture(model, os.path.join(results_dir, 'model_architecture.txt'))

    # 6. 训练
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=hp.patience_es, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=hp.factor_lr, patience=hp.patience_lr, min_lr=hp.min_lr),
        ModelCheckpoint(os.path.join(results_dir, 'best_model.h5'), monitor='val_accuracy', save_best_only=True)
    ]
    history = model.fit(
        X_tr, y_tr_oh,
        batch_size=hp.batch_size,
        epochs=hp.epochs,
        validation_data=(X_val, to_categorical(y_val, hp.num_classes)),
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weights
    )

    # 7. 评估与可视化
    cm = evaluate_model(model, X_val, y_val, summary_path, hp.num_classes)
    plot_training_history(history, results_dir)
    plot_confusion_matrix(cm, results_dir)

    # 8. 保存超参数等总结
    with open(summary_path, 'a', encoding='utf-8') as f:
        f.write("\n超参数配置：\n")
        for k, v in vars(hp).items():
            f.write(f"{k} = {v}\n")

    print(f"训练完成，结果保存至 {results_dir}")

if __name__ == "__main__":
    main()
