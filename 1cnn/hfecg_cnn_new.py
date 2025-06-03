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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight  # 新增：计算类别权重

# 创建结果保存目录
os.makedirs("results", exist_ok=True)

# 数据加载和预处理函数（调整：接收all_files参数，避免内部重复获取）
def preprocess_data(all_files, labels_df):
    ecg_samples = []
    labels = []
    patient_ids = []  # 记录每个样本对应的患者ID
    
    # 构建姓名到标签的映射字典
    name_to_label = dict(zip(labels_df['name'], labels_df['labels']))
    
    # 处理每个文件
    for file_path in all_files:
        base_name = os.path.basename(file_path)
        name = base_name.split('_beat_')[0]  # 从文件名提取患者ID
        
        if name not in name_to_label:
            # print(f"Warning: No label found for {name}")
            continue
            
        label = name_to_label[name]
        try:
            # 加载并归一化ECG数据
            df = pd.read_csv(file_path)
            ecg_data = df.iloc[:, 1:13].values  # 提取12导联数据
            max_val = np.max(np.abs(ecg_data))
            ecg_data = ecg_data / max_val  # 归一化到[-1, 1]
            
            ecg_samples.append(ecg_data)
            labels.append(label)
            patient_ids.append(name)  # 记录患者ID（与样本顺序一致）
        except FileNotFoundError:
            print(f"Warning: File not found - {file_path}")
            continue
    
    # 标签编码（0=负类，1=正类）
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    
    return (np.array(ecg_samples), 
            np.array(labels), 
            np.array(patient_ids))  # 返回样本、标签、患者ID


def build_lead_wise_model(input_shape):
    # 输入形状：(时序长度, 12导联)
    inputs = layers.Input(shape=input_shape)
    
    # -------------------- 导联独立特征提取分支 --------------------
    # 拆分12个导联（每个导联形状：(时序长度, 1)）
    lead_branches = []
    for i in range(12):
        # 提取第i个导联（保留通道维度）
        lead = layers.Lambda(lambda x: x[..., i:i+1])(inputs)
        
        # 导联独立的特征提取（参数共享：所有导联使用同一组卷积核）
        x = layers.Conv1D(32, kernel_size=5, padding='same')(lead)  # 仅处理当前导联的时序
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Conv1D(64, kernel_size=5, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)  # 时序长度减半
        
        x = layers.Conv1D(128, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.GlobalAveragePooling1D()(x)  # 每个导联输出128维特征
        
        lead_branches.append(x)
    
    # -------------------- 导联特征融合 --------------------
    # 拼接12个导联的特征向量（总维度：12×128=1536）
    merged_features = layers.Concatenate()(lead_branches)
    
    # -------------------- 分类头 --------------------
    x = layers.Dense(256, activation='relu')(merged_features)  # 融合后增加非线性变换
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)  # 二分类输出
    
    # 模型编译
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
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
    # 加载标签文件
    labels_df = pd.read_csv("labels.csv", names=['name', 'labels'], header=0, encoding='utf-8')
    data_dir = "../preprocessed_data"
    
    # 获取所有ECG文件路径
    all_files = glob(f"{data_dir}/*_beat_*.csv")
    if not all_files:
        raise ValueError(f"No ECG files found in {data_dir}")
    
    # 预处理数据并获取患者ID
    X, y, patient_ids = preprocess_data(all_files, labels_df)
    print(f"原始数据: 样本数={X.shape[0]}, 正样本数={np.sum(y==1)}, 负样本数={np.sum(y==0)}")
    
    # 患者级分层划分
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=88)
    train_idx, val_idx = next(gss.split(X, y, groups=patient_ids))
    X_train, X_test = X[train_idx], X[val_idx]
    y_train, y_test = y[train_idx], y[val_idx]
    
    # 统计训练集和验证集的正负样本分布
    train_pos = np.sum(y_train == 1)
    train_neg = len(y_train) - train_pos
    val_pos = np.sum(y_test == 1)
    val_neg = len(y_test) - val_pos
    print(f"训练集: 正样本={train_pos}, 负样本={train_neg}, 正样本占比={train_pos/len(y_train):.2%}")
    print(f"验证集: 正样本={val_pos}, 负样本={val_neg}, 正样本占比={val_pos/len(y_test):.2%}")

    # -------------------- 样本均衡化核心逻辑 --------------------
    # 计算类别权重
    class_weights = compute_class_weight(
        class_weight='balanced',  # 自动根据类别频率计算权重
        classes=np.unique(y_train),  # 训练集的类别标签
        y=y_train
    )
    class_weights = dict(enumerate(class_weights))  # 转换为{0: w0, 1: w1}格式
    print(f"类别权重: 负类={class_weights[0]:.2f}, 正类={class_weights[1]:.2f}")

    # -------------------- 模型训练 --------------------
    input_shape = X_train[0].shape
    model = build_lead_wise_model(input_shape)
    model.summary()

    # 回调函数
    early_stopping = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6, verbose=1)
    checkpoint = ModelCheckpoint('results/best_model.h5', monitor='val_accuracy', save_best_only=True)

    # 训练（添加class_weight参数）
    history = model.fit(X_train, y_train,
                        epochs=256,
                        batch_size=256,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping, reduce_lr, checkpoint],
                        class_weight=class_weights)  # 关键：使用类别权重均衡样本

    # 测试评估
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

    # 保存模型
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    acc_str = f"{test_acc:.4f}".replace('.', '_')
    model_name = f"results/model_{timestamp}_ac_{acc_str}.h5"
    model.save(model_name)
    print(f"模型保存至 {model_name}")

    # 保存训练曲线
    plot_history(history)
    print("训练曲线图已保存至 results/training_history.png")

if __name__ == "__main__":
    main()