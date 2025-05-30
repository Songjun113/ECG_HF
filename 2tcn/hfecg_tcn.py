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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# 创建结果保存目录
os.makedirs("results", exist_ok=True)

# 集中管理的超参数配置
class Hyperparameters:
    def __init__(self):
        # 数据处理参数
        self.data_dir = "raw_data"
        self.labels_file = "labels.csv"
        
        # 模型参数
        self.Kt = 11  # TCN卷积核大小
        self.pt = 0.3  # Dropout比例
        self.Ft = 11  # 特征图数量
        
        # 训练参数
        self.num_classes = 2
        self.batch_size = 32
        self.epochs = 100
        self.learning_rate = 1e-4
        self.patience_es = 8  # 早停耐心值
        self.patience_lr = 5  # 学习率降低耐心值
        self.factor_lr = 0.5  # 学习率降低因子
        self.min_lr = 1e-6    # 最小学习率
        
        # K折交叉验证参数
        self.n_splits = 5
        self.random_state = 42

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

def build_tcn_model(input_shape, hp):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv1D(filters=2, kernel_size=hp.Kt, padding='causal', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    residual = layers.Conv1D(filters=hp.Ft, kernel_size=1, use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)
    residual = layers.ReLU()(residual)

    for dilation_rate in [1, 2, 4]:
        y = layers.Conv1D(filters=hp.Ft, kernel_size=hp.Kt, padding='causal', 
                          dilation_rate=dilation_rate, use_bias=False)(x)
        y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
        y = layers.Dropout(hp.pt)(y)
        y = layers.Conv1D(filters=hp.Ft, kernel_size=hp.Kt, padding='causal', 
                          dilation_rate=dilation_rate, use_bias=False)(y)
        y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
        y = layers.Dropout(hp.pt)(y)
        x = layers.Add()([residual, y])
        x = layers.ReLU()(x)
        residual = x

    x = layers.Flatten()(x)
    x = layers.Dense(hp.num_classes, 
                     activation='softmax' if hp.num_classes > 1 else 'sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.learning_rate),
                  loss='sparse_categorical_crossentropy' if hp.num_classes > 1 else 'binary_crossentropy',
                  metrics=['accuracy'])
    return model

def export_model_architecture(model, filepath):
    """将模型架构导出到文本文件"""
    with open(filepath, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"模型架构已导出至: {filepath}")

def plot_history(histories, results_dir):
    avg_acc = np.mean([h.history['val_accuracy'][-1] for h in histories])
    plt.figure(figsize=(12, 5))
    for i, history in enumerate(histories):
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label=f'Fold {i+1}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['val_accuracy'], label=f'Fold {i+1}')
        plt.xlabel('Epochs')
        plt.ylabel('Val Accuracy')
        plt.title('Validation Accuracy')

    plt.subplot(1, 2, 1)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.legend()
    plt.tight_layout()
    plot_path = f'{results_dir}/kfold_training_history.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"平均验证精度：{avg_acc:.4f}，训练曲线图已保存至 {plot_path}")

def main():
    # 初始化超参数
    hp = Hyperparameters()
    
    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建以时间戳命名的结果文件夹
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # 加载标签数据
    labels_df = pd.read_csv(hp.labels_file, names=['name', 'labels'], header=0, encoding='utf-8')
    
    # 数据预处理
    X, y = preprocess_data(hp.data_dir, labels_df)

    input_shape = X.shape[1:]
    skf = StratifiedKFold(n_splits=hp.n_splits, shuffle=True, random_state=hp.random_state)
    histories = []

    # 构建并保存基础模型架构
    base_model = build_tcn_model(input_shape, hp)
    model_arch_path = f'{results_dir}/model_architecture.txt'
    export_model_architecture(base_model, model_arch_path)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n正在训练第 {fold+1} 折...")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 为每一折创建新模型
        model = build_tcn_model(input_shape, hp)

        # 设置回调函数
        early_stopping = EarlyStopping(monitor='val_loss', patience=hp.patience_es, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=hp.factor_lr, 
                                     patience=hp.patience_lr, min_lr=hp.min_lr)
        checkpoint_path = f'{results_dir}/model_fold_{fold+1}.h5'
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True)

        # 训练模型
        history = model.fit(X_train, y_train,
                            epochs=hp.epochs,
                            batch_size=hp.batch_size,
                            validation_data=(X_val, y_val),
                            callbacks=[early_stopping, reduce_lr, checkpoint],
                            verbose=1)

        histories.append(history)
        print(f"第 {fold+1} 折训练完成，模型保存在 {checkpoint_path}")
        
        # 保存该折的模型架构（可选，如需详细了解每一折的细微差异）
        # fold_model_arch_path = f'{results_dir}/model_architecture_fold_{fold+1}.txt'
        # export_model_architecture(model, fold_model_arch_path)

    # 绘制并保存训练历史
    plot_history(histories, results_dir)
    
    # 保存超参数配置
    hp_config_path = f'{results_dir}/hyperparameters.txt'
    with open(hp_config_path, 'w') as f:
        for attr, value in vars(hp).items():
            f.write(f"{attr} = {value}\n")
    print(f"超参数配置已保存至: {hp_config_path}")
    
    print(f"所有 K 折训练完成，结果保存在 {results_dir} 文件夹中。")

if __name__ == "__main__":
    main()