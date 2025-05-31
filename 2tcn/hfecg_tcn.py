import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from collections import Counter

# ====================== 配置模块 ======================
class Hyperparameters:
    """集中管理超参数"""
    def __init__(self):
        self.data_dir = "raw_data"
        self.labels_file = "labels.csv"
        self.Kt = 8 # 卷积核大小
        self.pt = 0.5 # dropout比例
        self.Ft = 3 # 卷积核数量
        self.num_classes = 2 # 类别数量
        self.batch_size = 8 # 批大小
        self.epochs = 100
        self.learning_rate = 1e-3 # 学习率
        self.patience_es = 16 # early stopping patience
        self.patience_lr = 8 # learning rate scheduler patience
        self.factor_lr = 0.8 # learning rate scheduler factor
        self.min_lr = 1e-6 # 最小学习率
        self.n_splits = 2   # K折交叉验证折数
        self.random_state = 42

# ====================== 数据处理模块 ======================
def load_ecg_data(file_path):
    """加载单文件ECG数据"""
    df = pd.read_csv(file_path)
    return df.iloc[:, 1:13].values  # 提取12导联数据

def preprocess_data(data_dir, labels_df):
    """数据预处理"""
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
    labels_encoded = le.fit_transform(labels)
    return np.array(ecg_samples), np.array(labels_encoded)

# ====================== 模型构建模块 ======================
def build_tcn_model(input_shape, hp):
    """构建TCN模型"""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(8, hp.Kt, padding='causal', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    residual = layers.Conv1D(hp.Ft, 1, use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)
    residual = layers.ReLU()(residual)
    
    for dilation_rate in [1, 2, 4]:
        y = layers.Conv1D(hp.Ft, hp.Kt, padding='causal', dilation_rate=dilation_rate, use_bias=False)(x)
        y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
        y = layers.Dropout(hp.pt)(y)
        y = layers.Conv1D(hp.Ft, hp.Kt, padding='causal', dilation_rate=dilation_rate, use_bias=False)(y)
        y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
        y = layers.Dropout(hp.pt)(y)
        x = layers.Add()([residual, y])
        x = layers.ReLU()(x)
        residual = x
    
    x = layers.Flatten()(x)
    outputs = layers.Dense(hp.num_classes, activation='softmax' if hp.num_classes>1 else 'sigmoid')(x)
    return models.Model(inputs, outputs)

# ====================== 模型工具模块 ======================
def export_model_architecture(model, filepath):
    """导出模型架构到文件"""
    with open(filepath, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"模型架构已导出至: {filepath}")

# ====================== 可视化模块 ======================
def plot_training_subplots(histories, results_dir, n_splits):
    """绘制训练曲线"""
    rows = int(np.ceil(n_splits / 4))
    cols = min(n_splits, 4)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows), sharex='col', sharey='row')
    
    for i, history in enumerate(histories):
        ax = axes[i//cols, i%cols] if rows > 1 else axes[i%cols]
        ax.plot(history.history['accuracy'], label='Training', color='blue')
        ax.plot(history.history['val_accuracy'], label='Validation', color='orange')
        ax.set_title(f'Fold {i+1} Accuracy')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/kfold_training_subplots.png')
    plt.close()

def plot_confusion_matrices(cm_list, results_dir, n_splits, class_names=['Class 0', 'Class 1']):
    """绘制混淆矩阵"""
    rows = int(np.ceil(n_splits / 4))
    cols = min(n_splits, 4)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows))
    
    for i, cm in enumerate(cm_list):
        ax = axes[i//cols, i%cols] if rows > 1 else axes[i%cols]
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'Fold {i+1} Confusion Matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        
        thresh = cm.max() / 2.
        for j in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                ax.text(k, j, format(cm[j, k], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[j, k] > thresh else "black")
    
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/kfold_confusion_matrices.png')
    plt.show()
    plt.close()

# ====================== 训练与评估模块 ======================
def calculate_class_weights(y_train):
    """计算类别权重"""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    return dict(zip(classes, class_weights))

def evaluate_model(model, X_val, y_val, fold, results_summary_path, num_classes):
    """模型评估"""
    print(f"\n评估第 {fold+1} 折模型...")
    eval_results = []
    y_pred_prob = model.predict(X_val)
    
    if num_classes == 2:
        y_pred_prob = y_pred_prob[:, 1]
        auc = roc_auc_score(y_val, y_pred_prob)
        eval_results.append(f"AUC: {auc:.4f}")
        y_pred = (y_pred_prob > 0.5).astype(int)
    else:
        y_pred = np.argmax(y_pred_prob, axis=1)
    
    cm = confusion_matrix(y_val, y_pred)
    report = classification_report(y_val, y_pred)
    
    with open(results_summary_path, 'a', encoding='utf-8') as f:
        f.write(f"\n=== 第 {fold+1} 折评估结果 ===\n")
        f.write('\n'.join(eval_results))
        f.write(f"\n混淆矩阵:\n{cm}\n")
        f.write(f"\n分类报告:\n{report}\n")
        f.write("===========================\n")
    
    return cm

# ====================== 主流程模块 ======================


def main():
    hp = Hyperparameters()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)
    results_summary_path = os.path.join(results_dir, 'training_summary.txt')
    
    # 数据加载与预处理
    labels_df = pd.read_csv(hp.labels_file, names=['name', 'labels'], header=0, encoding='utf-8')
    X, y = preprocess_data(hp.data_dir, labels_df)
    
    # 全局信息记录
    with open(results_summary_path, 'w', encoding='utf-8') as f:
        f.write(f"训练时间: {timestamp}\n")
        f.write(f"全局类别分布: {dict(zip(*np.unique(y, return_counts=True)))}\n")
    
    y_one_hot = to_categorical(y, hp.num_classes)
    input_shape = X.shape[1:]
    
    # 模型架构导出
    base_model = build_tcn_model(input_shape, hp)
    export_model_architecture(base_model, f"{results_dir}/model_architecture.txt")
    
    # K折交叉验证
    skf = StratifiedKFold(n_splits=hp.n_splits, shuffle=True, random_state=hp.random_state)
    histories = []
    all_cms = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n处理第 {fold+1} 折...")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]  # 使用原始标签（非one-hot编码）
        
        # 打印原始训练集类别分布
        print(f"第 {fold+1} 折训练集原始类别分布: {Counter(y_train)}")
        
        # 应用SMOTE平衡训练集
        smote = SMOTE(random_state=hp.random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train.reshape(X_train.shape[0], -1), y_train)
        
        # 将X_train_resampled重塑回原始形状
        X_train_resampled = X_train_resampled.reshape(-1, X_train.shape[1], X_train.shape[2])
        y_train_resampled_one_hot = to_categorical(y_train_resampled, hp.num_classes)
        
        # 打印平衡后的训练集类别分布
        print(f"第 {fold+1} 折训练集平衡后类别分布: {Counter(y_train_resampled)}")
        
        # 计算类别权重（尽管SMOTE已平衡数据，权重仍可提供额外帮助）
        class_weight = calculate_class_weights(y_train_resampled)
        
        # 构建并训练模型
        model = build_tcn_model(input_shape, hp)
        model.compile(optimizer=tf.keras.optimizers.Adam(hp.learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=hp.patience_es, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=hp.factor_lr, patience=hp.patience_lr, min_lr=hp.min_lr),
            ModelCheckpoint(f"{results_dir}/model_fold_{fold+1}.h5", monitor='val_accuracy', save_best_only=True)
        ]
        
        history = model.fit(X_train_resampled, y_train_resampled_one_hot,
                            batch_size=hp.batch_size,
                            epochs=hp.epochs,
                            validation_data=(X_val, to_categorical(y_val, hp.num_classes)),
                            callbacks=callbacks,
                            verbose=1,
                            class_weight=class_weight)
        
        histories.append(history)
        cm = evaluate_model(model, X_val, y_val, fold, results_summary_path, hp.num_classes)
        all_cms.append(cm)
    
    # 结果可视化
    plot_training_subplots(histories, results_dir, hp.n_splits)
    plot_confusion_matrices(all_cms, results_dir, hp.n_splits)
    
    # 保存最终总结
    with open(results_summary_path, 'a', encoding='utf-8') as f:
        f.write("\n超参数配置:\n" + "\n".join([f"{k} = {v}" for k, v in vars(hp).items()]))
    
    print(f"训练完成，结果保存至 {results_dir}")

if __name__ == "__main__":
    main()