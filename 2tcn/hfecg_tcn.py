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
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from tensorflow.keras.utils import to_categorical

# 创建结果保存目录
os.makedirs("results", exist_ok=True)

# 集中管理的超参数配置
class Hyperparameters:
    def __init__(self):
        # 数据处理参数
        self.data_dir = "raw_data"
        self.labels_file = "labels.csv"
        
        # 模型参数
        self.Kt = 16  # TCN卷积核大小
        self.pt = 0.4  # Dropout比例
        self.Ft = 11  # 特征图数量
        
        # 训练参数
        self.num_classes = 2
        self.batch_size = 16
        self.epochs = 100
        self.learning_rate = 1e-3
        self.patience_es = 24  # 早停耐心值
        self.patience_lr = 8  # 学习率降低耐心值
        self.factor_lr = 0.8  # 学习率降低因子
        self.min_lr = 1e-6    # 最小学习率
        
        # K折交叉验证参数
        self.n_splits = 2
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
    x = layers.Conv1D(filters=8, kernel_size=hp.Kt, padding='causal', use_bias=False)(inputs)
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
    return model

def export_model_architecture(model, filepath):
    """将模型架构导出到文本文件"""
    with open(filepath, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"模型架构已导出至: {filepath}")

def plot_training_subplots(histories, results_dir, n_splits):
    """绘制每个折的训练/验证曲线子图"""
    rows = int(np.ceil(n_splits / 4))  # 每行4个子图
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
    plot_path = f'{results_dir}/kfold_training_subplots.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"训练曲线子图已保存至 {plot_path}")

def plot_confusion_matrices(cm_list, results_dir, n_splits, class_names=['Class 0', 'Class 1']):
    """绘制所有折的混淆矩阵子图"""
    rows = int(np.ceil(n_splits / 4))  # 每行4个子图
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
        
        # 添加数值标注
        thresh = cm.max() / 2.
        for j in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                ax.text(k, j, format(cm[j, k], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[j, k] > thresh else "black")
    
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
    plt.tight_layout()
    plot_path = f'{results_dir}/kfold_confusion_matrices.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"混淆矩阵子图已保存至 {plot_path}")

def calculate_class_weights(y_train):
    """计算类别权重以处理不平衡数据"""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    
    # 转换为字典格式，以便Keras使用
    class_weight_dict = dict(zip(classes, class_weights))
    print(f"计算得到的类别权重: {class_weight_dict}")
    return class_weight_dict

def evaluate_model(model, X_val, y_val, fold, results_summary_path):
    """评估模型并保存详细的性能指标到总结果文件"""
    print(f"\n正在评估第 {fold+1} 折模型...")
    eval_results = []
    
    # 获取预测概率
    y_pred_prob = model.predict(X_val)
    
    # 对于二分类，获取正类的概率
    if hp.num_classes == 2:
        y_pred_prob = y_pred_prob[:, 1]
        # 计算AUC
        auc = roc_auc_score(y_val, y_pred_prob)
        eval_results.append(f"第 {fold+1} 折 AUC: {auc:.4f}")
    
    # 获取预测类别
    y_pred = np.argmax(y_pred_prob, axis=1) if hp.num_classes > 2 else (y_pred_prob > 0.5).astype(int)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_val, y_pred)
    eval_results.append(f"第 {fold+1} 折 混淆矩阵:\n{cm}")
    
    # 计算分类报告
    report = classification_report(y_val, y_pred)
    eval_results.append(f"第 {fold+1} 折 分类报告:\n{report}")
    
    # 写入总结果文件
    with open(results_summary_path, 'a', encoding='utf-8') as f:
        f.write(f"\n\n=== 第 {fold+1} 折 评估结果 ===\n")
        f.write('\n'.join(eval_results))
        if hp.num_classes == 2:
            f.write(f"\nAUC: {auc:.4f}\n")
        f.write("===========================\n")
    
    return cm, '\n'.join(eval_results)

def main():
    global hp
    hp = Hyperparameters()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)
    results_summary_path = os.path.join(results_dir, 'training_summary.txt')

    labels_df = pd.read_csv(hp.labels_file, names=['name', 'labels'], header=0, encoding='utf-8')
    X, y = preprocess_data(hp.data_dir, labels_df)
    
    # 打印全局类别分布并写入总文件
    unique, counts = np.unique(y, return_counts=True)
    global_dist = f"全局类别分布: {dict(zip(unique, counts))}"
    print(global_dist)
    with open(results_summary_path, 'w', encoding='utf-8') as f:
        f.write(f"训练时间: {timestamp}\n")
        f.write(f"{global_dist}\n")
        f.write("===========================\n")
    
    y_one_hot = to_categorical(y, num_classes=hp.num_classes)
    input_shape = X.shape[1:]

    # ====================== 平衡划分逻辑修改 ======================
    # 1. 按类别分组，获取各分类别的索引
    class_indices = [np.where(y == c)[0] for c in range(hp.num_classes)]
    class_counts = [len(inds) for inds in class_indices]
    print(f"各分类别样本数: {class_counts}")
    
    # 2. 确定每折每个类别的样本数（取最小类别样本数进行均分）
    min_class_samples = min(class_counts)
    n_per_fold_per_class = min_class_samples // hp.n_splits  # 每折每个类别样本数
    if n_per_fold_per_class == 0:
        raise ValueError("类别样本数不足，无法进行平衡划分！")
    
    # 3. 对每个类别进行欠采样（多数类）或保留（少数类）
    balanced_class_indices = []
    for c in range(hp.num_classes):
        indices = class_indices[c]
        if len(indices) > n_per_fold_per_class * hp.n_splits:
            # 欠采样多数类，保留n_per_fold_per_class * hp.n_splits个样本
            np.random.seed(hp.random_state)  # 固定随机种子确保可复现
            sampled_indices = np.random.choice(indices, size=n_per_fold_per_class * hp.n_splits, replace=False)
        else:
            sampled_indices = indices  # 少数类直接使用全部样本
        balanced_class_indices.append(sampled_indices)
    
    # 4. 自定义K折划分：每个折中每个类别取n_per_fold_per_class个样本
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=hp.n_splits, shuffle=True, random_state=hp.random_state)
    histories = []
    all_cms = []

    # 构建并保存基础模型架构（不变）
    base_model = build_tcn_model(input_shape, hp)
    model_arch_path = f'{results_dir}/model_architecture.txt'
    export_model_architecture(base_model, model_arch_path)
    with open(results_summary_path, 'a', encoding='utf-8') as f:
        with open(model_arch_path, 'r') as arch_f:
            f.write("\n模型架构:\n")
            f.write(arch_f.read())
        f.write("===========================\n")

    for fold in range(hp.n_splits):
        train_idx = []
        val_idx = []
        for c in range(hp.num_classes):
            indices = balanced_class_indices[c]
            # 对当前类别划分第fold折的训练/验证索引
            split = list(kf.split(indices))[fold]  # 获取第fold折的划分
            train_c_idx, val_c_idx = split
            train_idx.extend(indices[train_c_idx])
            val_idx.extend(indices[val_c_idx])
        
        # 转换为numpy数组并打乱顺序（避免同类样本连续）
        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)
        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)
        
        # 打印当前折的标签分布
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        train_counts = np.bincount(y_train_fold)
        val_counts = np.bincount(y_val_fold)
        print(f"\n===== 第 {fold+1} 折 =====")
        print(f"训练集类别分布: {dict(zip(np.unique(y_train_fold), train_counts))}")
        print(f"验证集类别分布: {dict(zip(np.unique(y_val_fold), val_counts))}")
        with open(results_summary_path, 'a', encoding='utf-8') as f:
            f.write(f"\n第 {fold+1} 折训练集分布: {dict(zip(np.unique(y_train_fold), train_counts))}\n")
            f.write(f"第 {fold+1} 折验证集分布: {dict(zip(np.unique(y_val_fold), val_counts))}\n")
        
        # 划分数据（与原逻辑一致）
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_one_hot[train_idx], y_one_hot[val_idx]
        
        # 计算类别权重（可选，若需进一步平衡训练）
        class_weight = calculate_class_weights(np.argmax(y_train, axis=1))
        
        # 训练模型（与原逻辑一致）
        model = build_tcn_model(input_shape, hp)
        model.compile(optimizer=tf.keras.optimizers.Adam(hp.learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        callbacks = [EarlyStopping(monitor='val_loss', patience=hp.patience_es, restore_best_weights=True),
                     ReduceLROnPlateau(monitor='val_loss', factor=hp.factor_lr, patience=hp.patience_lr, min_lr=hp.min_lr),
                     ModelCheckpoint(f'{results_dir}/model_fold_{fold+1}.h5', monitor='val_accuracy', save_best_only=True)]
        
        history = model.fit(X_train, y_train,
                            epochs=hp.epochs,
                            batch_size=hp.batch_size,
                            validation_data=(X_val, y_val),
                            callbacks=callbacks,
                            verbose=1,
                            class_weight=class_weight)
        
        histories.append(history)
        print(f"第 {fold+1} 折训练完成，模型保存在 {results_dir}/model_fold_{fold+1}.h5")
        
        # 评估模型（与原逻辑一致）
        cm, _ = evaluate_model(model, X_val, np.argmax(y_val, axis=1), fold, results_summary_path)
        all_cms.append(cm)
    
    # 后续绘图和保存结果逻辑不变
    plot_training_subplots(histories, results_dir, hp.n_splits)
    plot_confusion_matrices(all_cms, results_dir, hp.n_splits)
    
    # 计算并保存总体混淆矩阵到总文件
    avg_cm = np.mean(all_cms, axis=0)
    avg_cm_str = f"\n平均混淆矩阵:\n{avg_cm}"
    print(avg_cm_str)
    with open(results_summary_path, 'a', encoding='utf-8') as f:
        f.write(avg_cm_str)
        f.write("\n===========================\n")
    
    # 保存超参数配置到总文件
    hp_config = "\n超参数配置:\n" + "\n".join([f"{attr} = {value}" for attr, value in vars(hp).items()])
    print(hp_config)
    with open(results_summary_path, 'a', encoding='utf-8') as f:
        f.write(hp_config)
        f.write("\n===========================\n")
    
    print(f"所有 K 折训练完成，结果保存在 {results_dir} 文件夹中，总摘要文件: {results_summary_path}")


if __name__ == "__main__":
    main()