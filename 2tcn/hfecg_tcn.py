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
        self.epochs = 50
        self.learning_rate = 1e-3
        self.patience_es = 8  # 早停耐心值
        self.patience_lr = 5  # 学习率降低耐心值
        self.factor_lr = 0.3  # 学习率降低因子
        self.min_lr = 1e-6    # 最小学习率
        
        # K折交叉验证参数
        self.n_splits = 16
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
        
        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
        
        # 绘制ROC曲线
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Fold {fold+1}')
        plt.legend(loc="lower right")
        plt.savefig(f'{os.path.dirname(results_summary_path)}/roc_curve_fold_{fold+1}.png')
        plt.close()
    
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
    global hp  # 让hp在全局范围内可用，供evaluate_model函数使用
    # 初始化超参数
    hp = Hyperparameters()
    
    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建以时间戳命名的结果文件夹
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建总结果文件路径
    results_summary_path = os.path.join(results_dir, 'training_summary.txt')
    
    # 加载标签数据
    labels_df = pd.read_csv(hp.labels_file, names=['name', 'labels'], header=0, encoding='utf-8')
    
    # 数据预处理
    X, y = preprocess_data(hp.data_dir, labels_df)
    
    # 打印全局类别分布并写入总文件
    unique, counts = np.unique(y, return_counts=True)
    global_dist = f"全局类别分布: {dict(zip(unique, counts))}"
    print(global_dist)
    with open(results_summary_path, 'w', encoding='utf-8') as f:
        f.write(f"训练时间: {timestamp}\n")
        f.write(f"{global_dist}\n")
        f.write("===========================\n")
    
    # 将标签转换为独热编码
    from tensorflow.keras.utils import to_categorical
    y_one_hot = to_categorical(y, num_classes=hp.num_classes)

    input_shape = X.shape[1:]
    skf = StratifiedKFold(n_splits=hp.n_splits, shuffle=True, random_state=hp.random_state)
    histories = []
    all_cms = []  # 存储所有折的混淆矩阵

    # 构建并保存基础模型架构
    base_model = build_tcn_model(input_shape, hp)
    model_arch_path = f'{results_dir}/model_architecture.txt'
    export_model_architecture(base_model, model_arch_path)
    # 将模型架构写入总文件
    with open(results_summary_path, 'a', encoding='utf-8') as f:
        with open(model_arch_path, 'r') as arch_f:
            f.write("\n模型架构:\n")
            f.write(arch_f.read())
        f.write("===========================\n")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # 训练前标签分布统计
        y_train_raw = y[train_idx]
        unique_train, counts_train = np.unique(y_train_raw, return_counts=True)
        label_dist = f"第 {fold+1} 折训练前标签分布（训练集）: {dict(zip(unique_train, counts_train))}"
        print(f"\n{label_dist}")
        # 写入总结果文件
        with open(results_summary_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{label_dist}\n")

        print(f"\n正在训练第 {fold+1} 折...")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_one_hot[train_idx], y_one_hot[val_idx]  # 使用独热编码的标签

        # 计算类别权重（针对独热编码调整）
        class_weight = calculate_class_weights(np.argmax(y_train, axis=1))

        # 为每一折创建新模型
        model = build_tcn_model(input_shape, hp)
        
        # 编译模型，使用categorical_crossentropy损失函数
        model.compile(optimizer=tf.keras.optimizers.Adam(hp.learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

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
                            verbose=1,
                            class_weight=class_weight)

        histories.append(history)
        print(f"第 {fold+1} 折训练完成，模型保存在 {checkpoint_path}")
        
        # 评估模型（传入总结果文件路径）
        cm, _ = evaluate_model(model, X_val, np.argmax(y_val, axis=1), fold, results_summary_path)  # 传入类别索引而非独热编码
        all_cms.append(cm)
        

    # 绘制并保存训练历史
    plot_history(histories, results_dir)
    
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
    