import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import joblib

# ====================== 配置模块 ======================
class Hyperparameters:
    """集中管理超参数"""
    def __init__(self):
        self.data_dir = "raw_data"
        self.labels_file = "labels.csv"
        self.num_classes = 2  # 类别数量
        self.random_state = 66

# ====================== 数据处理模块 ======================
def load_ecg_data(file_path):
    """加载单文件ECG数据"""
    df = pd.read_csv(file_path)
    return df.iloc[:, 1:17].values  # 提取16导联数据

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

# ====================== 模型工具模块 ======================
def train_and_evaluate_model(model, param_grid, X_train, y_train, X_val, y_val, model_name, results_dir, results_summary_path):
    """训练和评估模型"""
    print(f"\nTraining {model_name}...")
    
    # 进行网格搜索
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # 获取最佳参数和分数
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    # 保存最佳参数到结果汇总
    with open(results_summary_path, 'a', encoding='utf-8') as f:
        f.write(f"\n=== {model_name} ===\n")
        f.write(f"Best parameters: {best_params}\n")
        f.write(f"Best cross-validation accuracy: {best_score:.4f}\n")
    
    # 获取最佳模型
    best_model = grid_search.best_estimator_
    
    # 在验证集上预测
    y_pred = best_model.predict(X_val)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_val, y_pred)
    
    # 绘制并保存混淆矩阵
    plot_confusion_matrix(cm, results_dir, model_name)
    
    # 计算分类报告
    report = classification_report(y_val, y_pred)
    
    # 保存分类报告到结果汇总
    with open(results_summary_path, 'a', encoding='utf-8') as f:
        f.write(f"\nClassification Report:\n{report}\n")
    
    # 保存最佳模型
    joblib.dump(best_model, f"{results_dir}/{model_name}_best_model.joblib")
    
    print(f"{model_name} training completed.")

# ====================== 可视化模块 ======================
def plot_confusion_matrix(cm, results_dir, model_name, class_names=['Class 0', 'Class 1']):
    """绘制混淆矩阵"""
    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks([0, 1], class_names)
    plt.yticks([0, 1], class_names)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.colorbar(im, shrink=0.6)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/confusion_matrix_{model_name}.png')
    plt.close()

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
    
    # 划分训练集和验证集
    X_train_3d, X_val_3d, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=hp.random_state)
    
    # 打印原始训练集和验证集类别分布
    print(f"训练集原始类别分布: {Counter(y_train)}")
    print(f"验证集类别分布: {Counter(y_val)}")
    
    # 将3D数据展平为2D
    n_samples_train, n_timesteps, n_channels = X_train_3d.shape
    X_train_flat = X_train_3d.reshape(n_samples_train, n_timesteps * n_channels)
    n_samples_val = X_val_3d.shape[0]
    X_val_flat = X_val_3d.reshape(n_samples_val, n_timesteps * n_channels)
    
    # 应用欠采样平衡训练集
    rus = RandomUnderSampler(random_state=hp.random_state)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train_flat, y_train)
    
    # 打印平衡后的训练集类别分布
    print(f"训练集平衡后类别分布: {Counter(y_train_resampled)}")
    
    # 定义模型和参数网格
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    param_grid_xgb = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    param_grid_svm = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    
    models = {
        'RandomForest': (RandomForestClassifier(random_state=hp.random_state), param_grid_rf),
        'XGBoost': (XGBClassifier(random_state=hp.random_state), param_grid_xgb),
        'SVM': (SVC(random_state=hp.random_state), param_grid_svm)
    }
    
    # 训练和评估每个模型
    for model_name, (model, param_grid) in models.items():
        train_and_evaluate_model(model, param_grid, X_train_resampled, y_train_resampled, X_val_flat, y_val, model_name, results_dir, results_summary_path)
    
    print(f"训练完成，结果保存至 {results_dir}")

if __name__ == "__main__":
    main()