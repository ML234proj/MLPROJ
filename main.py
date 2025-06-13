import pandas as pd
import numpy as np
import joblib
import os
import time
from src.data_preprocessing import get_train_test_data
from src.feature_engineering import feature_engineering
from src.model_training import train_model, ensemble_models
from src.prediction import generate_submission
from config import PATH_CONFIG

def main():
    start_time = time.time()
    print("贷款逾期预测项目 - 开始处理")
    
    # 确保目录存在
    os.makedirs(PATH_CONFIG['output_dir'], exist_ok=True)
    os.makedirs(PATH_CONFIG['model_dir'], exist_ok=True)
    
    # 1. 数据加载与预处理
    print("阶段1: 数据加载与预处理...")
    train_df, test_df = get_train_test_data()
    print(f"训练集形状: {train_df.shape}, 测试集形状: {test_df.shape}")
    
    # 保存测试集ID
    test_ids = test_df['id'].copy()
    train_ids = train_df['id'].copy()  # 保存训练集ID（可选）
    
    # 2. 特征工程
    print("\n阶段2: 特征工程...")
    X_train = train_df.drop(columns=['id', 'isDefault'], errors='ignore')
    y_train = train_df['isDefault']
    
    # 训练集特征工程
    print("处理训练集特征...")
    X_train_processed, encoders_info = feature_engineering(X_train, is_train=True)
    encoders, categorical_features = encoders_info
    PATH_CONFIG['categorical_features'] = categorical_features  # 更新全局配置
    
    # 保存编码器
    encoders_path = f"{PATH_CONFIG['model_dir']}/feature_encoders.pkl"
    joblib.dump((encoders, categorical_features), encoders_path)
    print(f"特征编码器及类别特征保存至: {encoders_path}")
    
    # 测试集特征工程
    print("\n处理测试集特征...")
    X_test = test_df.drop(columns=['id'], errors='ignore')
    X_test_processed, _ = feature_engineering(X_test, is_train=False, encoders=(encoders, categorical_features))
    
    # 3. 模型训练
    print("\n阶段3: 模型训练...")
    trained_models = train_model(X_train_processed, y_train)
    
    # 4. 预测
    print("\n阶段4: 生成预测结果...")
    predictions = ensemble_models(trained_models, X_test_processed)
    
    # 5. 生成提交文件
    print("\n阶段5: 生成提交文件...")
    generate_submission(test_ids, predictions)
    
    # 计时结束
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n项目完成! 总耗时: {duration:.2f}秒 ({duration/60:.2f}分钟)")

if __name__ == '__main__':
    main()