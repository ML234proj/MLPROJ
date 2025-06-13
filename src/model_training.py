import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score, 
                            precision_score, recall_score, classification_report,
                            confusion_matrix, log_loss)
from sklearn.utils import class_weight
from config import PATH_CONFIG

def train_model(X, y):
    # 1. 更精确的类别权重计算
    classes = np.unique(y)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=y)
    class_weights = dict(zip(classes, weights))
    
    # 2. 优化模型参数
    model = LGBMClassifier(
        n_estimators=3000,  # 增加树的数量
        learning_rate=0.01,  # 降低学习率
        num_leaves=127,  # 增加叶子节点数
        max_depth=-1,  # 不限制深度
        min_child_samples=50,  # 增加最小叶子样本数
        subsample=0.8,  # 降低行采样率
        colsample_bytree=0.7,  # 降低列采样率
        reg_alpha=0.3,  # 增加L1正则化
        reg_lambda=0.3,  # 增加L2正则化
        random_state=42,
        n_jobs=-1,
        class_weight=class_weights,  # 使用精确计算的权重
        objective='binary',
        boosting_type='gbdt',
        importance_type='gain'  # 按增益计算特征重要性
    )
    
    # K折交叉验证
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(X.shape[0])
    best_models = []  # 存储每折最佳模型
    fold_metrics = []  # 存储各折评估指标
    
    # 提前停止回调
    early_stopping = lgb.early_stopping(stopping_rounds=100, verbose=True)
    log_evaluation = lgb.log_evaluation(period=100)
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
        print(f"\n{'='*30} Training fold {fold+1}/5 {'='*30}")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
        
        # 训练模型
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric=['auc', 'binary_logloss', 'binary_error'],
            callbacks=[early_stopping, log_evaluation]
        )
        
        # 预测验证集
        valid_preds = model.predict_proba(X_valid)[:, 1]
        oof_preds[valid_idx] = valid_preds
        
        # ========== 多指标评估 ==========
        # 寻找最佳阈值（使用更精确的方法）
        best_threshold = find_optimal_threshold(y_valid, valid_preds)
        valid_preds_class = (valid_preds >= best_threshold).astype(int)
        
        # 计算各项指标
        fold_auc = roc_auc_score(y_valid, valid_preds)
        fold_f1 = f1_score(y_valid, valid_preds_class)
        fold_acc = accuracy_score(y_valid, valid_preds_class)
        fold_precision = precision_score(y_valid, valid_preds_class)
        fold_recall = recall_score(y_valid, valid_preds_class)
        fold_logloss = log_loss(y_valid, valid_preds)
        
        # 保存当前折指标
        metrics = {
            'fold': fold+1,
            'auc': fold_auc,
            'f1': fold_f1,
            'accuracy': fold_acc,
            'precision': fold_precision,
            'recall': fold_recall,
            'logloss': fold_logloss,
            'threshold': best_threshold
        }
        fold_metrics.append(metrics)
        
        # 显示分类报告和混淆矩阵
        print(f"\nFold {fold+1} Classification Report:")
        print(classification_report(y_valid, valid_preds_class))
        
        print(f"Confusion Matrix:")
        print(confusion_matrix(y_valid, valid_preds_class))
        
        # 获取最佳迭代次数
        best_iteration = model.booster_.best_iteration
        
        # 特征重要性分析
        importance = pd.DataFrame({
            'feature': model.booster_.feature_name(),
            'importance': model.booster_.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 features for fold {fold+1}:")
        print(importance.head(10))
        
        # 保存模型
        model_path = f"{PATH_CONFIG['model_dir']}/lgb_fold{fold+1}_best.pkl"
        joblib.dump(model, model_path)
        print(f"Best model (iteration={best_iteration}) saved to {model_path}")
        
        best_models.append(model)
    
    # ========== 整体评估 ==========
    print("\n" + "="*50)
    print("Cross-Validation Metrics Summary:")
    metrics_df = pd.DataFrame(fold_metrics).set_index('fold')
    print(metrics_df)
    
    # 计算平均指标
    mean_metrics = metrics_df.mean()
    print("\nAverage Metrics:")
    print(f"AUC:      {mean_metrics['auc']:.5f}")
    print(f"F1:       {mean_metrics['f1']:.5f}")
    print(f"Accuracy: {mean_metrics['accuracy']:.5f}")
    print(f"Precision:{mean_metrics['precision']:.5f}")
    print(f"Recall:   {mean_metrics['recall']:.5f}")
    print(f"LogLoss:  {mean_metrics['logloss']:.5f}")
    
    # 保存完整模型
    full_model_path = f"{PATH_CONFIG['model_dir']}/lgb_full_model.pkl"
    joblib.dump(best_models, full_model_path)
    print(f"Full model saved to {full_model_path}")
    
    return best_models

def find_optimal_threshold(y_true, y_pred_proba):
    """通过最大化F1分数寻找最佳阈值（优化版）"""
    best_threshold = 0.5
    best_f1 = 0
    
    # 使用更高效的搜索方法
    thresholds = np.linspace(0.2, 0.8, 200)
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            
    return best_threshold

def ensemble_models(models, X):
    """使用加权平均（基于验证集AUC）进行模型集成"""
    predictions = []
    weights = []
    
    for model in models:
        # 获取验证集AUC作为权重
        auc_score = model.booster_.best_score['valid_0']['auc']
        weights.append(auc_score)
        
        # 预测概率
        preds = model.predict_proba(X)[:, 1]
        predictions.append(preds)
    
    # 加权平均
    weights = np.array(weights) / sum(weights)
    final_predictions = np.zeros_like(predictions[0])
    
    for i in range(len(models)):
        final_predictions += predictions[i] * weights[i]
    
    # 校准预测概率
    final_predictions = np.clip(final_predictions, 0.01, 0.99)
    
    return final_predictions