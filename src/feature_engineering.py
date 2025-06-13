import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from config import PATH_CONFIG

def feature_engineering(df, is_train=True, encoders=None):
    # 创建编码器副本避免修改原始配置
    categorical_features = PATH_CONFIG['categorical_features'].copy()
    
    # 确保不处理ID列
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    # 1. 处理高基数特征
    high_card_feats = ['purpose', 'postCode', 'regionCode']
    
    # 训练时初始化编码器，测试时使用已训练的编码器
    if encoders is None:
        encoders = {}
        for feat in high_card_feats:
            if feat in df.columns:
                # 填充缺失值
                df[feat].fillna('missing', inplace=True)
                # 转换为字符串
                df[feat] = df[feat].astype(str)
                
                # 创建和训练编码器
                encoder = LabelEncoder()
                df[feat] = encoder.fit_transform(df[feat])
                encoders[feat] = encoder
    else:  # 测试集模式
        for feat in high_card_feats:
            if feat in df.columns and feat in encoders:
                # 填充缺失值
                df[feat].fillna('missing', inplace=True)
                # 转换为字符串
                df[feat] = df[feat].astype(str)
                
                # 处理训练集中未见过的类别
                unique_classes = set(df[feat].unique())
                encoder_classes = set(encoders[feat].classes_)
                unknown_classes = unique_classes - encoder_classes
                
                if unknown_classes:
                    # 处理未知类别 - 标记为 'unknown'
                    df.loc[df[feat].isin(unknown_classes), feat] = 'unknown'
                    # 添加未知类别到编码器
                    encoders[feat].classes_ = np.append(encoders[feat].classes_, 'unknown')
                
                try:
                    # 转换特征
                    df[feat] = encoders[feat].transform(df[feat])
                except ValueError:
                    # 如果仍然有未知类别，创建新的编码器
                    encoder = LabelEncoder()
                    df[feat] = encoder.fit_transform(df[feat])
                    encoders[feat] = encoder
    
    # 2. 创建新特征
    # 收入与贷款比率
    if 'installment' in df.columns and 'annualIncome' in df.columns:
        # 添加微小值避免除零错误
        df['installment_ratio'] = df['installment'] / (df['annualIncome'] + 1e-5)
    
    # 贷款与收入比率
    if 'loanAmnt' in df.columns and 'annualIncome' in df.columns:
        df['loan_to_income'] = df['loanAmnt'] / (df['annualIncome'] + 1e-5)
    
    # 利息期限乘积
    if 'interestRate' in df.columns and 'term' in df.columns:
        df['interest_term'] = df['interestRate'] * df['term']
    
    # FICO评分平均值
    if 'ficoRangeLow' in df.columns and 'ficoRangeHigh' in df.columns:
        df['fico_average'] = (df['ficoRangeLow'] + df['ficoRangeHigh']) / 2
    
    # 新增特征：债务负担比率
    if 'dti' in df.columns and 'annualIncome' in df.columns:
        df['debt_burden'] = df['dti'] * df['annualIncome'] / 1000
    
    # 新增特征：信用利用率与收入的交互
    if 'revolUtil' in df.columns and 'annualIncome' in df.columns:
        df['revol_util_income'] = df['revolUtil'] * np.log1p(df['annualIncome'])
    
    # 3. 数值特征分箱（改用KBinsDiscretizer）
    bin_cols = ['annualIncome', 'dti', 'revolUtil', 'fico_average', 'loanAmnt']
    new_cat_features = []
    
    for col in bin_cols:
        if col in df.columns:
            # 填充缺失值
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            
            # 使用KBinsDiscretizer进行分箱
            try:
                # 使用分位数分箱策略
                discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
                bin_col = f'{col}_bin'
                df[bin_col] = discretizer.fit_transform(df[[col]]).astype(int)
                new_cat_features.append(bin_col)
            except Exception as e:
                print(f"分箱失败 {col}: {str(e)}，使用中位数分箱")
                # 使用中位数分箱：大于中位数为1，否则为0
                df[f'{col}_bin'] = (df[col] > median_val).astype(int)
                new_cat_features.append(f'{col}_bin')
    
    # 4. 组合特征
    # 新增特征：信用历史与贷款金额的交互
    if 'creditHistory' in df.columns and 'loanAmnt' in df.columns:
        df['credit_loan_interaction'] = df['creditHistory'] * df['loanAmnt']
    
    # 5. 更新类别特征列表（将分箱特征加入）
    categorical_features += new_cat_features
    
    # 6. 特征变换：对数转换偏态分布特征
    skewed_features = ['annualIncome', 'loanAmnt', 'installment']
    for feat in skewed_features:
        if feat in df.columns:
            # 添加1避免负值和零值问题
            df[f'log_{feat}'] = np.log1p(df[feat])
    
    return df, (encoders, categorical_features)