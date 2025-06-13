import pandas as pd
import numpy as np
import re
from datetime import datetime
from config import PATH_CONFIG

def map_employment_length(x):
    # 处理特殊情况
    if pd.isnull(x) or x is None or str(x).lower() in ['nan', 'na', '']:
        return 0
    elif '10+' in str(x) or '10 years' in str(x) or '10+ years' in str(x):
        return 10
    elif '<' in str(x) or 'less' in str(x).lower():
        # 提取数字部分
        match = re.search(r'\d+', str(x))
        return int(match.group()) if match else 0
    else:
        try:
            # 尝试提取数字
            return int(re.search(r'\d+', str(x)).group())
        except:
            return 0

def parse_credit_line(date_str):
    if pd.isnull(date_str) or date_str is None or str(date_str).lower() in ['nan', 'na', '']:
        return 1970
    
    date_str = str(date_str).strip()
    
    # 尝试多种日期格式
    formats = [
        '%b-%Y', '%b %Y', '%Y-%b', '%Y %b',
        '%B-%Y', '%B %Y', '%Y-%B', '%Y %B',
        '%m-%Y', '%m/%Y', '%Y-%m', '%Y/%m'
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.year
        except:
            continue
    
    # 尝试提取4位数字
    year_match = re.search(r'\d{4}', date_str)
    if year_match:
        return int(year_match.group())
    
    # 尝试提取2位数字
    year_match = re.search(r'\d{2}', date_str)
    if year_match:
        year = int(year_match.group())
        return 1900 + year if year > 50 else 2000 + year
    
    return 1970

def preprocess_data(df, is_train=True):
    # 重命名列以处理可能的拼写错误
    rename_dict = {}
    if 'earliesCreditLine' in df.columns:
        rename_dict['earliesCreditLine'] = 'earliestCreditLine'
    if 'employmentlength' in df.columns:
        rename_dict['employmentlength'] = 'employmentLength'
    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)
    
    # 1. 处理employmentLength
    if 'employmentLength' in df.columns:
        df['employmentLength'] = df['employmentLength'].apply(map_employment_length)
    else:
        df['employmentLength'] = 0  # 如果列不存在则创建并填充0
    
    # 2. 处理日期特征
    if 'issueDate' in df.columns:
        df['issueDate'] = pd.to_datetime(df['issueDate'], errors='coerce')
        df['issueYear'] = df['issueDate'].dt.year
        df['issueMonth'] = df['issueDate'].dt.month
        
        # 处理NaN值
        mode_year = df['issueYear'].mode()[0] if not df['issueYear'].isnull().all() else 2015
        df['issueYear'] = df['issueYear'].fillna(mode_year)
        df['issueMonth'] = df['issueMonth'].fillna(df['issueMonth'].mode()[0])
    else:
        df['issueYear'] = 2015
        df['issueMonth'] = 1
    
    # 3. 处理earliestCreditLine
    if 'earliestCreditLine' in df.columns:
        df['earliestCreditYear'] = df['earliestCreditLine'].apply(parse_credit_line)
    else:
        df['earliestCreditYear'] = 2000
    
    # 4. 计算信用历史
    df['creditHistory'] = df['issueYear'] - df['earliestCreditYear']
    df['creditHistory'] = df['creditHistory'].clip(0, 50)  # 处理异常值
    
    # 5. 删除原始日期列（如果存在）
    cols_to_drop = []
    if 'issueDate' in df.columns:
        cols_to_drop.append('issueDate')
    if 'earliestCreditLine' in df.columns:
        cols_to_drop.append('earliestCreditLine')
    
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
    
    # 6. 处理缺失值（更智能的方法）
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            # 使用中位数填充数值特征
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
    
    # 7. 确保所有配置的特征都存在
    required_features = ['id'] + PATH_CONFIG['features'] + ['creditHistory', 'issueYear', 'issueMonth']  # 添加id列
    if is_train:
        required_features.append('isDefault')  # 训练集需要标签列
    
    # 添加缺失的列并用0填充
    for feature in required_features:
        if feature not in df.columns:
            df[feature] = 0
    
    # 8. 选择需要的列
    df = df[required_features]
    
    return df

def get_train_test_data():
    # 读取数据
    train_df = pd.read_csv(PATH_CONFIG['train_data'])
    test_df = pd.read_csv(PATH_CONFIG['test_data'])
    
    # 打印列名用于调试
    print("训练集列名:", train_df.columns.tolist())
    print("测试集列名:", test_df.columns.tolist())
    
    # 数据预处理
    train_processed = preprocess_data(train_df, is_train=True)
    test_processed = preprocess_data(test_df, is_train=False)
    
    return train_processed, test_processed