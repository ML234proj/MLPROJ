import pandas as pd
import os
from config import PATH_CONFIG

def generate_submission(test_ids, predictions):
    # 创建提交数据框
    submission = pd.DataFrame({
        'id': test_ids,
        'isDefault': predictions
    })
    
    # 确保输出目录存在
    os.makedirs(PATH_CONFIG['output_dir'], exist_ok=True)
    
    # 保存结果
    output_path = f"{PATH_CONFIG['output_dir']}/{PATH_CONFIG['result_name']}"
    submission.to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}")
    
    return submission