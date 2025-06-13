# 贷款逾期预测系统

## 项目结构

```bash
proj/
├── data/                  # 数据存储目录
│   ├── train.csv          # 训练数据集 (80万条)
│   ├── testA.csv          # 测试集A (20万条)
│   └── sample_submit.csv  # 提交文件示例
│
├── src/                   # 源代码目录
│   ├── data_preprocessing.py  # 数据预处理模块
│   ├── feature_engineering.py # 特征工程模块
│   ├── model_training.py      # 模型训练模块
│   └── prediction.py          # 预测模块
│
├── models/                # 模型存储目录
│   └── lgb_model.pkl      # 训练好的LightGBM模型
│
├── outputs/               # 输出结果目录
│   └── submission.csv     # 生成的预测结果文件
│
├── main.py                # 项目主入口文件
├── requirements.txt       # Python依赖库列表
└── README.md              # 项目说明文档
```

## 项目部署

### 安装步骤

1. **克隆项目仓库**
```bash
git clone https://github.com/yourusername/loan-default-prediction.git
cd loan-default-prediction
```

2. **安装依赖库**
```bash
pip install -r requirements.txt
```

3. **准备数据文件**
   -https://tianchi.aliyun.com/competition/entrance/531830/information
   - 将训练数据(train.csv)和测试数据(testA.csv)放入`data/`目录
   - 确保文件结构与项目结构一致

## 运行方法

### 完整流程运行
执行全流程：数据预处理 → 特征工程 → 模型训练 → 预测生成
```bash
python main.py
```

### 分模块运行

1. **单独运行数据预处理**
```bash
python src/data_preprocessing.py
```

2. **单独运行特征工程**
```bash
python src/feature_engineering.py
```

3. **单独训练模型**
```bash
python src/model_training.py
```

4. **单独生成预测结果**
```bash
python src/prediction.py
```

### 结果查看
- 训练完成后，模型保存在`models/`目录
- 预测结果生成在`outputs/submission.csv`
- 结果文件格式符合比赛提交要求
