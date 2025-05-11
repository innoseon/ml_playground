import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np # 假设你有数据加载和预处理

# 1. 假设你已经加载并准备好了数据 X (特征) 和 y (标签)
# 例如:
# data = pd.read_csv('your_dataset.csv')
# X = data.drop('target_column', axis=1)
# y = data['target_column']
# 进行必要的预处理...
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 这里用一个简单的示例数据代替
X_train = np.random.rand(100, 5)
y_train = np.random.randint(0, 2, 100)
X_test = np.random.rand(20, 5)
y_test = np.random.randint(0, 2, 20)

print("--- 数据准备阶段 ---")
print(f"X_train 的形状: {X_train.shape}")
print(f"y_train 的形状: {y_train.shape}")
print(f"X_test 的形状: {X_test.shape}")
print(f"y_test 的形状: {y_test.shape}")
print("\n")

# 2. 实例化模型 (使用默认参数)
print("--- 模型实例化阶段 ---")
model = xgb.XGBClassifier()
print("XGBoost 分类器已实例化。")
print("\n")

# 3. 训练模型
print("--- 模型训练阶段 ---")
model.fit(X_train, y_train)
print("模型训练完成。")
print("\n")

# 4. 进行预测
print("--- 模型预测阶段 ---")
y_pred = model.predict(X_test)
print(f"对测试集进行预测的前5个结果: {y_pred[:5]}")
print("\n")

# 5. 评估模型
print("--- 模型评估阶段 ---")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100.0:.2f}%")