from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 正常显示中文
plt.rcParams["axes.unicode_minus"] = False     # 正常显示负号

data=fetch_california_housing()
X=data.data
Y=data.target

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

scaler=StandardScaler()
x_train_scaler=scaler.fit_transform(x_train)
x_test_scaler=scaler.transform(x_test)

#正规方程
estimator=LinearRegression()
estimator.fit(x_train,y_train)

y_pre=estimator.predict(x_test)

mse=mean_squared_error(y_test,y_pre)

#梯度下降
sgd=SGDRegressor(max_iter=1000,tol=1e-3,random_state=42)
sgd.fit(x_train_scaler,y_train)
y_pre_sgd=sgd.predict(x_test_scaler)
mse_sgd=mean_squared_error(y_test,y_pre_sgd)

#岭回归
ridge = Ridge(alpha=1.0)
ridge.fit(x_train_scaler, y_train)
y_pred_ridge = ridge.predict(x_test_scaler)

import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ======================
# 优化版：真实值 vs 预测值
# ======================
plt.figure(figsize=(12, 5))

# 左图：线性回归（清晰散点）
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pre, s=3, alpha=0.2, c='#1f77b4')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=1.5)
plt.xlabel('真实房价')
plt.ylabel('预测房价')
plt.title('线性回归：真实值 vs 预测值')
plt.grid(alpha=0.3)

# 右图：SGD（用六边形密度图，专治点密集）
plt.subplot(1, 2, 2)
plt.hexbin(y_test, y_pre_sgd, gridsize=50, cmap='Blues', mincnt=1)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=1.5)
plt.xlabel('真实房价')
plt.ylabel('预测房价')
plt.title('SGD：真实值 vs 预测值（密度图）')
plt.colorbar(label='点密度')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ======================
# 误差分布（优化版）
# ======================
plt.figure(figsize=(8, 5))
error_lr = y_test - y_pre
error_sgd = y_test - y_pre_sgd

plt.hist(error_lr, bins=50, alpha=0.4, label='线性回归误差', color='blue')
plt.hist(error_sgd, bins=50, alpha=0.4, label='SGD误差', color='red')
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel('预测误差')
plt.ylabel('数量')
plt.title('模型误差分布')
plt.legend()
plt.grid(alpha=0.3)
plt.show()