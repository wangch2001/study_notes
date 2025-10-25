import numpy as np

# sigmoid激活函数
def sigmoid(x):
    return 1/(1+np.exp(-x))

# softmax激活函数
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))  # 防止溢出
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

# cross_entropy（二分类版本）
def cross_entropy(y_pred,y_true):
    y_pred_prob=sigmoid(y_pred)
    loss = -(y_true * np.log(y_pred_prob) + (1-y_true) * np.log(1-y_pred_prob))
    return np.mean(loss)

# cross_entropy（多分类版本）
def cross_entropy(y_pred,y_true):
    y_pred_prob = softmax(y_pred)
    # 计算每个样本的真实标签与预测概率分布的交叉熵
    loss_per_sample = -np.sum(y_true * np.log(y_pred_prob), axis=-1)
    # 返回所有样本的平均交叉熵损失
    return np.mean(loss_per_sample)

# 均方误差 (Mean Squared Error, MSE)
def mean_squared_error(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

# 均方根误差 (Root Mean Squared Error, RMSE)
def root_mean_squared_error(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))

# 平均绝对误差 (Mean Absolute Error, MAE)
def mean_absolute_error(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))

# 示例验证
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.1, 7.8])

print("MSE :", mean_squared_error(y_pred, y_true))
print("RMSE:", root_mean_squared_error(y_pred, y_true))
print("MAE :", mean_absolute_error(y_pred, y_true))
