def sigmoid(x):
    return 1/(1+np.exp(x))
def cross_entropy(y_pred,y_true):
    y_pred_prob=sigmoid(y_pred)
    loss=-(np.log(y_pred_prob)*y_true)+(1-y_true)*(1-np.log(y_pred_prob))
    
    return np.mean(loss)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))  # 防止溢出
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def cross_entropy(y_pred,y_true):
    y_pred_prob = softmax(y_pred)

    # 计算每个样本的真实标签与预测概率分布的交叉熵
    loss_per_sample = -np.sum(y_true * np.log(y_pred_prob), axis=-1)

    # 返回所有样本的平均交叉熵损失
    return np.mean(loss_per_sample)
