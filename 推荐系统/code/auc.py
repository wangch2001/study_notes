# 方法一：梯形法：排序 + 积分近似（统计秩次法）
# 把样本按预测分数降序排列；
# 当遍历到一个负样本时，n_cum 中的值代表：在它之前有多少个正样本；
# 所有这些正样本数相加，就得到了模型把正样本排在负样本前的数量；
# 除以所有正负样本对的总数，就得到 AUC。
# 时间复杂度O(nlogn)
import numpy as np

def auc(y_true, y_pred):
    # 按预测得分从大到小排序
    sorted_idx = np.argsort(y_pred)[::-1]
    y_true_sorted = np.array(y_true)[sorted_idx]

    # 计算正负样本数
    pos_num = np.sum(y_true_sorted)
    neg_num = len(y_true_sorted) - pos_num

    # 累计正样本数（相当于每个点的TP）
    n_cum = np.cumsum(y_true_sorted)

    # 对于每个负样本位置，计算其之前出现的正样本数
    auc = np.sum(n_cum[y_true_sorted == 0]) / (neg_num * pos_num)

    return auc

y_true = [0, 0, 1, 1]
y_pred = [0.1, 0.4, 0.35, 0.8]

print(auc(y_true, y_pred))


# 方法二：“概率排名”定义法
# AUC = 正确排序的正负样本对数/正负样本对总数
# 直观，容易理解，复杂度O(n^2)
def auc(y_true, y_pred):
    fz = 0.0  # 分子
    fm = 0.0  # 分母
    n = len(y_true)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if y_true[i] != y_true[j]:  # 一正一负的样本对
                fm += 1  # 总的正负样本对数 +1
                # 情况1：预测顺序正确
                if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
                        (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                    fz += 1  # 如果预测顺序正确
                # 情况2：预测得分相等
                elif y_pred[i] == y_pred[j]:
                    fz += 0.5
    return fz / fm if fm > 0 else 0

y_true = [0, 0, 1, 1]
y_pred = [0.1, 0.4, 0.35, 0.8]
print(auc(y_true, y_pred))

