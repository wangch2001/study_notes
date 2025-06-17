auc：
直接定义：ROC线下面积,即假阳性（横轴）和真阳性随着预测阈值变化而截断的过程
物理意义：随机选取一对正负样本，模型对正样本打分大于负样本的概率
```py
def auc(y_true,y_pred):
    fz=0
    fm=0
    for i in range(0,len(y_true)-1):
        for j in range(i+1,len(y_true)):
            if(y_true[i]!=y_true[j]):
                fm+=1
                if(y_true[i]>y_true[j] and y_pred[i]>y_pred[j]) or (y_true[i]<y_true[j] and y_pred[i]<y_pred[j]):
                fz+=1
    return fz/fm
```
优点：比较稳定，受数据集影响小，能够真实反应模型的性能；
缺点：指标太过笼统，不能反应acc、rec等指标；只关注正负样本的关系，正样本负样本内部的关系

改良：阿里的GAUC
用户广告之间的排序是个性化的，不同用户的排序结果不太好比较，这可能导致全局auc并不能反映真实情况。
GAUC计算每个用户的auc，然后加权平均
