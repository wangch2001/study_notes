# Loss 手撕

按**面试最常见 + 手撕难度适中**整理。重点记住两件事：**它解决什么问题**，以及**核心公式怎么写成代码**。

```python
import torch
import torch.nn.functional as F
```

| Loss | 常见用途 |
| --- | --- |
| MSE | 回归 |
| BCE | 二分类 |
| Cross Entropy | 多分类、LLM token prediction |
| KL Divergence | 分布对齐、蒸馏、RLHF |
| BPR Loss | 推荐排序 |
| Pairwise Hinge Loss | 排序 |
| Triplet Loss | 度量学习、人脸/检索 |
| Contrastive Loss | 对比学习 |
| InfoNCE | 对比学习、双塔召回 |
| Huber Loss | 鲁棒回归 |

---

## 1. MSE Loss

用于**回归问题**。

$$
L=\frac{1}{N}\sum_i(\hat y_i-y_i)^2
$$

```python
def mse_loss(pred, target):
    return ((pred - target) ** 2).mean()
```

典型问题：房价预测、连续值预测。

---

## 2. BCE Loss

用于**二分类**，例如是否点击、是否购买、是否是垃圾邮件。

$$
L=-[y\log p+(1-y)\log(1-p)]
$$

```python
def bce_loss(pred, target):
    # pred 是经过 sigmoid 后的概率；target 为 0/1
    eps = 1e-8
    pred = torch.clamp(pred, eps, 1 - eps)
    loss = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
    return loss.mean()
```

工程中更常用：

```python
# 输入 logits，不要先 sigmoid，数值更稳定
loss = F.binary_cross_entropy_with_logits(logits, target)
```

---

## 3. Cross Entropy Loss

用于**多分类**，例如猫/狗/鸟分类；LLM 预测下一个 token。

$$
L=-\log p_y
$$

其中 `p_y` 是正确类别的概率。

```python
def cross_entropy_loss(logits, labels):
    # logits: [B, C]，未经 softmax
    # labels: [B]，正确类别下标
    log_probs = F.log_softmax(logits, dim=-1)
    target_log_prob = log_probs[torch.arange(logits.size(0)), labels]
    return -target_log_prob.mean()
```

最核心就是：`-log(p_correct)`。

---

## 4. KL Divergence

用于比较**两个概率分布的差异**：知识蒸馏、RLHF、模型分布约束、VAE。

$$
KL(P||Q)=\sum_iP_i\log\frac{P_i}{Q_i}
$$

```python
def kl_loss(p_logits, q_logits):
    # 计算 KL(P || Q)
    log_p = F.log_softmax(p_logits, dim=-1)
    log_q = F.log_softmax(q_logits, dim=-1)
    p = log_p.exp()
    kl = p * (log_p - log_q)
    return kl.sum(dim=-1).mean()
```

核心：`p * (log_p - log_q)`。

---

## 5. BPR Loss

推荐系统中常见的**排序损失**。目标是正样本分数大于负样本分数。

$$
L=-\log\sigma(s_{pos}-s_{neg})
$$

```python
def bpr_loss(pos_score, neg_score):
    diff = pos_score - neg_score
    return -F.logsigmoid(diff).mean()

# 等价写法
def bpr_loss_softplus(pos_score, neg_score):
    return F.softplus(-(pos_score - neg_score)).mean()
```

---

## 6. Pairwise Hinge Loss

也是**排序问题**。要求正样本分数至少比负样本高 `margin`：

$$
s_{pos}\ge s_{neg}+m
$$

$$
L=\max(0,m-s_{pos}+s_{neg})
$$

```python
def hinge_loss(pos_score, neg_score, margin=1.0):
    return torch.relu(margin - pos_score + neg_score).mean()
```

当 `pos=5, neg=2, margin=1` 时，loss 为 `max(0, 1 - 5 + 2) = 0`，说明排序已足够好。

---

## 7. Triplet Loss

常用于人脸识别、图片检索、Embedding 学习、语义检索。

三个样本：`anchor`、`positive`、`negative`。要求 anchor 离 positive 更近，离 negative 更远。

$$
L=\max(0,d(a,p)-d(a,n)+m)
$$

```python
def triplet_loss(anchor, positive, negative, margin=1.0):
    pos_dist = ((anchor - positive) ** 2).sum(dim=-1)
    neg_dist = ((anchor - negative) ** 2).sum(dim=-1)
    return torch.relu(pos_dist - neg_dist + margin).mean()
```

---

## 8. Contrastive Loss

经典 Siamese Network 常用。相似样本拉近，不相似样本推远。

约定 `label=1` 表示相似，`label=0` 表示不相似：

$$
L=yd^2+(1-y)\max(0,m-d)^2
$$

```python
def contrastive_loss(x1, x2, label, margin=1.0):
    dist = torch.norm(x1 - x2, dim=-1)
    pos_loss = label * dist ** 2
    neg_loss = (1 - label) * torch.relu(margin - dist) ** 2
    return (pos_loss + neg_loss).mean()
```

---

## 9. InfoNCE

对 **LLM / 推荐 / 多模态 / 双塔** 都很重要。

一个 batch 中 `user_i` 与 `item_i` 是正样本，对角线外的配对都是负样本；目标是让相似度矩阵的对角线分数最大。

```python
def info_nce_loss(x, y, temperature=0.07):
    # x, y: [B, D]；第 i 行 x 与第 i 行 y 是正样本
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    logits = x @ y.T / temperature     # [B, B]
    labels = torch.arange(x.size(0), device=x.device)
    return F.cross_entropy(logits, labels)
```

---

## 10. Huber Loss

用于**回归问题**，但比 MSE 对异常值更鲁棒：小误差像 MSE，大误差像 MAE。

$$
L_\delta(e)=
\begin{cases}
\frac12e^2,& |e|\le\delta\\
\delta(|e|-\frac12\delta),& |e|>\delta
\end{cases}
$$

```python
def huber_loss(pred, target, delta=1.0):
    error = torch.abs(pred - target)
    small = 0.5 * error ** 2
    large = delta * (error - 0.5 * delta)
    loss = torch.where(error <= delta, small, large)
    return loss.mean()
```

---

## 面试最值得优先背的

时间有限时，至少练到能直接写：

```text
1. MSE
2. BCE
3. Cross Entropy
4. KL Divergence
5. BPR Loss
6. Triplet Loss
7. InfoNCE
```

如果面的是**推荐 / 搜索 / LLM 算法岗**，可优先复习：

```text
Cross Entropy → BCE → BPR → InfoNCE → KL → Triplet → MSE
```

高频追问：**BPR、Triplet Loss、InfoNCE 有什么区别？**

- BPR：一个正样本分数要高于一个负样本分数，常用于推荐排序。
- Triplet：约束 anchor 到正、负样本的距离差，常用于 embedding 度量学习。
- InfoNCE：一个正样本同时和 batch 中多个负样本竞争，常用于双塔召回和对比学习。
