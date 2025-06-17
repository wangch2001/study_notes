**1.总览**
面临的问题
- 规模大：召回时负采样，线上时采用hash映射和最近邻检索
- 新鲜度：增加example age这一参数
- 噪声：更细致的特征工程，采用神经网络进行召回和排序
**2.训练数据选择**
训练样本来源于全部的YouTube观看记录
用户看完了的视频作为正样本，负样本采取负采样策略
训练数据中对于每个用户选取相同的样本数，防止活跃用户影响效果
历史搜索tokens随机打乱，防止最后一次搜索对结果影响过大
采用最后一次点击做测试集，防止出现数据穿越

example age 特征：用这个特征来捕捉视频的生命周期，对新视频采用更多推荐
**3.候选集生成（召回）**
对用户的观看序列和搜索token做embedding，与其他信息(地理位置、年龄等)做拼接后输入三层神经网络。
利用用户embedding和视频embedding的相似度来衡量用户点击视频的概率，召回时采用最近邻搜索来计算相似度。

**4. 排序模型**
对于召回的到的几百个候选集，经过拼接后输入三层DNN网络进行优化
impression video ID embedding: 当前要计算的video的embedding
watched video IDs average embedding: 用户观看过的最后N个视频embedding的average pooling
language embedding: 用户语言的embedding和当前视频语言的embedding
time since last watch: 自上次观看同channel视频的时间
#previous impressions: 该视频已经被曝光给该用户的次数
优化目标为:加权交叉熵,即正样本*观看时长的交叉熵函数，用p/1-p来估计观看时长