# YOLOV2

改进：

- V2版本舍弃Dropout，卷积后全部加入Batch Normalization
- 网络的每一层的输入都做了归一化，收敛相对更容易。
- 经过Batch Normalization处理后的网络会提升2%的mAP
- 从现在的角度来看，Batch Normalization已经成为网络的必备处理

YOLOV2更大的分辨率：

- V1训练时使用的是$224*224$,.测试时使用的$448*448$。
- 可能导致模型水土不服，V2训练时额外又进行了10次$448*448$的微调。
- 使用高分辨率分类器之后，YOLOv2的mAP提升了约4%。



网络结构：

- DarkNet，实际输入为$416*416$
- **没有FC层**，5次降采样 $13*13$
- $1*1$卷积节省了很多参数



网格大小$13*13$



YOLOV2**聚类提取先验框**：
faster-rcnn系列中选择的先验比例都是常规的，但不一定完全适合数据集合。

K-means聚类中的距离：$d(\text { box }, \text { centroids })=1-I O U(\text { box }, \text { centroids })$

![image-20211102135026840](https://ricky1999.oss-cn-beijing.aliyuncs.com/img/image-20211102135026840.png)

K=5的时候，比较折中。所以V2版本有5个先验框。



YOLO-V2——Anchor Box

通过引入anchor boxes 使得预测的box数量更多 （$13*13*n$）

跟faster rcnn系列不同的是先验框并不是直接按照长宽固定比给定。

| without anchor | 69.5mAP | 81%recall |
| -------------- | :------ | --------- |
| with anchor    | 69.2mAP | 88%recall |



YOLOV2直接预测相对位置：


![image-20211102140031776](https://ricky1999.oss-cn-beijing.aliyuncs.com/img/image-20211102140031776.png)



计算公式
$$
\begin{aligned}
b_{x} &=\sigma\left(t_{x}\right)+c_{x} \\
b_{y} &=\sigma\left(t_{y}\right)+c_{y} \\
b_{w} &=p_{w} e^{t_{w}} \\
b_{h} &=p_{h} e^{t_{h}}
\end{aligned}
$$


![image-20211102140612584](https://ricky1999.oss-cn-beijing.aliyuncs.com/img/image-20211102140612584.png)

$\sigma\left(t_{x}\right)$和$\sigma\left(t_{y}\right)$都是相对格子的位置。

![image-20211102141127687](https://ricky1999.oss-cn-beijing.aliyuncs.com/img/image-20211102141127687.png)



### 感受野



![image-20211102141427465](https://ricky1999.oss-cn-beijing.aliyuncs.com/img/image-20211102141427465.png)





为什么使用小的卷积核？

小的卷积所需要的参数会更少，并且卷积过程越多，特征提取会越细致。加入非线性变换也随之增多，还不会增大权重的个数。

![image-20211102142012587](https://ricky1999.oss-cn-beijing.aliyuncs.com/img/image-20211102142012587.png)





最后一层的感受野太大了， 小目标可能丢失了。需要融合之前的特征。

如果只关注最后一层feature map小物体可能丢失。

![image-20211102184023007](https://ricky1999.oss-cn-beijing.aliyuncs.com/img/image-20211102184023007.png)

V2多尺度：
![image-20211102184246943](https://ricky1999.oss-cn-beijing.aliyuncs.com/img/image-20211102184246943.png)





# YOLOV3



- 最大的改进就是网络结构，使其更加适合小目标的检测。
- 特征做的更加精致，融入多持续特征图信息来预测不同规则物体
- 先验框更加丰富，3中scale，每三个规格，一共9种。
- softmax改进，预测多标签任务。



##### 多scale

为了能检测到不同大小的物体，设计了3个scale。

![image-20211102193008857](https://ricky1999.oss-cn-beijing.aliyuncs.com/img/image-20211102193008857.png)

##### scale经典变化方法



左图：图像金字塔；右图：单一的输入

![image-20211102193825674](https://ricky1999.oss-cn-beijing.aliyuncs.com/img/image-20211102193825674.png)

特征金字塔，速度慢；不选



左图：对不同特征分别利用；右图：不同的特征融合后预测。

![image-20211102193950972](https://ricky1999.oss-cn-beijing.aliyuncs.com/img/image-20211102193950972.png)





##### 残差连接

为了更好的特征

**从今天的角度来看，基本所有的网络都用上了残差连接方法。**

V3中也用了resnet思想。堆叠更多的层来进行特征提取。

![image-20211102194550858](https://ricky1999.oss-cn-beijing.aliyuncs.com/img/image-20211102194550858.png)



##### 核心架构

- 没有池化和全连接层，全部卷积
- 下采样通过stride为2实现
- 3种scale，更多先验框
- 基本上将当下经典算法全部融入

![image-20211102195116749](https://ricky1999.oss-cn-beijing.aliyuncs.com/img/image-20211102195116749.png)

![image-20211102195428819](https://ricky1999.oss-cn-beijing.aliyuncs.com/img/image-20211102195428819.png)





先验框设计：

![image-20211102195541745](https://ricky1999.oss-cn-beijing.aliyuncs.com/img/image-20211102195541745.png)

![image-20211102195702225](https://ricky1999.oss-cn-beijing.aliyuncs.com/img/image-20211102195702225.png)

##### softmax层替代

物体检测任务中可能一个物体有多个标签

logistic激活函数来完成，这样能预测每一个类别是/不是



-----------------------------------------

# YOLOV3源码





















































