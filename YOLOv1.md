# YOLO

YOLO是Onestage算法。速度快，准确率相对低。

Two-stage网络速度慢，效果好。通用框架MaskRcnn。

实时性任务需要使用YOLO。

### 评价指标

map：综合衡量准确率和召回率的出的结果。

准确率：
$$
\text { Precision }=\frac{T P}{T P+F P}
$$
Recall：
$$
\text { Recall }=\frac{T P}{T P+F N}
$$


IOU：交并比
$$
\mathrm{IoU}=\frac{\text { Area of Overlap }}{\text { Area of Union }}
$$
![image-20211030202710310](C:\Users\Administrator.WIN10-805071653\AppData\Roaming\Typora\typora-user-images\image-20211030202710310.png)



### YOLOV1

**回归问题，一个CNN搞定**

#### **整体网络架构**

输入图像的大小不能变

![image-20211030203415386](https://ricky1999.oss-cn-beijing.aliyuncs.com/img/image-20211030203415386.png)

输出是一个$S*S*30$的张量。

![image-20211030203448462](https://ricky1999.oss-cn-beijing.aliyuncs.com/img/image-20211030203448462.png)

![image-20211030203657727](https://ricky1999.oss-cn-beijing.aliyuncs.com/img/image-20211030203657727.png)

损失函数定义:

![image-20211030203152890](https://ricky1999.oss-cn-beijing.aliyuncs.com/img/image-20211030203152890.png)





























