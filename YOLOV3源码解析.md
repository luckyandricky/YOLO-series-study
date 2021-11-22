# YOLOV3源码解析

**数据集准备**

数据集：COCO2014格式

class.names 改成你任务里面有的类别的名字

train.txt、val.txt写好对应的路径。

custom.data --->修改



训练代码的修改：
![image-20211103115026567](https://ricky1999.oss-cn-beijing.aliyuncs.com/img/image-20211103115026567.png)



训练时不会一次性读取所有的数据。是在训练的时候实时的读取的。

Pytorch输入网络的数据需要转化为Tensor的形式。





## Darknet

网络模型定义：


```python
class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.yolo_layers = [layer[0]
                            for layer in self.module_list if isinstance(layer[0], YOLOLayer)]
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)
```

加载配置文件。yolov3.cfg

写forward函数。

```python
    def forward(self, x):
        img_size = x.size(2)
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                combined_outputs = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
                group_size = combined_outputs.shape[1] // int(module_def.get("groups", 1))
                group_id = int(module_def.get("group_id", 0))
                x = combined_outputs[:, group_size * group_id : group_size * (group_id + 1)] # Slice groupings used by yolo v4
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                x = module[0](x, img_size)
                yolo_outputs.append(x)
            layer_outputs.append(x)
        return yolo_outputs if self.training else torch.cat(yolo_outputs, 1)

```

### Route层

是一个拼接层

![image-20211116192045185](https://ricky1999.oss-cn-beijing.aliyuncs.com/imgs/image-20211116192045185.png)



### Shortcut层

并不是维度凭借，是加法操作。

### Yolo层

yolov3有三个yolo层。最复杂的一个层。

![image-20211116195451822](https://ricky1999.oss-cn-beijing.aliyuncs.com/imgs/image-20211116195451822.png)

![image-20211116195725874](https://ricky1999.oss-cn-beijing.aliyuncs.com/imgs/image-20211116195725874.png)

首先定义一些变量，然后构建YOLO层。

yolo_layer = YOLOLayer(......)

```python
class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors #先验框大小
        self.num_anchors = len(anchors) #先验框个数
        self.num_classes = num_classes #分类的个数
        self.ignore_thres = 0.5 #阈值
        self.mse_loss = nn.MSELoss() #损失函数
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size
```

![image-20211116200828053](https://ricky1999.oss-cn-beijing.aliyuncs.com/imgs/image-20211116200828053.png)

可以看到forward的时候，shortcut层做的是加法。



### 预测结果计算

YOLO层的一个前向传播

```python
def forward(self, x, targets=None, img_dim=None):
    # Tensors for cuda support
    print (x.shape)  
    FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor #设置GPU还是CPU格式
    LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
    ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

    self.img_dim = img_dim #
    num_samples = x.size(0)
    grid_size = x.size(2)

    prediction = (
        x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
        .permute(0, 1, 3, 4, 2)
        .contiguous()
    )#改变了x的shape
    print (prediction.shape)
```

![image-20211116202902517](https://ricky1999.oss-cn-beijing.aliyuncs.com/imgs/image-20211116202902517.png)







![image-20211116203113373](https://ricky1999.oss-cn-beijing.aliyuncs.com/imgs/image-20211116203113373.png)



输出值：注意最后一个类别是使用的**sigmoid函数**而不是softmax。预测80个类别中的每一个是否属于。

预测出来的x、y是相对位置。

![image-20211116204452659](https://ricky1999.oss-cn-beijing.aliyuncs.com/imgs/image-20211116204452659.png)

黑点相对于绿色点的相对位置。

![image-20211116204543122](https://ricky1999.oss-cn-beijing.aliyuncs.com/imgs/image-20211116204543122.png)

将相对位置转换为绝对位置。

![image-20211116210554824](https://ricky1999.oss-cn-beijing.aliyuncs.com/imgs/image-20211116210554824.png)

最后将boxes还原到原始图中。

### 损失计算

![image-20211116211040532](https://ricky1999.oss-cn-beijing.aliyuncs.com/imgs/image-20211116211040532.png)

![image-20211116212646783](https://ricky1999.oss-cn-beijing.aliyuncs.com/imgs/image-20211116212646783.png)

```python
考虑前景和背景
obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)  # obj，anchor包含物体, 即为1，默认为0 考虑前景
noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1) # noobj, anchor不包含物体, 则为1，默认为1 考虑背景
```

```python
# Set noobj mask to zero where iou exceeds ignore threshold
for i, anchor_ious in enumerate(ious.t()): # IOU超过了指定的阈值就相当于有物体了
    noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0
```

超过阈值，也相当于有物体。

![image-20211116222512011](https://ricky1999.oss-cn-beijing.aliyuncs.com/imgs/image-20211116222512011.png)

计算与真实值的损失：
x[obj_mask], tx[obj_mask] 只计算有目标的。所以要用obj_mask中为1的就表示有物体。

![image-20211121204432091](C:\Users\赵宪锐\AppData\Roaming\Typora\typora-user-images\image-20211121204432091.png)

前景损失和背景损失直接用bce_loss计算就可以。取值范围已经是0-1了直接用bce_loss

```python
self.bce_loss(pred_conf[obj_mask], tconf[obj_mask]) 
```

最麻烦的就是标签格式的转换。



### 反向传播：自动求解

























