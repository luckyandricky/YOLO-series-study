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







