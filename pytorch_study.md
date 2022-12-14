# pytorch_study

这是一个有关pytorch的学习库里面有着一些适合新手学习的模型，该仓库在不断更新，预计在2022.10.1号前，使用pytorch将一些基础模型进行复现。

目前已经复现的模型有：

- pytorch解决回归问题
- MNIST手写数字识别
- CIFAR10数据集分类

## pytorch解决回归问题

## MNIST手写数字识别

训练100次结果如下：

![image-20220922215614508](https://makedown-1304519375.cos.ap-beijing.myqcloud.com/makedown/image-20220922215614508.png)

模型已完全收敛，正确率稳定在0.978，模型架构如下：

```python
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
minist_module                            [32, 10]                  --
├─Conv2d: 1-1                            [32, 10, 24, 24]          260
├─Conv2d: 1-2                            [32, 1, 10, 10]           91
├─Linear: 1-3                            [32, 1000]                101,000
├─Linear: 1-4                            [32, 500]                 500,500
├─Linear: 1-5                            [32, 10]                  5,010
==========================================================================================
Total params: 606,861
Trainable params: 606,861
Non-trainable params: 0
Total mult-adds (M): 24.49
==========================================================================================
Input size (MB): 0.10
Forward/backward pass size (MB): 1.89
Params size (MB): 2.43
Estimated Total Size (MB): 4.41
==========================================================================================
```

数据集问题：

#在MNIST/raw文件下，数据集解压即可。

## CIFAR10数据集分类

VGG的模型架构如下：

```python
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
vgg_module                               [32, 10]                  --
├─Conv2d: 1-1                            [32, 64, 32, 32]          1,792
├─Conv2d: 1-2                            [32, 64, 32, 32]          36,928
├─Conv2d: 1-3                            [32, 64, 32, 32]          (recursive)
├─MaxPool2d: 1-4                         [32, 64, 16, 16]          --
├─Conv2d: 1-5                            [32, 128, 16, 16]         73,856
├─Conv2d: 1-6                            [32, 128, 16, 16]         147,584
├─MaxPool2d: 1-7                         [32, 128, 8, 8]           --
├─Conv2d: 1-8                            [32, 256, 8, 8]           295,168
├─Conv2d: 1-9                            [32, 256, 8, 8]           590,080
├─MaxPool2d: 1-10                        [32, 256, 4, 4]           --
├─Conv2d: 1-11                           [32, 512, 4, 4]           1,180,160
├─Conv2d: 1-12                           [32, 512, 4, 4]           2,359,808
├─MaxPool2d: 1-13                        [32, 512, 2, 2]           --
├─Linear: 1-14                           [32, 1024]                2,098,176
├─Linear: 1-15                           [32, 512]                 524,800
├─Linear: 1-16                           [32, 10]                  5,130
==========================================================================================
Total params: 7,313,482
Trainable params: 7,313,482
Non-trainable params: 0
Total mult-adds (G): 8.00
==========================================================================================
Input size (MB): 0.39
Forward/backward pass size (MB): 63.31
Params size (MB): 29.25
Estimated Total Size (MB): 92.96
==========================================================================================
```



训练结果如下：

![image-20220924112653271](https://makedown-1304519375.cos.ap-beijing.myqcloud.com/makedown/image-20220924112653271.png)

#在论文中这个模型的正确概率经过微调能达到97.6%。但不经过微调直接进行训练正确率只能达到87%左右，若想提高正确率可以通过加入反转等操作，可以显著提高模型正确率。

数据集问题：

