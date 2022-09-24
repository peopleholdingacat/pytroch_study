import torch
from torch.utils.data import  Dataset,DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import *
import pickle
from torchinfo import summary
from sklearn.model_selection import train_test_split
import os
config={
    "batch_size":32,
    "epoch":100,
    "device":"ADAPTIVE"
}
class datasets(Dataset):
    def __init__(self,x_data,y_data):
       self.x=x_data
       #self.y=torch.zeros((len(y_data), 10)).scatter_(1, torch.tensor(y_data).long().reshape(-1, 1), 1)
       self.y=y_data
       self.len=len(self.x)

    def __getitem__(self, idx):

       return self.x[idx],self.y[idx]

    def __len__(self):
        return self.len

#torchvision导入cifar10
# cifar10 = torchvision.datasets.CIFAR10(
#     root='datasets',
#     train=True,
#     download=True
# )
# cifar10_test = torchvision.datasets.CIFAR10(
#     root='datasets',
#     train=False,
#     download=True
# )
class cifar10_dataset:
    def __init__(self):
        self.data_path="datasets/cifar-10-batches-py"
        self.file_list=None
        self.root=None
        self.num_cases_per_batch=10000
        self.label_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.num_vis=3072
        self.batch_label=[]
        self.labels=[]
        self.data=[]
        self.filenames=[]
        self.dataset_make()

        pass
    def dataset_make(self):

        for root, dirs, files in os.walk(self.data_path):

           self.file_list=files
           self.root=root
        for i in self.file_list:
            with open(self.root+"/"+i, 'rb') as lbpath:  # rb表示的是读取二进制数据
                data = pickle.load(lbpath, encoding='latin1')
                """
                输出结果为:dict_keys(['batch_label', 'labels', 'data', 'filenames'])
                """
                for k in range(self.num_cases_per_batch):

                    self.batch_label.append(data["batch_label"])
                    self.labels.append(data["labels"][k])
                    self.data.append(data["data"][k])
                    self.filenames.append(data["filenames"][k])

    def DataLoader_make(self):

        pic=torch.tensor(self.data).view(-1,3,32,32)
        X_train, X_test, Y_train, Y_test = train_test_split(
            pic, self.labels, test_size=0.20, random_state=42)
        dataset_train=datasets(X_train/255.0+0.1,Y_train)
        dataset_test=datasets(X_test/255.0+0.1,Y_test)
        train_loader = DataLoader(
            dataset=dataset_train,
            batch_size=config["batch_size"],
            shuffle=True,
        )
        test_loader = DataLoader(
            dataset=dataset_test,
            batch_size=config["batch_size"],
            shuffle=True,
        )
        return train_loader,test_loader
    def show_pic(self,img,label,is_only=True):
        """
              :param img: 图片的数组形式(一个batch)batch的大小必须为32
              :param label: 图片的标签
              :return:
              """
        if is_only:
            # 将一组图片组合成一张


            plt.imshow(img)
            plt.xlabel(self.label_names[label.data])
            plt.show()
        else:
            plt.figure(figsize=(85, 45))
            for i in range(32):
                plt.subplot(8, 4, i + 1)
                plt.imshow(img[i])
                # plt.xlabel(label[i].data)
            plt.tight_layout()  # 自动调整子批次参数，使子批次适合图形区域
            plt.show()
dataset=cifar10_dataset()
train_loader,test_loader=dataset.DataLoader_make()
# """
# 展示图片
# """
images, labels = next(iter(train_loader))
print(images.shape)
print(labels.shape)
dataset.show_pic(images[0],labels[0],is_only=True)
# class vgg_module(torch.nn.Module):
#     """
#     这个没有使用残差结构，导致模型训练效果不太好，即loss不下降的现象
#
#     """
#     def __init__(self):
#         super().__init__()  # 继承父类的函数
#         self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3),padding=1)
#         self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3),padding=1)
#         self.max_pool1=torch.nn.MaxPool2d(kernel_size=(2,2),stride=2)
#         self.conv3=torch.nn.Conv2d(64,128,kernel_size=(3,3),padding=1)
#         self.conv4=torch.nn.Conv2d(128,128,kernel_size=(3,3),padding=1)
#         self.max_pool2=torch.nn.MaxPool2d(kernel_size=(2,2),stride=2)
#         self.conv5=torch.nn.Conv2d(128,256,kernel_size=(3,3),padding=1)
#         self.conv6=torch.nn.Conv2d(256,256,kernel_size=(3,3),padding=1)
#         self.max_pool3=torch.nn.MaxPool2d(kernel_size=(2,2),stride=2)
#         self.conv7=torch.nn.Conv2d(256,512,kernel_size=(3,3),padding=1)
#         self.conv8=torch.nn.Conv2d(512,512,kernel_size=(3,3),padding=1)
#         self.max_pool4=torch.nn.MaxPool2d(kernel_size=(2,2),stride=2)
#         self.fn1=torch.nn.Linear(2048,1024)
#         self.fn2=torch.nn.Linear(1024,512)
#         self.fn3 = torch.nn.Linear(512, 10)
#     def forward(self,inputs):
#         x_shape=inputs.size(0)
#
#         x=self.conv1(inputs)
#         x=self.conv2(x)
#         x=self.max_pool1(x)
#         x=F.relu(x)
#        # x=F.dropout(x,0.2)
#         x=self.conv3(x)
#         x=self.conv4(x)
#         x=self.max_pool2(x)
#         x=F.relu(x)
#        # x = F.dropout(x, 0.2)
#
#         x=self.conv5(x)
#         x=self.conv6(x)
#         x=self.max_pool3(x)
#         x=F.relu(x)
#         #x = F.dropout(x, 0.2)
#
#         x=self.conv7(x)
#         x=self.conv8(x)
#         x=self.max_pool4(x)
#         x=F.relu(x)
#        # x=F.dropout(x,0.2)
#
#         x=x.view(x_shape,-1)
#
#         x=self.fn1(x)
#         x=F.relu(x)
#
#         x=self.fn2(x)
#         x=F.relu(x)
#
#         x=F.dropout(x,0.2)
#         x=self.fn3(x)
#         x=F.log_softmax(x,dim=1)
#         return x
class Block(torch.nn.Module):
    def __init__(self, inchannel, outchannel, res=True):
        super(Block, self).__init__()
        self.res = res     # 是否带残差连接
        self.left = torch.nn.Sequential(
            torch.nn.Conv2d(inchannel, outchannel, kernel_size=(3,3), padding=1, bias=False),
            torch.nn.BatchNorm2d(outchannel),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(outchannel, outchannel, kernel_size=(3,3), padding=1, bias=False),
            torch.nn.BatchNorm2d(outchannel),
        )
        if inchannel != outchannel:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(inchannel, outchannel, kernel_size=(1,1), bias=False),
                torch.nn.BatchNorm2d(outchannel),
            )
        else:
            self.shortcut = torch.nn.Sequential()

        self.relu = torch.nn.Sequential(
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.left(x)
        if self.res:
            out += self.shortcut(x)
        out = self.relu(out)
        return out


class VGG_Model(torch.nn.Module):
    def __init__(self, cfg=[64, 'M', 128,  'M', 256, 'M', 512, 'M'], res=True):
        super(VGG_Model, self).__init__()
        self.res = res       # 是否带残差连接
        self.cfg = cfg       # 配置列表
        self.inchannel = 3   # 初始输入通道数
        self.futures = self.make_layer()
        # 构建卷积层之后的全连接层以及分类器：
        self.classifier = torch.nn.Sequential(torch.nn.Dropout(0.4),            # 两层fc效果还差一些
                                        torch.nn.Linear(4 * 512, 10), )   # fc，最终Cifar10输出是10类

    def make_layer(self):
        layers = []
        for v in self.cfg:
            if v == 'M':
                layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(Block(self.inchannel, v, self.res))
                self.inchannel = v    # 输入通道数改为上一层的输出通道数
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.futures(x)
        # view(out.size(0), -1): change tensor size from (N ,H , W) to (N, H*W)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


if config["device"]=="GPU":
    device = torch.device("cuda:0")
elif config["device"]=="CPU":
    device = torch.device("CPU")
elif config["device"]=="ADAPTIVE":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    raise "Without this option, please select one in the following GPU, CPU and ADAPTIVE and fill in the previous CONFIG dictionary"
print("使用的设备为:",device)
model=VGG_Model()
summary(model, input_size=(32,3, 32, 32))
model.to(device)
loss1=torch.nn.CrossEntropyLoss()

#训练模型
optimizer = optim.Adam(model.parameters())#优化函数
def train_model(model,device,train_loader,optimizer,pbar,train_accuracy,test_loss):
    model.train()#模型训练
    for batch_index,(data ,target) in enumerate(train_loader):#一批中的一个，（图片，标签）
        data,target = data.to(device),target.to(device)#部署到DEVICE上去
        optimizer.zero_grad()#梯度初始化为0
        #print(target)
        output = model(data.to(torch.float32))#训练后的结果
        # print(target.shape)
        # print(output.shape)
     #   print(output.shape,target.shape)
        loss =loss1(output,target.long())#多分类计算损失函数
        loss.backward()#反向传播 得到参数的梯度参数值
        optimizer.step()#参数优化
        if batch_index % 300 == 0:  # 每3000个打印一次
            pbar.set_postfix(loss='%.4f' % float(loss), acc=float(train_accuracy),test_loss=test_loss)
        pbar.update(1)
def test_model(model,device,text_loader):
    model.eval()#模型验证
    correct = 0.0#正确
    Accuracy = 0.0#正确率
    test_loss = 0.0
    with torch.no_grad():#不会计算梯度，也不会进行反向传播
        for data,target in text_loader:
            data,target = data.to(device),target.to(device)#部署到device上
            output = model(data.to(torch.float32))#处理后的结果
            test_loss += F.cross_entropy(output,target.long()).item()#计算测试损失之和
            pred = output.argmax(dim=1)#找到概率最大的下标（索引）
            correct += pred.eq(target.view_as(pred)).sum().item()#累计正确的次数
        test_loss /= len(test_loader.dataset)#损失和/数据集的总数量 = 平均loss
        Accuracy = 100.0*correct / len(text_loader.dataset)#正确个数/数据集的总数量 = 正确率

    return Accuracy,test_loss
acc=0
test_loss=0
for epoch in range(1,config["epoch"]+1):
    with tqdm(total=len(train_loader),
              desc='Epoch {}/{}'.format(epoch, config["epoch"])) as pbar:
        train_model(model,device,train_loader,optimizer,pbar,train_accuracy=acc,test_loss=test_loss)
        Accuracy,test_loss=test_model(model,device,test_loader)
        acc=Accuracy
        test_loss=test_loss
torch.save(model.state_dict(),'model.ckpt')#保存为model.ckpt
