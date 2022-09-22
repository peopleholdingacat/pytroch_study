import torch
from torch.utils.data import  Dataset,DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import *
import os
import gzip
batch_size=32
EPOCHS=100
"""
#torch的继承的Dataset类里面需要实现三个函数1.len,2.getitem,3.add
torch.utils.data.Dataset的源码如下:
    class Dataset(object):
        An abstract class representing a Dataset.
        All other datasets should subclass it. All subclasses should override
        ``__len__``, that provides the size of the dataset, and ``__getitem__``,
        supporting integer indexing in range from 0 to len(self) exclusive.
       
     
        def __getitem__(self, index):
            raise NotImplementedError
     
        def __len__(self):
            raise NotImplementedError
     
        def __add__(self, other):
            return ConcatDataset([self, other])
    可知：我们需要在自定义的数据集类中继承Dataset类，同时还需要实现两个方法：
    __len__方法，能够实现通过全局的len()方法获取其中的元素个数
    __getitem__方法，能够通过传入索引的方式获取数据，例如通过dataset[i]获取其中的第i条数据
"""
"""
如果不想通过图片形式导入可以使用TORCHVISION的类里面的MNIST数据集导入
"""
# datasets=MNIST(root='./MNIST',train=True,download=True)
# train_loader = DataLoader(dataset=datasets,
#                            batch_size=5,
#                            shuffle=True)

"""
以下方法是重写dataset类的方法
"""


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

class Dataset_Mnist:
    def __init__(self):
        self.test_path_image="./MNIST/MNIST/raw/t10k-images-idx3-ubyte"
        self.test_path_label="./MNIST/MNIST/raw/t10k-labels-idx1-ubyte"
        self.train_path_image="./MNIST/MNIST/raw/train-images-idx3-ubyte"
        self.train_path_label="./MNIST/MNIST/raw/train-labels-idx1-ubyte"
        self.datset_train=None
        self.datset_test=None
        self.y_test = None
        self.y_train = None
        self.x_test = None
        self.x_train = None
        self.len = None
        self.dataset_get()
        self.len=len(self.y_train)

        pass
    def __len__(self):
        if self.len==None:
            raise "Dataset length information was not obtained"
        return self.len
    def __getitem__(self, item):
        #if self.y_train!=None:
        return self.x_train[item], self.y_train[item]
        #raise "Make the dataset before iterating"
        #return self.x_train[item],self.y_train[item],self.x_test[item],self.y_test[item]

    def dataset_get(self):
        with open(self.train_path_label, 'rb') as lbpath:  # rb表示的是读取二进制数据
            self.y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

        with  open(self.train_path_image, 'rb') as imgpath:
            self.x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(self.y_train), 28, 28)
        with open(self.test_path_label, 'rb') as lbpath:  # rb表示的是读取二进制数据
            self.y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
        with  open(self.test_path_image, 'rb') as imgpath:
            self.x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(self.y_test), 28, 28)

        return self.x_train,self.y_train,self.x_test,self.y_test
    def DataLoder_make(self):

        train_loader = DataLoader(
            dataset=datasets(self.x_train,self.y_train),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            dataset=datasets(self.x_test, self.y_test),
            batch_size=batch_size,
            shuffle=True,
        )
        print(train_loader)
        print(test_loader)
        return train_loader,test_loader
    def show_pic(self,img,label,is_only=True):
       """
       :param img: 图片的数组形式(一个batch)batch的大小必须为32
       :param label: 图片的标签
       :return:
       """
       if is_only:
           #将一组图片组合成一张
           img=torch.unsqueeze(img,0)
           img = img.numpy().transpose(1,2,0)
           plt.imshow(img)
           plt.xlabel(label.data)
           plt.show()
       else:
           plt.figure(figsize=(85, 45))
           for i in range(32):
               plt.subplot(8, 4, i+1)
               plt.imshow(img[i])
               #plt.xlabel(label[i].data)
           plt.tight_layout()  # 自动调整子批次参数，使子批次适合图形区域
           plt.show()
dataset=Dataset_Mnist()
train_loader,test_loader=dataset.DataLoder_make()
# """
# 展示图片
# """
# images, labels = next(iter(train_loader))
# print(images.shape)
# print(labels.shape)
# dataset.show_pic(images,labels,is_only=False)

class minist_module(torch.nn.Module):
    def __init__(self):
        super().__init__()  # 继承父类的函数
        self.conv1=torch.nn.Conv2d(1,10,kernel_size=(5,5))
        self.conv2=torch.nn.Conv2d(10,1,kernel_size=(3,3))
        self.fn1=torch.nn.Linear(100,1000)
        self.fn2=torch.nn.Linear(1000,500)
        self.fn3 = torch.nn.Linear(500, 10)
    def forward(self,input):
        input_size=input.size(0)
        input=torch.unsqueeze(input,dim=1)
        x=self.conv1(input)
        x=F.relu(x)
        x=F.max_pool2d(x,2,2)
        x=self.conv2(x)
        x=F.relu(x)
       # print(x.shape)
        x=x.view(input_size,-1)

        x=self.fn1(x)
        x=F.relu(x)
        x=self.fn2(x)
        x=F.relu(x)
        x=self.fn3(x)

        x=F.log_softmax(x,dim=1)
        return x
#模型传入cuda上

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用的设备为:",device)
from torchinfo import summary

model=minist_module()
summary(model, input_size=(32, 28, 28))
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
for epoch in range(1,EPOCHS+1):
    with tqdm(total=len(train_loader),
              desc='Epoch {}/{}'.format(epoch, EPOCHS)) as pbar:
        train_model(model,device,train_loader,optimizer,pbar,train_accuracy=acc,test_loss=test_loss)
        Accuracy,test_loss=test_model(model,device,test_loader)
        acc=Accuracy
        test_loss=test_loss
torch.save(model.state_dict(),'model.ckpt')#保存为model.ckpt
