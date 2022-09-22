#y=2*x**2*sin(4*x)
#求y-2*x**2*sin(4*x)什么时候为0
#求2和4
import torch
import math
import matplotlib.pyplot as plt
from torch.autograd import Variable
#设置学习率
"""
画出曲线
"""
lr=0.0001
x1=[torch.tensor(i) for i in range(100)]
y1=[torch.tensor(2*i+4) for i in x1]
plt.plot(x1, y1)
plt.show()
#生成待优化的张量a，b，x和函数y
a=torch.tensor(torch.randn(1),requires_grad=True)
b=torch.tensor(torch.randn(1),requires_grad=True)
for i in range(1000):
    b_grads=torch.zeros(1)
    a_grads=torch.zeros(1)
    for k in range(len(x1)):
        x=x1[k]
        y=y1[k]

        loss=(a*x+b- y)**2
        grads=[-(2/len(x1))*x*(y-((a*x)+b)),-(2/len(x1)*(y-((a*x)+b)))]
        #梯度更新
        b_grads= b_grads+grads[1]/len(x1)
        a_grads = a_grads + grads[0]/len(x1)
    a=a-lr*b_grads
    b=b-lr*a_grads
print(a,b)