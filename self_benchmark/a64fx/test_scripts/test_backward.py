import torch
from torch import nn
from torch.autograd import Function
import pdb


class AddModel(nn.Module):
    def __init__(self, n):
        super(AddModel, self).__init__()
        # tensor长度
        self.n = n
        # 定义可训练参数a和b
        self.a = nn.Parameter(torch.Tensor(self.n))
        self.b = nn.Parameter(torch.Tensor(self.n))
        # 正态分布初始化参数a和b
        self.a.data.normal_(mean=0.0, std=1.0)
        self.b.data.normal_(mean=0.0, std=1.0)

    def forward(self):
        # 求a^2与b^2
        a2 = torch.square(self.a)
        b2 = torch.square(self.b)
        c = a2 + b2
        # print(c)
        # 调用自定义cuda算子对两个平方数求和
        d = AddModelFunction.apply(c, b2, self.n)
        return d

class AddModelFunction(Function):
    @staticmethod
    def forward(ctx, a, b, n):
        c = torch.sqrt(a**2 + b**2)
        d = c + a
        ctx.save_for_backward(a, c)
        print(ctx.needs_input_grad)
        return d

    @staticmethod
    def backward(ctx, grad_output):
        a, c = ctx.saved_tensors
        print(c)
        return (grad_output, grad_output, None)

n = 1024
print("Initializing model...")
# 定义模型
model = AddModel(n)
# 将模型中所有参数拷贝到GPU端
model.to(device="cpu")
print("Initializing optimizer...")
# 定义优化器
opt = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(500):
    # 清空优化器缓存
    opt.zero_grad()
    # 前向传播
    output = model()
    if epoch % 25 == 0:
        print(output.requires_grad, output.grad_fn)
    # 求loss
    loss = output.sum()
    # 反向传播
    loss.backward()
    # 更新参数
    opt.step()
    if epoch % 25 == 0:
        print("epoch {:>3d}: loss = {:>8.3f}".format(epoch, loss))

