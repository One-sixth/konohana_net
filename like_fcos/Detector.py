"""
暂时没有分类，只有热图和框
"""

from model_utils_torch import *
from like_fcos.like_fcos_net import Net
from like_fcos.NetTail import NetTail


class Detector(nn.Module):
    def __init__(self, img_hw, use_res2block=False):
        super().__init__()
        self.net = Net(use_res2block=use_res2block)
        self.det = NetTail(img_hw)

    def forward(self, x):
        y = self.net(x)
        y = self.det(y)
        return y


if __name__ == '__main__':
    net = Detector((360, 640)).cuda()
    print_params_size2(net)
    imgs = torch.ones(3, 3, 360, 640).cuda()
    boxes = net(imgs)
    print(boxes.shape)
