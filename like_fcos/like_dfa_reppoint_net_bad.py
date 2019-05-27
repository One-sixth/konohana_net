"""
吃显存过于惊人，闲置。。。
批量大小为6时，吃内存接近12G
"""


from model_utils_torch import *


norm = nn.BatchNorm2d
norm_args = {'eps': 1e-8, 'momentum': 0.9}


class BottleneckBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, act, **kwargs):
        super().__init__()
        self.shortcut = Conv2D(in_ch, out_ch, 3, 2, 'same', None, bias=norm, norm_kwargs=norm_args, use_fixup_init=True)
        self.conv1 = Conv2D(in_ch, out_ch, 1, 1, 'same', act, bias=norm, norm_kwargs=norm_args, use_fixup_init=True)
        self.conv2 = Conv2D(out_ch, out_ch, 3, 2, 'same', act, bias=norm, norm_kwargs=norm_args, use_fixup_init=True)

    def forward(self, x):
        y2 = self.shortcut(x)
        y = self.conv1(x)
        y = self.conv2(y)
        y = y + y2
        return y

class ResBlockA(nn.Module):
    def __init__(self, in_ch, out_ch, stride, act, **kwargs):
        super().__init__()
        self.conv1 = Conv2D(in_ch, out_ch, 3, 1, 'same', act, bias=norm, norm_kwargs=norm_args, use_fixup_init=True)
        self.conv2 = Conv2D(out_ch, out_ch, 3, 1, 'same', act, groups=8, bias=norm, norm_kwargs=norm_args, use_fixup_init=True)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return x + y


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        act = nn.LeakyReLU(0.1)
        # 416x416
        self.emb_conv1 = Conv2D(3, 8, 1, 1, 'same', act, bias=norm, norm_kwargs=norm_args, use_fixup_init=True)
        self.conv2 = Conv2D(8, 16, 3, 1, 'same', act, bias=norm, norm_kwargs=norm_args, use_fixup_init=True)

        # tower1
        #------------------------------------------
        # 208x208
        self.bt1a1 = BottleneckBlock(16, 48, 2, act)
        self.gb1a2 = group_block(48, 48, 1, act, ResBlockA, 3)

        # 104x104
        self.bt1b1 = BottleneckBlock(48, 96, 2, act)
        self.gb1b2 = group_block(96, 96, 1, act, ResBlockA, 3)

        # 52x52
        self.bt1c1 = BottleneckBlock(96, 192, 2, act)
        self.gb1c2 = group_block(192, 192, 1, act, ResBlockA, 3)

        # tower2
        #------------------------------------------

        self.reduce_dim_conv2a = Conv2D(48 + 192, 48, 1, 1, 'same', act, bias=norm, norm_kwargs=norm_args, use_fixup_init=True)

        # 104x104
        self.bt2a1 = BottleneckBlock(48, 48, 2, act)
        self.gb2a2 = group_block(48, 48, 1, act, ResBlockA, 3)

        self.reduce_dim_conv2b = Conv2D(48 + 96, 48, 1, 1, 'same', act, bias=norm, norm_kwargs=norm_args, use_fixup_init=True)

        # 52x52
        self.bt2b1 = BottleneckBlock(48, 96, 2, act)
        self.gb2b2 = group_block(96, 96, 1, act, ResBlockA, 3)

        self.reduce_dim_conv2c = Conv2D(96+192, 96, 1, 1, 'same', act, bias=norm, norm_kwargs=norm_args, use_fixup_init=True)

        # 26x26
        self.bt2c1 = BottleneckBlock(96, 192, 2, act)
        self.gb2c2 = group_block(192, 192, 1, act, ResBlockA, 3)

        # tower3
        # ------------------------------------------

        self.reduce_dim_conv3a = Conv2D(48+192, 48, 1, 1, 'same', act, bias=norm, norm_kwargs=norm_args, use_fixup_init=True)

        # 52x52
        self.bt3a1 = BottleneckBlock(48, 48, 2, act)
        self.gb3a2 = group_block(48, 48, 1, act, ResBlockA, 3)

        self.reduce_dim_conv3b = Conv2D(48+96, 48, 1, 1, 'same', act, bias=norm, norm_kwargs=norm_args, use_fixup_init=True)

        # 26x26
        self.bt3b1 = BottleneckBlock(48, 96, 2, act)
        self.gb3b2 = group_block(96, 96, 1, act, ResBlockA, 3)

        self.reduce_dim_conv3c = Conv2D(96+192, 96, 1, 1, 'same', act, bias=norm, norm_kwargs=norm_args, use_fixup_init=True)

        # 13x13
        self.bt3c1 = BottleneckBlock(96, 192, 2, act)
        self.gb3c2 = group_block(192, 192, 1, act, ResBlockA, 3)

        # decoder
        # ------------------------------------------

        self.decoder_conv1a = Conv2D(192, 192, 1, 1, 'same', act, bias=norm, norm_kwargs=norm_args, use_fixup_init=True)

        self.decoder_conv2a = Conv2D(96, 192, 1, 1, 'same', act, bias=norm, norm_kwargs=norm_args, use_fixup_init=True)
        self.decoder_conv2b = Conv2D(192, 192, 1, 1, 'same', act, bias=norm, norm_kwargs=norm_args, use_fixup_init=True)

        self.decoder_conv3a = Conv2D(48, 192, 1, 1, 'same', act, bias=norm, norm_kwargs=norm_args, use_fixup_init=True)
        self.decoder_conv3b = Conv2D(96, 192, 1, 1, 'same', act, bias=norm, norm_kwargs=norm_args, use_fixup_init=True)
        self.decoder_conv3c = Conv2D(192, 192, 1, 1, 'same', act, bias=norm, norm_kwargs=norm_args, use_fixup_init=True)

        # out
        # ------------------------------------------
        self.out_conv1 = Conv2D(192, 192, 1, act=act, bias=norm, norm_kwargs=norm_args)
        self.out_conv2 = Conv2D(192, 1+2*9+30, 1, act=None, bias=True)


    def forward(self, x):

        y = self.emb_conv1(x)
        y = self.conv2(y)

        # tower1
        # ------------------------------------------
        y1a = self.bt1a1(y)
        y1a = self.gb1a2(y1a)

        y1b = self.bt1b1(y1a)
        y1b = self.gb1b2(y1b)

        y1c = self.bt1c1(y1b)
        y1c = self.gb1c2(y1c)

        # tower2
        # ------------------------------------------
        y2a = F.interpolate(y1c, scale_factor=4)
        y2a = torch.cat([y2a, y1a], dim=1)
        y2a = self.reduce_dim_conv2a(y2a)

        y2a = self.bt2a1(y2a)
        y2a = self.gb2a2(y2a)

        y2b = torch.cat([y2a, y1b], dim=1)
        y2b = self.reduce_dim_conv2b(y2b)

        y2b = self.bt2b1(y2b)
        y2b = self.gb2b2(y2b)

        y2c = torch.cat([y2b, y1c], dim=1)
        y2c = self.reduce_dim_conv2c(y2c)

        y2c = self.bt2c1(y2c)
        y2c = self.gb2c2(y2c)

        # tower3
        # ------------------------------------------
        y3a = F.interpolate(y2c, scale_factor=4)
        y3a = torch.cat([y3a, y2a], dim=1)
        y3a = self.reduce_dim_conv3a(y3a)

        y3a = self.bt3a1(y3a)
        y3a = self.gb3a2(y3a)

        y3b = torch.cat([y3a, y2b], dim=1)
        y3b = self.reduce_dim_conv3b(y3b)

        y3b = self.bt3b1(y3b)
        y3b = self.gb3b2(y3b)

        y3c = torch.cat([y3b, y2c], dim=1)
        y3c = self.reduce_dim_conv3c(y3c)

        y3c = self.bt3c1(y3c)
        y3c = self.gb3c2(y3c)

        # decoder
        # ------------------------------------------

        yd1 = self.decoder_conv1a(y1c)

        yd2 = self.decoder_conv2a(y2b)

        yd3 = self.decoder_conv2b(y2c)
        yd3 = F.interpolate(yd3, scale_factor=2.)

        yd2 = yd2 + yd3

        yd4 = self.decoder_conv3a(y3a)

        yd5 = self.decoder_conv3b(y3b)
        yd5 = F.interpolate(yd5, scale_factor=2.)

        yd6 = self.decoder_conv3c(y3c)
        yd6 = F.interpolate(yd6, scale_factor=4.)

        yd3 = yd4 + yd5 + yd6

        yd = yd1 + yd2 + yd3

        y = self.out_conv1(yd)
        y = self.out_conv2(y)

        return y


if __name__ == '__main__':
    net = Net()
    img = torch.ones(6, 3, 416, 416)
    out = net(img)
    print(out.shape)
