"""
模型格式，俩个塔
"""

from model_utils_torch import *
import config.a_config as cfg


norm = nn.BatchNorm2d
norm_args = {'eps': 1e-8, 'momentum': 0.9}

conv_kwargs = {'bias': norm, 'norm_kwargs': norm_args, 'use_fixup_init': True}


class ResBlockA(nn.Module):
    def __init__(self, in_ch, out_ch, stride, act, scale=1, ker_sz=5, **kwargs):
        super().__init__()
        self.stride = stride
        if stride == 1 or in_ch != out_ch:
            self.shortcut = Conv2D(in_ch, out_ch, 1, stride, 'same', None, **conv_kwargs)
        else:
            self.shortcut = nn.Identity()
        inter_ch = int(out_ch * scale)
        self.conv1 = Conv2D(in_ch, inter_ch, 1, 1, 'same', act, **conv_kwargs)
        self.conv2 = DwConv2D(inter_ch, 1, ker_sz, stride, 'same', act, **conv_kwargs)
        self.conv3 = Conv2D(inter_ch, out_ch, 1, 1, 'same', None, **conv_kwargs)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        if self.stride == 1:
            y2 = self.shortcut(x)
            y = y2 + y
        return y


class Res2BlockA(nn.Module):
    def __init__(self, in_ch, out_ch, stride, act, scale=1, n_level=4, ker_sz=5, **kwargs):
        super().__init__()
        self.stride = stride
        self.n_level = n_level
        if stride == 1 or in_ch != out_ch:
            self.shortcut = Conv2D(in_ch, out_ch, 1, stride, 'same', None, **conv_kwargs)
        else:
            self.shortcut = nn.Identity()
        inter_ch = int(out_ch * scale)
        self.conv1 = Conv2D(in_ch, inter_ch, 1, 1, 'same', act, **conv_kwargs)
        if self.stride == 1:
            self.conv2_group = nn.ModuleList([DwConv2D(inter_ch // n_level, 1, ker_sz, stride, 'same', act, **conv_kwargs) for _ in range(n_level)])
        else:
            self.conv2_group = DwConv2D(inter_ch, 1, ker_sz, stride, 'same', act, **conv_kwargs)
        self.conv3 = Conv2D(inter_ch, out_ch, 1, 1, 'same', None, **conv_kwargs)

    def forward(self, x):
        y = self.conv1(x)
        # res2block
        if self.stride == 1:
            yi = torch.chunk(y, self.n_level, 1)
            yo = []
            for i in range(self.n_level):
                if i == 0:
                    y = yi[i]
                    y = self.conv2_group[i](y)
                    yo.append(y)
                else:
                    y = yi[i] + yo[i - 1]
                    y = self.conv2_group[i](y)
                    yo.append(y)
            y = torch.cat(yo, 1)
        else:
            y = self.conv2_group(y)
        #
        y = self.conv3(y)
        if self.stride == 1:
            y2 = self.shortcut(x)
            y = y2 + y
        return y


class Net(nn.Module):

    def __init__(self, use_res2block=False):
        super().__init__()
        act = nn.LeakyReLU(0.1, True)

        rb_type = ResBlockA
        if use_res2block:
            rb_type = Res2BlockA

        # 416x416
        self.emb_conv1 = Conv2D(3, 12, 1, 1, 'same', act, **conv_kwargs)
        self.gb1 = group_block(12, 24, 2, act, rb_type, 2, scale=2)
        # 208x208
        self.gb2 = group_block(24, 48, 2, act, rb_type, 3, scale=2)
        # 104x104
        self.gb3 = group_block(48, 96, 2, act, rb_type, 6, scale=2)
        # 52x52
        self.gb4 = group_block(96, 144, 2, act, rb_type, 4, scale=2)
        # 26x26
        self.gb5 = group_block(144, 192, 2, act, rb_type, 4, scale=2)
        # 13x13
        self.gb6 = group_block(192, 192, 2, act, rb_type, 4, scale=2)
        # 7x7

        # tower2
        self.rd_conv1 = Conv2D(96+144+192+192, 256, 1, 1, 'same', act, **conv_kwargs)
        self.gb7 = group_block(256, 220, 1, act, rb_type, 3, scale=2)

        self.out_heatmap_conv1 = Conv2D(220, 220, 1, 1, 'same', act, **conv_kwargs)
        self.out_heatmap_conv2 = Conv2D(220, 220, 1, 1, 'same', act, **conv_kwargs)
        self.out_heatmap_conv3 = Conv2D(220, 220, 1, 1, 'same', act, **conv_kwargs)
        self.out_heatmap_conv4 = Conv2D(220, 1+1, 1, 1, 'same', None, bias=True, use_fixup_init=True)
        # 这里预测2个值，信心度和中心度

        self.out_coord_conv1 = Conv2D(220, 220, 1, 1, 'same', act, **conv_kwargs)
        self.out_coord_conv2 = Conv2D(220, 220, 1, 1, 'same', act, **conv_kwargs)
        self.out_coord_conv3 = Conv2D(220, 220, 1, 1, 'same', act, **conv_kwargs)
        self.out_coord_conv4 = Conv2D(220, 4+1, 1, 1, 'same', None, bias=True, use_fixup_init=True)
        # 这里预测5个值，4个是上左下右距离，1个是缩放倍数

        self.out_classes_conv1 = Conv2D(220, 220, 1, 1, 'same', act, **conv_kwargs)
        self.out_classes_conv2 = Conv2D(220, 220, 1, 1, 'same', act, **conv_kwargs)
        self.out_classes_conv3 = Conv2D(220, 220, 1, 1, 'same', act, **conv_kwargs)
        self.out_classes_conv4 = Conv2D(220, cfg.full_classes_len, 1, 1, 'same', None, bias=True, use_fixup_init=True)
        # 预测30个类，部分是无效类

    def forward(self, x):

        y = self.emb_conv1(x)

        y = self.gb1(y)
        y = self.gb2(y)
        y = y1 = self.gb3(y)
        # 52x52
        y = y2 = self.gb4(y)
        # 26x26
        y = y3 = self.gb5(y)
        # 13x13
        y = y4 = self.gb6(y)
        # 7x7

        # tower2
        y2 = resize_ref(y2, y1)
        y3 = resize_ref(y3, y1)
        y4 = resize_ref(y4, y1)

        y = torch.cat([y1, y2, y3, y4], 1)

        y = self.rd_conv1(y)
        y = self.gb7(y)

        heatmap = self.out_heatmap_conv1(y)
        heatmap = self.out_heatmap_conv2(heatmap)
        heatmap = self.out_heatmap_conv3(heatmap)
        heatmap = self.out_heatmap_conv4(heatmap)

        coord = self.out_coord_conv1(y)
        coord = self.out_coord_conv2(coord)
        coord = self.out_coord_conv3(coord)
        coord = self.out_coord_conv4(coord)

        classes = self.out_classes_conv1(y)
        classes = self.out_classes_conv2(classes)
        classes = self.out_classes_conv3(classes)
        classes = self.out_classes_conv4(classes)

        y = torch.cat([heatmap, coord, classes], dim=1)

        return y


if __name__ == '__main__':
    net = Net().cuda()
    print_params_size2(net)
    img = torch.ones(3, 3, 360, 640).cuda()
    out = net(img)
    print(out.shape)
    loss = (out - 3).mean()
    loss.backward()
    print(loss)
