from model_utils_torch import *
import config.a_config as cfg


class NetTail(nn.Module):

    def __init__(self, img_hw=(416, 416)):
        super().__init__()
        self.img_hw = nn.Parameter(torch.tensor(img_hw, dtype=torch.float32))
        self.scale = nn.Parameter(torch.ones(1, 4, 1, 1), True)
        self.bias = nn.Parameter(torch.zeros(1, 4, 1, 1), True)

    def heatmap_process(self, heatmap):
        return heatmap.sigmoid()

    def ious_process(self, ious):
        return ious.sigmoid()

    def coord_process(self, coord, box_scale):
        # 新增缩放级别
        box_scale = torch.exp(box_scale)

        grid_hw = coord.shape[2:]
        # 加0.5，代表以格子中心为相对坐标
        grid_y = torch.arange(0, grid_hw[0], 1, dtype=coord.dtype, device=coord.device) + 0.5
        grid_x = torch.arange(0, grid_hw[1], 1, dtype=coord.dtype, device=coord.device) + 0.5
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x)
        grid_y = grid_y[None, None]
        grid_x = grid_x[None, None]
        grid_yx = torch.cat([grid_y, grid_x], 1)
        grid_yxyx = grid_yx.repeat(coord.shape[0], 2, 1, 1)

        # bx9x2x52x52
        # 得到这几个点的最大yx，和最小yx，为包围框的角点
        coord = coord * box_scale * self.scale
        boxes_top_left = -coord[:, :2] + self.bias[:, :2]
        boxes_bottom_right = coord[:, 2:] + self.bias[:, 2:]
        boxes = torch.cat([boxes_top_left, boxes_bottom_right], dim=1)
        boxes = boxes + grid_yxyx
        # 现在boxes内坐标为格子坐标，单位为格子长度

        grid_hwhw = grid_hw + grid_hw
        boxes_percent = boxes / torch.tensor(grid_hwhw, dtype=coord.dtype, device=coord.device).reshape(1, 4, 1, 1)
        # 这是百分比坐标
        img_hw = self.img_hw.repeat(2).reshape(1, 4, 1, 1)
        boxes_real = boxes_percent * img_hw
        # 这是像素坐标
        return boxes_real

    def classes_process(self, classes):
        return classes.sigmoid()

    def forward(self, x):
        heatmap = x[:, :1]
        ious = x[:, 1: 2]
        coords = x[:, 2: 2+4]
        coords_scale = x[:, 6: 6+1]
        classes = x[:, 7: 7+cfg.full_classes_len]

        heatmap = self.heatmap_process(heatmap)
        ious = self.ious_process(ious)
        coords = self.coord_process(coords, coords_scale)
        classes = self.classes_process(classes)
        y = torch.cat([heatmap, ious, coords, classes], dim=1)
        return y


if __name__ == '__main__':
    detector = NetTail((416, 416))
    a = torch.randint(-1, 5, size=(6, 1 + 1 + 4 + 1 + cfg.full_classes_len, 26, 26), dtype=torch.float32)
    boxes = detector(a)
    print(boxes.shape)
