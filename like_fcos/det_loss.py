# coding=utf-8
from model_utils_torch import *
from tensorboardX import SummaryWriter
import config.a_config as cfg


class DetLoss:
    def __init__(self):
        super().__init__()
        self.writer = SummaryWriter('./logs')

    @staticmethod
    def calc_iou(out_boxes: torch.Tensor, label_boxes: torch.Tensor):
        eps = 1e-8

        # out_boxes = xywh_to_x1y1x2y2(out_boxes)
        # label_boxes = xywh_to_x1y1x2y2(label_boxes)

        x_max = torch.where(out_boxes[:, 2] < label_boxes[:, 2], out_boxes[:, 2], label_boxes[:, 2])
        x_min = torch.where(out_boxes[:, 0] > label_boxes[:, 0], out_boxes[:, 0], label_boxes[:, 0])

        y_max = torch.where(out_boxes[:, 3] < label_boxes[:, 3], out_boxes[:, 3], label_boxes[:, 3])
        y_min = torch.where(out_boxes[:, 1] > label_boxes[:, 1], out_boxes[:, 1], label_boxes[:, 1])

        area_width = (x_max - x_min).clamp_min(0.)
        area_height = (y_max - y_min).clamp_min(0.)

        area = area_width * area_height
        all_area = out_boxes[:, 2] * out_boxes[:, 3] + label_boxes[:, 2] * label_boxes[:, 3] - area
        all_area = all_area + eps

        iou = area / all_area
        iou = iou[:, None, :, :]

        return iou

    @staticmethod
    def calc_heat_loss(out_heatmap, label_heatmap, label_bin, weight=1.):
        # 现在改为，中心度大于0.5的都是1
        l1 = (label_bin - out_heatmap).abs()
        l2 = torch.where(label_heatmap > 0.5, l1*2, l1)
        return l2 * weight

    @staticmethod
    def calc_ious_loss(out_ious, label_ious, label_heatmap, iou_loss_calc_thresh, weight=1.):
        # l1 = (out_ious - label_ious).abs()
        # l2 = l1 * (label_heatmap > iou_loss_calc_thresh).float()
        # 现在改成逼近中心度
        l1 = (label_heatmap - out_ious).abs()
        # 中心度大于0.5的部分，要求loss更强
        l2 = torch.where(label_heatmap > 0.5, l1*2, l1)
        return l2 * weight

    @staticmethod
    def calc_coords_loss(out_boxes, label_boxes, label_heatmap, label_bin, weight=1.):
        # y1x1y2x2
        l1 = (label_boxes - out_boxes).abs()
        # sqrt(0) 会导致反传时梯度nan
        l2 = torch.sqrt(l1+1e-8)
        # loss 强度随中心度变化
        l2 = l2.mean(dim=1, keepdim=True) * label_bin * label_heatmap
        return l2 * weight

    @staticmethod
    def calc_classes_loss(out_classes, label_classes, label_heatmap, label_bin, classes_mask, weight=1.):
        # 这里修改为类似conf的方式
        n_classes = classes_mask.sum()
        # big_loss_class_weight = n_classes / 4
        # big_loss_class_weight = 2
        classes_mask = torch.reshape(classes_mask, [1, -1, 1, 1])
        # y1x1y2x2
        l1 = (label_classes - out_classes).abs() * classes_mask
        l2 = l1
        # 这里正类loss翻倍
        l2 = torch.where(label_classes > 0.5, l2 * 2, l2)
        # 这里中心部分loss再次翻倍
        l2 = torch.where(label_bin > 0.5, l2 * 2, l2)
        # loss 强度随中心度变化
        # 这里不能用mean，因为会把被mask的无效class算进去
        l2 = (l2.sum(dim=1, keepdim=True) / n_classes) # * label_bin * label_heatmap
        return l2 * weight

    def calc_loss(self, net_out, label, classes_mask, global_step):

        # 开始计算iou loss的中心度阀值
        iou_loss_calc_thresh = 0.2

        out_heatmap = net_out[:, :1]
        out_ious = net_out[:, 1:2]
        out_coords = net_out[:, 2:2+4]
        out_classes = net_out[:, 6:6+cfg.full_classes_len]

        label_heatmap = label[:, :1]
        label_coords = label[:, 1:1+4]
        label_classes = label[:, 5:5+cfg.full_classes_len]

        # 使用一个阀值，控制包围框中心附近的区域才计算阀值
        conf_thresh = 0.6
        label_bin = (label_heatmap > conf_thresh).float()

        # 计算iou
        with torch.no_grad():
            label_ious = self.calc_iou(out_coords, label_coords)

        cell_heat_loss = self.calc_heat_loss(out_heatmap, label_heatmap, label_bin)
        cell_ious_loss = self.calc_ious_loss(out_ious, label_ious, label_heatmap, iou_loss_calc_thresh)
        cell_coord_loss = self.calc_coords_loss(out_coords, label_coords, label_heatmap, label_bin)
        cell_classes_loss = self.calc_classes_loss(out_classes, label_classes, label_heatmap, label_bin, classes_mask)

        # each_sample_count = ((label_heatmap > 0).float().sum(dim=(2, 3)) + 1e-4)
        each_sample_count = ((label_bin > 0).float().sum(dim=(2, 3)) + 1e-1)
        # 现在这里是每个样本的分开的loss
        each_sample_heat_loss = cell_heat_loss.mean(dim=(2, 3))
        each_sample_ious_loss = cell_ious_loss.mean(dim=(2, 3))
        # coord_loss 因为有的格子是不计算的loss的，所以除数用正例来计算，+1是因为偶尔会出现奇怪的大数，来源估计是因为sqrt(1e-8)
        each_sample_coords_loss = cell_coord_loss.sum(dim=(2, 3)) / each_sample_count
        each_sample_classes_loss = cell_classes_loss.mean(dim=(2, 3)) # / each_sample_count

        loss = each_sample_heat_loss + each_sample_ious_loss + each_sample_coords_loss + each_sample_classes_loss

        loss = loss.mean()

        with torch.no_grad():

            if global_step % 10 == 0:
                # 记录
                # 因为使用的是 fcos 的方式，所有这里用中心度大于等于0.5为正样本
                label_pos_heatmap = label_bin
                out_pos_heatmap = (out_heatmap > conf_thresh).float()
                n_cell = torch.prod(torch.tensor(out_heatmap.shape, dtype=torch.float, device=out_heatmap.device))
                # 真样本数量，加 1e-8 为了防止除0
                n_label_heat_real = label_pos_heatmap.sum() + 1e-8
                # 预测为真的正样本数量
                n_out_heat_true_pos = (out_pos_heatmap * label_pos_heatmap).sum()
                # 预测为假的正样本数量
                n_out_heat_false_pos = ((1-out_pos_heatmap) * label_pos_heatmap).sum()
                # 预测为真的负样本数量
                n_out_heat_true_neg = (out_pos_heatmap * (1-label_pos_heatmap)).sum()
                # 预测为假的负样本数量
                n_out_heat_false_neg = ((1-out_pos_heatmap) * (1-label_pos_heatmap)).sum()
                # 下面为百分比
                n_out_heat_true_pos_percent = n_out_heat_true_pos / n_label_heat_real
                n_out_heat_false_pos_percent = n_out_heat_false_pos / n_label_heat_real
                n_out_heat_true_neg_percent = n_out_heat_true_neg / (n_cell - n_label_heat_real)
                n_out_heat_false_neg_percent = n_out_heat_false_neg / (n_cell - n_label_heat_real)

                pos_ious = label_ious * label_pos_heatmap

                # 平均iou
                mean_real_ious = pos_ious.sum() / label_pos_heatmap.sum()
                # 预测为真的正样本中的平均iou
                mean_ious_true_real = (pos_ious * (out_pos_heatmap * label_pos_heatmap)).sum() / (n_out_heat_true_pos + 1e-8)

                self.writer.add_scalar('heat/n_out_heat_true_pos', n_out_heat_true_pos, global_step)
                self.writer.add_scalar('heat/n_out_heat_false_pos', n_out_heat_false_pos, global_step)
                self.writer.add_scalar('heat/n_out_heat_true_neg', n_out_heat_true_neg, global_step)
                self.writer.add_scalar('heat/n_out_heat_false_neg', n_out_heat_false_neg, global_step)
                self.writer.add_scalar('heat/heat_max_diff', cell_heat_loss.max(), global_step)
                self.writer.add_scalar('heat/heat_min_diff', cell_heat_loss.min(), global_step)
                self.writer.add_scalar('heat/heat_mean_diff', cell_heat_loss.mean(), global_step)

                self.writer.add_scalar('heat_percent/n_out_heat_true_pos_percent', n_out_heat_true_pos_percent, global_step)
                self.writer.add_scalar('heat_percent/n_out_heat_false_pos_percent', n_out_heat_false_pos_percent, global_step)
                self.writer.add_scalar('heat_percent/n_out_heat_true_neg_percent', n_out_heat_true_neg_percent, global_step)
                self.writer.add_scalar('heat_percent/n_out_heat_false_neg_percent', n_out_heat_false_neg_percent, global_step)

                self.writer.add_scalar('iou/mean_real_ious', mean_real_ious, global_step)
                self.writer.add_scalar('iou/mean_ious_true_real', mean_ious_true_real, global_step)
                self.writer.add_scalar('iou/max_iou', pos_ious.max(), global_step)
                self.writer.add_scalar('iou/max_iou_diff', cell_ious_loss.max(), global_step)
                self.writer.add_scalar('iou/mean_iou_diff', each_sample_ious_loss.mean(), global_step)

                self.writer.add_scalar('coord/coord_max_diif', cell_coord_loss.max(), global_step)

                self.writer.add_scalar('class/classes_max_diif', cell_classes_loss.max(), global_step)

                self.writer.add_scalar('all/each_sample_heat_loss', each_sample_heat_loss.sum(), global_step)
                self.writer.add_scalar('all/each_sample_ious_loss', each_sample_ious_loss.sum(), global_step)
                self.writer.add_scalar('all/each_sample_coords_loss', each_sample_coords_loss.sum(), global_step)
                self.writer.add_scalar('all/each_sample_classes_loss', each_sample_classes_loss.sum(), global_step)

                self.writer.add_scalar('loss', loss, global_step)

        return loss

    def __call__(self, net_out, label, classes_mask, global_step):
        return self.calc_loss(net_out, label, classes_mask, global_step)
